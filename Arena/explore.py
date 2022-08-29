from datetime import datetime, timedelta
from flask import Flask, render_template, Response, request, jsonify
import re
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil import parser as date_parser
import yaml

import Arena.config as config
from Arena.utils import to_integer

app = Flask(__name__)


@app.route('/')
def explore():
    return render_template('explore.html', experiment_dir=config.explore_experiment_dir)


@app.route('/experiment_results', methods=['POST'])
def experiment_results():
    data = request.json
    ea = ExperimentAnalyzer(**data)
    df = ea.get_experiments()
    if len(df) < 1:
        return Response('No experiments found')

    return Response(ea.render(df))
    # return Response(df.to_html(classes='table-responsive'))


@app.route('/animal_ids', methods=['POST'])
def animal_ids():
    experiment_dir = request.json.get('experiment_dir')
    return jsonify([d.name for d in Path(experiment_dir).glob('*') if not d.name.startswith('.')])


class NoDataException(Exception):
    """No data"""


@dataclass
class ExperimentAnalyzer:
    start_date: datetime
    end_date: datetime
    bug_types: str = None
    animal_id: str = None
    is_first_day: bool = False
    movement_type: str = None
    num_of_strikes: int = 0
    block_type: str = None
    experiment_dir: str = field(default=config.explore_experiment_dir)
    errors: list = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.start_date, str):
            self.start_date = date_parser.parse(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = date_parser.parse(self.end_date + 'T23:59')

    def get_experiments(self) -> pd.DataFrame:
        res_df = []
        print(f'experiments folder: {self.experiment_dir}')

        for animal_dir in self.animals_dirs:
            for day_dir in self.get_days(animal_dir):
                for block_dir in day_dir.glob('block*'):
                    try:
                        info_path = block_dir / 'info.yaml'
                        if not info_path.exists():
                            continue
                        info = self.get_block_info(info_path)
                        info['day'] = datetime.strptime(day_dir.name, '%Y%m%d').strftime('%d.%m.%y')
                        if not self.is_block_match_conditions(info):
                            continue
                        trial_data = self.get_trials_data(block_dir, info)
                        res_df.append(trial_data)

                    except NoDataException:
                        pass

                    except ImportError as exc:
                        self.log(f'Error loading {block_dir.name}; {exc}')

        if len(res_df) > 0:
            res_df = pd.concat(res_df)
            res_df.rename(columns={'name': 'experiment'}, inplace=True)
            res_df = self.group_by_experiment_and_trial(res_df)
            res_df = self.filter_trials_by_metric(res_df)

        return res_df

    @staticmethod
    def get_trials_data(block_path: Path, info: dict) -> pd.DataFrame:
        version = info.get('version', '1.1')
        if version.startswith('1'):
            ta = TrialsAnalyzerV1(block_path, info)
        else:
            ta = TrialsAnalyzerV2(block_path, info)

        return ta.get_data()

    @staticmethod
    def group_by_experiment_and_trial(res_df: pd.DataFrame) -> pd.DataFrame:
        exp_group = group(res_df)
        exp_df = exp_group[['trial_start']].first()
        exp_df['trial_start'] = remove_tz(exp_group['trial_start'].first())

        if 'is_hit' in res_df.columns:
            n_strikes = exp_group['is_hit'].count()
            n_hits = group(res_df.query('is_hit==1'))['is_hit'].count()
            n_hits_reward = group(res_df.query('is_hit==1&is_reward_bug==1'))['is_reward_bug'].count()
            exp_df['num_of_strikes'] = n_strikes
            exp_df['strike_accuracy'] = to_percent(n_hits / n_strikes)
            exp_df['reward_accuracy'] = to_percent(n_hits_reward / n_hits)
            first_strikes_times = remove_tz(exp_group['time'].first())
            exp_df['time_to_first_strike'] = (first_strikes_times - exp_df['trial_start']).astype('timedelta64[s]')
        else:
            exp_df = exp_df.assign(num_of_strikes=0, strike_accuracy=np.nan, reward_accuracy=np.nan,
                                   time_to_first_strike=np.nan)

        META_COLS = ['duration', 'bug_speed', 'block_type', 'movement_type', 'reward_bugs', 'temperature', 'bad_label', 'version']
        exp_cols = [x for x in META_COLS if x in res_df.columns]
        exp_df = pd.concat([exp_df, exp_group[exp_cols].first()], axis=1)
        exp_df = exp_df.sort_values(by=['trial_start', 'block', 'trial'])
        exp_df['trial_start'] = localize_dt(exp_df['trial_start'])
        exp_df.drop(columns=['num_trials', 'exp_time'], inplace=True, errors='ignore')

        return exp_df

    def filter_trials_by_metric(self, res_df):
        metrics = ['num_of_strikes']
        for metric in metrics:
            res_df = res_df.query(f'{metric} >= {getattr(self, metric)}')

        return res_df

    def is_in_date_range(self, exp_path):
        try:
            exp_date = date_parser.parse(exp_path)
            return self.start_date <= exp_date <= self.end_date
        except Exception as exc:
            self.log(f'Error parsing experiment {exp_path}; {exc}')

    def is_block_match_conditions(self, info):
        return self._is_match_conditions(['bug_types', 'movement_type', 'block_type'], info)

    def _is_match_conditions(self, attrs2check, info):
        for attr in attrs2check:
            req_value = getattr(self, attr)
            if not req_value:
                continue
            info_value = info.get(attr)
            if not info_value or \
                    (isinstance(req_value, (int, float)) and req_value != to_integer(info_value)) or \
                    (isinstance(req_value, str) and req_value not in info_value):
                return False
        return True

    @staticmethod
    def get_experiment_time(info: dict) -> datetime:
        return date_parser.parse(info.get('name').split('_')[-1])

    @staticmethod
    def get_block_info(p: Path):
        with p.open('r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return data

    def log(self, msg):
        self.errors.append(msg)
        print(msg)

    def get_days(self, animal_dir: Path) -> list:
        l = [a for a in animal_dir.glob('*') if not a.name.startswith('.')]
        if self.is_first_day:
            l.sort(key=lambda x: x.name)
            return [l[0]]
        else:
            return [day_dir for day_dir in l if self.is_in_date_range(day_dir.name)]

    @property
    def animals_dirs(self):
        print(self.animal_id)
        if self.animal_id is None or self.animal_id == 'All':
            return [p for p in Path(self.experiment_dir).glob('*') if not p.name.startswith('.')]
        else:
            return [Path(self.experiment_dir) / self.animal_id]

    def render(self, df):
        """Convert experiments DF to styled HTML"""
        cm = sns.light_palette("forestgreen", as_cmap=True)
        cm.set_bad("white")
        # Set CSS properties for th elements in dataframe
        th_props = [
            ('font-size', '14px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('color', '#6d6d6d'),
            ('background-color', '#f7f7f9'),
            ("border", "2px solid black"),
            ("white-space", "normal")
        ]
        # Set CSS properties for td elements in dataframe
        td_props = [
            ('font-size', '14px'),
            ('text-align', 'center'),
            ("border", "2px solid black"),
            ("white-space", "normal")
        ]
        styles = [
            dict(selector="th", props=th_props),
            dict(selector="td", props=td_props),
            dict(selector="tr:hover", props=[("background-color", "#ffff99")])
        ]
        df.columns = df.columns.map(lambda x: x.replace("_", " "))
        gradient_cols = ['num of strikes', 'strike accuracy', 'reward accuracy', 'temperature']
        html = df.style.background_gradient(cmap=cm, subset=gradient_cols) \
            .format({'strike accuracy': "{:.0%}",
                     'reward accuracy': "{:.0%}",
                     'duration': "{:.1f}",
                     'time to first strike': "{:.1f}",
                     'bug speed': "{:.0f}",
                     'movement type': "{}",
                     'reward bugs': "{}",
                     'temperature': "{:.2f}",
                     'bad label': "{}"
                     }, na_rep="-")\
            .set_table_styles(styles).render()
        html += f'<br/><div>Total Experiment Days: {len(df.index.get_level_values(0).unique())}</div>'
        html += f'<div>Total Blocks: {len(df.index.droplevel(2).unique())}</div>'
        html += f'<div>Total Trials: {len(df)}</div>'
        errors = "\n".join(self.errors)
        if errors:
            html += f'<br/><h4>Errors:</h4><pre>{errors}</pre>'
        return html


class TrialsAnalyzerV1:
    def __init__(self, block_path: Path, info: dict):
        self.block_path = block_path
        self.info = info
        self.errors = []

    def get_data(self):
        """Main function for getting all data from block trials"""
        experiment_info = {k: v for k, v in self.info.items() if not re.match(r'block\d+', k)}
        trials = []
        for trial_path in self.block_path.rglob('trial*'):
            if not trial_path.is_dir():
                continue
            try:
                trial_df = self.get_screen_touches(trial_path)
                for k, v in self.trial_metadata(trial_path).items():
                    trial_df[k] = v if len(trial_df) > 1 else [v]
                for key, value in self.info.items():
                    if key in trial_df.columns:
                        continue
                    elif isinstance(value, list):
                        value = ','.join(value)
                    trial_df[key] = value
                trials.append(trial_df)
            except ImportError as exc:
                self.log(f'Error loading trial {trial_path}; {exc}')

        if len(trials) < 1:
            raise NoDataException('No trials to concatenate')
        try:
            trials = pd.concat(trials)
        except Exception as exc:
            raise NoDataException(f'Unable to concat trials; {exc}; {self.block_path}')
        for key, value in experiment_info.items():
            if isinstance(value, list):
                value = ','.join(value)
            trials[key] = value

        return trials

    def trial_metadata(self, trial_path):
        trial_id = int(re.match(r'trial(\d+)', trial_path.stem)[1])
        trial_meta = dict()
        trial_meta['trial'] = trial_id
        trial_meta['block'] = int(re.match(r'block(\d+)', self.block_path.stem)[1])
        trial_meta['temperature'] = self.get_temperature(trial_path)
        trial_meta['trial_start'] = self.get_trial_times(trial_id)
        trial_meta['duration'] = self.get_trial_duration(trial_id)
        trial_meta['bad_label'] = self.exists_folder_labels(trial_path)
        return trial_meta

    def exists_folder_labels(self, trial_path: Path):
        """Look for bad labels of a trial"""
        file_labels = ['climbing', 'no_video', 'bad']
        for label in file_labels:
            if (trial_path / label).exists() or (trial_path / 'videos' / label).exists():
                # self.log(f'trial {trial_path} is labelled as {label}')
                return label

    def get_trial_times(self, trial_id):
        if trial_id in self.trials_times.index:
            return self.trials_times.loc[trial_id, 'start']

    def get_trial_duration(self, trial_id):
        if trial_id in self.trials_times.index:
            return self.trials_times.loc[trial_id, 'duration']

    @property
    @lru_cache()
    def trials_times(self):
        trials_times_path = self.block_path / 'trials_times.csv'
        res = pd.DataFrame()
        if trials_times_path.exists():
            res = pd.read_csv(trials_times_path, parse_dates=['start'], index_col=0)
        return res

    @staticmethod
    def get_screen_touches(trial_path: Path) -> pd.DataFrame:
        res = pd.DataFrame()
        touches_path = trial_path / 'screen_touches.csv'
        if touches_path.exists():
            res = pd.read_csv(touches_path, parse_dates=['time'], index_col=0).reset_index(drop=True)
        return res

    @staticmethod
    def get_temperature(trial_path: Path) -> pd.DataFrame:
        res = pd.DataFrame()
        temp_path = trial_path / 'temperature.csv'
        if not temp_path.exists():
            return np.nan
        res = pd.read_csv(temp_path, parse_dates=['time'], index_col=0).reset_index(drop=True)
        temperature = res['0'].mean()
        return temperature

    def log(self, msg):
        self.errors.append(msg)
        print(msg)


class TrialsAnalyzerV2(TrialsAnalyzerV1):
    pass


def render(df):
    """Convert experiments DF to styled HTML"""
    cm = sns.light_palette("green", as_cmap=True)
    # Set CSS properties for th elements in dataframe
    th_props = [
        ('font-size', '14px'),
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('color', '#6d6d6d'),
        ('background-color', '#f7f7f9')
    ]
    # Set CSS properties for td elements in dataframe
    td_props = [
        ('font-size', '14px'),
        ('text-align', 'center'),
    ]
    # Set table styles
    styles = [
        dict(selector="th", props=th_props),
        dict(selector="td", props=td_props)
    ]
    # .applymap(color_high, ['num_of_strikes']) \
    return df.style.background_gradient(cmap=cm, subset=['num_of_strikes', 'strike_accuracy', 'reward_accuracy',
                                                       'temperature']) \
        .format({'strike_accuracy': "{:.0%}",
                 'reward_accuracy': "{:.0%}",
                 'time_to_first_strike': "{:.1f}",
                 'temperature': "{:.2f}"}, na_rep="-") \
        .set_table_styles(styles).render()


def remove_tz(col):
    return pd.to_datetime(col, utc=True).dt.tz_convert('utc').dt.tz_localize(None)


def localize_dt(col: pd.Series):
    return col.dt.tz_localize('utc').dt.tz_convert('Asia/Jerusalem').dt.strftime('%H:%M:%S')


def group(df: pd.DataFrame):
    """Group-by experiment and trial"""
    return df.groupby(['animal_id', 'day', 'block', 'trial'])


def to_percent(x: pd.Series) -> pd.Series:
    x.fillna(0, inplace=True)
    return x
    # return x.map(lambda c: c * 100)