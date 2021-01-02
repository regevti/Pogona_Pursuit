from datetime import datetime, timedelta
from flask import Flask, render_template, Response, request
import re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil import parser as date_parser
import yaml

import config
from utils import to_integer

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


class NoDataException(Exception):
    """No data"""


@dataclass
class ExperimentAnalyzer:
    start_date: datetime
    end_date: datetime
    bug_types: str = None
    animal_id: int = None
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
        for exp_path in Path(self.experiment_dir).glob('*'):
            try:
                info_path = exp_path / 'experiment.yaml'
                if not info_path.exists() or not self.is_in_date_range(exp_path):
                    continue
                info = self.get_experiment_info(info_path)
                if not self.is_experiment_match_conditions(info):
                    continue

                trial_data = self.get_trials_data(exp_path, info)
                res_df.append(trial_data)

            except NoDataException:
                pass

            except Exception as exc:
                self.log(f'Error loading {exp_path.name}; {exc}')

        if len(res_df) > 0:
            res_df = pd.concat(res_df)
            res_df.rename(columns={'name': 'experiment'}, inplace=True)
            res_df = self.group_by_experiment_and_trial(res_df)
            res_df = self.filter_trials_by_metric(res_df)

        return res_df

    def get_trials_data(self, exp_path: Path, info: dict) -> pd.DataFrame:
        experiment_info = {k: v for k, v in info.items() if not re.match(r'block\d+', k)}
        trials = []
        for trial_path in exp_path.rglob('trial*'):
            if not trial_path.is_dir():
                continue
            if self.exists_folder_labels(trial_path):
                continue
            try:
                block_id, block_data = self.get_trial_block(trial_path, info)
                if not block_data:
                    print(f'block{block_id} has no info data')
                    continue
                if not self.is_block_match_conditions(block_data):
                    continue

                trial_df = self.get_screen_touches(trial_path)
                trial_meta = dict()
                trial_meta['trial'] = int(re.match(r'trial(\d+)', trial_path.stem)[1])
                trial_meta['block'] = block_id
                trial_meta['temperature'] = self.get_temperature(trial_path)
                trial_meta['trial_start'] = self.get_trial_start(trial_path, info, trial_meta['trial'])
                trial_meta['block_type'] = block_data.get('block_type', 'bugs')
                for k, v in trial_meta.items():
                    trial_df[k] = v if len(trial_df) > 1 else [v]
                for key, value in block_data.items():
                    if key in trial_df.columns:
                        continue
                    elif isinstance(value, list):
                        value = ','.join(value)
                    trial_df[key] = value
                trials.append(trial_df)
            except Exception as exc:
                self.log(f'Error loading trial {trial_path}; {exc}')

        if len(trials) < 1:
            raise NoDataException('No trials to concatenate')
        trials = pd.concat(trials)

        for key, value in experiment_info.items():
            trials[key] = value

        return trials

    @staticmethod
    def get_trial_block(trial_path: Path, info: dict):
        block_id = 1
        m_block = re.match(r'block(\d+)', trial_path.parent.name)
        if m_block:
            block_id = int(m_block[1])
        return block_id, info.get(f'block{block_id}', {})

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

    def get_trial_start(self, trial_path: Path, info: dict, trial_id: int) -> (datetime, None):
        """Get start time for a trial based on trajectory csv's first record or if not exists, based on calculating
        trial start using meta data such as trial_duration, ITI, etc.."""
        def _calculate_trial_start_from_meta():
            extra_time_recording = info.get('extra_time_recording', 0)
            block_id, block_data = self.get_trial_block(trial_path, info)
            time2block = 0
            for i in range(1, block_id):
                prev_block_data = info.get(f'block{i}', {})
                n_trials = prev_block_data.get('num_trials', 1)
                time2block += prev_block_data['trial_duration'] * n_trials + prev_block_data['iti'] * (n_trials - 1) + \
                    extra_time_recording * n_trials * 2
            if 'trial_duration' not in block_data:
                print(block_data)
            time2trials = block_data['trial_duration'] * (trial_id - 1) + block_data['iti'] * (trial_id - 1) + \
                          extra_time_recording * trial_id + extra_time_recording * (trial_id - 1) + time2block
            exp_time = self.get_experiment_time(info)
            return exp_time + timedelta(seconds=time2trials)

        traj_path = trial_path / 'bug_trajectory.csv'
        if not traj_path.exists():
            self.log(f'No trajectory file in {trial_path}')
        try:
            res = pd.read_csv(traj_path, parse_dates=['time'], index_col=0).reset_index(drop=True)
            assert len(res) > 0
        except Exception:
            return _calculate_trial_start_from_meta()

        if res['time'].dtype.name == 'object':
            res['time'] = pd.to_datetime(res['time'], unit='ms')
        return res['time'][0]

    def exists_folder_labels(self, trial_path: Path):
        file_labels = ['climbing', 'no_video']
        for label in file_labels:
            if (trial_path / label).exists():
                self.log(f'trial {trial_path} is labelled as {label}')
                return True

    @staticmethod
    def group_by_experiment_and_trial(res_df: pd.DataFrame) -> pd.DataFrame:
        exp_group = group(res_df)
        exp_df = exp_group[['animal_id']].first()
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

        META_COLS = ['bug_speed', 'block_type', 'movement_type', 'reward_bugs', 'temperature']
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
            exp_date = str(exp_path).split('_')[-1]
            exp_date = date_parser.parse(exp_date)
            return self.start_date <= exp_date <= self.end_date
        except Exception as exc:
            self.log(f'Error parsing experiment {exp_path}; {exc}')

    @staticmethod
    def get_experiment_time(info: dict) -> datetime:
        return date_parser.parse(info.get('name').split('_')[-1])

    def is_experiment_match_conditions(self, info):
        return self._is_match_conditions(['animal_id'], info)

    def is_block_match_conditions(self, block_info):
        return self._is_match_conditions(['bug_types', 'movement_type', 'block_type'], block_info)

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
    def get_experiment_info(p: Path):
        with p.open('r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return data

    def log(self, msg):
        self.errors.append(msg)
        print(msg)

    def render(self, df):
        """Convert experiments DF to styled HTML"""
        cm = sns.light_palette("green", as_cmap=True)
        # Set CSS properties for th elements in dataframe
        th_props = [
            ('font-size', '14px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('color', '#6d6d6d'),
            ('background-color', '#f7f7f9'),
            ("border", "2px solid black"),
            ("white-space", "nowrap")
        ]
        # Set CSS properties for td elements in dataframe
        td_props = [
            ('font-size', '14px'),
            ('text-align', 'center'),
            ("border", "2px solid black"),
            ("white-space", "nowrap")
        ]
        # Set table styles
        styles = [
            dict(selector="th", props=th_props),
            dict(selector="td", props=td_props),
            dict(selector="tr:hover", props=[("background-color", "#ffff99")])
        ]
        # .applymap(color_high, ['num_of_strikes']) \
        html = df.style.background_gradient(cmap=cm, subset=['num_of_strikes', 'strike_accuracy', 'reward_accuracy',
                                                           'temperature']) \
            .format({'strike_accuracy': "{:.0%}",
                     'reward_accuracy': "{:.0%}",
                     'time_to_first_strike': "{:.1f}",
                     'bug_speed': "{:.0f}",
                     'movement_type': "{}",
                     'reward_bugs': "{}",
                     'temperature': "{:.2f}"}, na_rep="-") \
            .set_table_styles(styles).render()
        html += f'<br/><div>Total Experiments: {len(df.index.get_level_values(0).unique())}</div>'
        html += f'<div>Total Trials: {len(df)}</div>'
        errors = "\n".join(self.errors)
        if errors:
            html += f'<br/><h4>Errors:</h4><pre>{errors}</pre>'
        return html


def remove_tz(col):
    return pd.to_datetime(col, utc=True).dt.tz_convert('utc').dt.tz_localize(None)


def localize_dt(col: pd.Series):
    return col.dt.tz_localize('utc').dt.tz_convert('Asia/Jerusalem').dt.strftime('%d-%m-%Y %H:%M:%S')


def group(df: pd.DataFrame):
    """Group-by experiment and trial"""
    return df.groupby(['experiment', 'block', 'trial'])


def to_percent(x: pd.Series) -> pd.Series:
    x.fillna(0, inplace=True)
    return x
    # return x.map(lambda c: c * 100)