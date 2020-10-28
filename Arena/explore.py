import re
import pandas as pd
from pathlib import Path
import datetime
from dateutil import parser as date_parser
from dataclasses import dataclass
import config
from utils import to_integer


@dataclass
class ExperimentAnalyzer:
    start_date: datetime.datetime
    end_date: datetime.datetime
    bug_types: str = None
    animal_id: int = None
    movement_type: str = None
    num_of_strikes: int = 0

    def __post_init__(self):
        if isinstance(self.start_date, str):
            self.start_date = date_parser.parse(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = date_parser.parse(self.end_date + 'T23:59')

    def get_experiments(self) -> pd.DataFrame:
        res_df = []
        print(f'experiments folder: {config.explore_experiment_dir}')
        for exp_path in Path(config.explore_experiment_dir).glob('*'):
            info_path = exp_path / 'experiment.log'
            if not info_path.exists() or not self.is_in_date_range(exp_path):
                continue
            info = self.get_experiment_info(info_path)
            if not self.is_match_conditions(info):
                continue
            trial_data = self.get_trial_data(exp_path, info)
            res_df.append(trial_data)

        if len(res_df) > 0:
            res_df = pd.concat(res_df)
            res_df.rename(columns={'name': 'experiment'}, inplace=True)
            res_df = self.group_by_experiment_and_trial(res_df)
            res_df = self.filter_trials_by_metric(res_df)

        return res_df

    def get_trial_data(self, exp_path: Path, info: dict) -> pd.DataFrame:
        trials = []
        for trial_path in exp_path.glob('trial*'):
            trial = self.get_screen_touches(trial_path)
            trial_num = int(trial_path.stem.split('trial')[-1])
            trial['trial'] = trial_num if len(trial) > 0 else [trial_num]
            trials.append(trial)

        trials = pd.concat(trials)
        for key, value in info.items():
            trials[key] = value

        return trials

    @staticmethod
    def get_screen_touches(trial_path: Path) -> pd.DataFrame:
        res = pd.DataFrame()
        touches_path = trial_path / 'screen_touches.csv'
        if touches_path.exists():
            res = pd.read_csv(touches_path, parse_dates=['time'], index_col=0).reset_index(drop=True)

        return res

    @staticmethod
    def group_by_experiment_and_trial(res_df: pd.DataFrame) -> pd.DataFrame:
        group = lambda x: x.groupby(['experiment', 'trial'])
        to_percent = lambda x: (x.fillna(0) * 100).map('{:.1f}%'.format)
        exp_group = group(res_df)
        exp_df = exp_group[['animal_id']].first()

        num_strikes = exp_group['is_hit'].count()
        exp_df['num_of_strikes'] = num_strikes
        exp_df['strike_accuracy'] = to_percent(group(res_df.query('is_hit==1'))['is_hit'].count() / num_strikes)
        exp_df['reward_accuracy'] = to_percent(
            group(res_df.query('is_reward_bug==1'))['is_reward_bug'].count() / num_strikes)

        exp_df['exp_time'] = [date_parser.parse(z[0].split('_')[-1]) for z in exp_group.count().index.to_list()]
        exp_df['exp_time'] = exp_df['exp_time'].dt.tz_localize('utc').dt.tz_convert('Asia/Jerusalem')
        trial_ids = exp_df.index.get_level_values(1)
        first_strikes = (exp_group['time'].first().dt.tz_convert('Asia/Jerusalem') - exp_df['exp_time']).astype(
            'timedelta64[s]')
        trial_start = exp_group['trial_duration'].first() * (trial_ids - 1) + \
                      exp_group['iti'].first() * (trial_ids - 1) + \
                      config.extra_time_recording * trial_ids + config.extra_time_recording * (trial_ids - 1)
        exp_df['time_to_first_strike'] = first_strikes - trial_start
        # exp_cols = [x for x in experiment_arguments() if x in res_df.columns and x != 'animal_id']
        # exp_df = pd.concat([exp_df, exp_group[exp_cols].first()], axis=1)
        exp_df = exp_df.sort_values(by=['exp_time', 'trial'])
        exp_df.drop(columns=['num_trials'], inplace=True, errors='ignore')

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
            print(f'Error parsing experiment {exp_path}; {exc}')

    def is_match_conditions(self, info):
        attrs2check = ['bug_types', 'animal_id', 'movement_type']
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
        """Load the experiment info file to data frame"""
        info = {}
        int_fields = ['iti', 'trial_duration', 'animal_id']
        with p.open('r') as f:
            m = re.finditer(r'(?P<key>\w+): (?P<value>\S+)', f.read())
            for r in m:
                key = r.groupdict()['key'].lower()
                value = r.groupdict()['value']
                if key in int_fields:
                    value = to_integer(value)
                info[key] = value
        return info
