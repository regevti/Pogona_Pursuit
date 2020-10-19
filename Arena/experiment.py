import re
from dateutil import parser as date_parser
from dotenv import load_dotenv
load_dotenv()
from utils import get_datetime_string, mkdir, is_debug_mode, turn_display_on, turn_display_off, to_integer
from arena import record
from cache import CacheColumns, RedisCache
from mqtt import MQTTPublisher, LOG_TOPICS, SUBSCRIPTION_TOPICS
from logger import EXPERIMENT_LOG
from pathlib import Path
from multiprocessing import Process, Queue
import pandas as pd
import argparse
import inspect
import time
import json

mqtt_client = MQTTPublisher()
EXPERIMENTS_DIR = 'experiments'
REWARD_TYPES = ['always', 'end_trial']
EXTRA_TIME_RECORDING = 30  # seconds


class Experiment:
    def __init__(self, name: str, animal_id: str, bug_types: str, cameras, cache: RedisCache = None,
                 trial_duration: int = 60, num_trials: int = 1, iti: int = 10, bug_speed=None, movement_type=None,
                 is_use_predictions: bool = False, time_between_bugs: int = None, reward_type='end_trial',
                 reward_bugs: str = None):
        self.name = f'{name}_{get_datetime_string()}'
        self.animal_id = animal_id
        self.cache = cache or RedisCache()
        self.num_trials = num_trials
        self.trial_duration = trial_duration
        self.iti = iti
        self.current_trial = 1
        self.cameras = cameras
        self.is_use_predictions = is_use_predictions
        self.bug_types = bug_types.split(',')
        self.reward_bugs = reward_bugs.split(',') if reward_bugs else self.bug_types
        self.reward_type = reward_type
        self.bug_speed = bug_speed
        self.movement_type = movement_type
        self.time_between_bugs = time_between_bugs

    def __str__(self):
        output = ''
        for obj in experiment_arguments():
            if obj in ['self', 'cache']:
                continue
            value = getattr(self, obj)
            if isinstance(value, list):
                value = ','.join(value)
            output += f'{obj}: {value}\n'
        return output

    @staticmethod
    def log(msg):
        print(msg)
        mqtt_client.publish_event(EXPERIMENT_LOG, msg)

    def start(self):
        self.log(f'>> Experiment {self.name} started\n')
        mkdir(self.experiment_path)
        self.save_experiment_log()
        self.init_experiment_cache()
        turn_display_on()
        mqtt_client.publish_command('hide_bugs')
        for i in range(self.num_trials):
            if not self.cache.get(CacheColumns.EXPERIMENT_NAME):
                self.log('experiment was stopped')
                break
            self.current_trial = i + 1
            if i != 0:
                time.sleep(self.iti)
            self.run_trial()

            self.log(self.trial_summary)

        time.sleep(3)
        mqtt_client.publish_command('end_experiment')
        return str(self)

    def run_trial(self):
        mkdir(self.trial_path)
        turn_display_on()
        mqtt_client.publish_command('led_light', 'on')
        self.cache.set(CacheColumns.EXPERIMENT_TRIAL_ON, True, timeout=self.trial_duration)
        self.cache.set(CacheColumns.EXPERIMENT_TRIAL_PATH, self.trial_path, timeout=self.trial_duration + self.iti)
        self.log(f'>> Trial {self.current_trial} recording started')
        process = self.start_recording()
        time.sleep(EXTRA_TIME_RECORDING)
        mqtt_client.publish_command('init_bugs', self.bug_options)
        self.log(f'>> Trial {self.current_trial} bugs initiated')

        t0 = time.time()
        while time.time() - t0 < self.trial_duration:
            process.join(1)
            if not process.is_alive():
                break

        mqtt_client.publish_command('hide_bugs')
        self.log(f'>> Trial {self.current_trial} bugs stopped')
        time.sleep(EXTRA_TIME_RECORDING)
        mqtt_client.publish_command('end_trial')
        turn_display_off()
        process.join(10)
        if process.is_alive():
            process.terminate()

    def start_recording(self) -> Process:
        """Start cameras recording on a separate process"""
        def _start_recording():
            record_duration = self.trial_duration + 2 * EXTRA_TIME_RECORDING
            if not is_debug_mode():
                acquire_stop = {'record_time': record_duration}
                if self.is_always_reward:
                    acquire_stop.update({'trial_alive': True})
                record(cameras=self.cameras, output=self.videos_path, is_auto_start=True, cache=self.cache,
                       is_use_predictions=self.is_use_predictions, **acquire_stop)
            else:
                time.sleep(record_duration)

        p = Process(target=_start_recording)
        p.start()

        return p

    def save_experiment_log(self):
        with open(f'{self.experiment_path}/experiment.log', 'w') as f:
            f.write(str(self))

    def init_experiment_cache(self):
        self.cache.set(CacheColumns.EXPERIMENT_NAME, self.name, timeout=self.experiment_duration)
        self.cache.set(CacheColumns.EXPERIMENT_PATH, self.experiment_path, timeout=self.experiment_duration)

        if self.is_always_reward:
            self.cache.set(CacheColumns.ALWAYS_REWARD, True, timeout=self.experiment_duration)

    @property
    def trial_summary(self):
        log = f'Summary of Trial {self.current_trial}:\n'
        touches_file = Path(self.trial_path) / LOG_TOPICS.get("touch", '')
        num_hits = 0
        if touches_file.exists() and touches_file.is_file():
            touches_df = pd.read_csv(touches_file, parse_dates=['time'], index_col=0).reset_index(drop=True)
            log += f'  Number of touches on the screen: {len(touches_df)}\n'
            num_hits = len(touches_df.query("is_hit == True"))
            log += f'  Number of successful hits: {num_hits}\n'
            num_hits_rewarded = len(touches_df.query("is_hit == True & is_reward_bug == True"))
            log += f'  Number of Rewarded hits: {num_hits_rewarded}'
        else:
            log += 'No screen strikes were recorded.'

        log += 2 * '\n'

        if num_hits and self.reward_type == 'end_trial':
            mqtt_client.publish_event(SUBSCRIPTION_TOPICS['reward'], '')

        return log

    @property
    def bug_options(self):
        return json.dumps({
            'numOfBugs': 1,
            'speed': self.bug_speed,
            'bugTypes': self.bug_types,
            'rewardBugs': self.reward_bugs,
            'movementType': self.movement_type,
            'timeBetweenBugs': self.time_between_bugs,
            'isStopOnReward': self.is_always_reward,
            'isLogTrajectory': True
        })

    @property
    def experiment_duration(self):
        return round((self.num_trials * self.trial_duration + (self.num_trials - 1) * self.iti) * 1.5)

    @property
    def experiment_path(self):
        return f'{EXPERIMENTS_DIR}/{self.name}'

    @property
    def trial_path(self):
        return f'{self.experiment_path}/trial{self.current_trial}'

    @property
    def videos_path(self):
        return f'{self.trial_path}/videos'

    @property
    def is_always_reward(self):
        return self.reward_type == 'always'


class ExperimentAnalyzer:
    def __init__(self, start_date, end_date, bug_types=None, animal_id=None):
        self.start_date = date_parser.parse(start_date)
        self.end_date = date_parser.parse(end_date + 'T23:59')
        self.bug_types = bug_types
        self.animal_id = animal_id

    def get_experiments(self) -> pd.DataFrame:
        res_df = []
        for exp_path in Path(EXPERIMENTS_DIR).glob('*'):
            info_path = exp_path / 'experiment.log'
            if not info_path.exists() or not self.is_in_date_range(exp_path):
                continue
            info = self.get_experiment_info(info_path)
            if not self.is_match_conditions(info):
                continue
            trial_data = self.get_trial_data(exp_path, info)
            res_df.append(trial_data)

        df = pd.concat(res_df)
        if len(df) > 0:
            df.rename(columns={'name': 'experiment'}, inplace=True)
            df = self.group_by_experiment_and_trial(df)
            # df.index.name = None  # for removing index title row
        return df

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
            res = pd.read_csv(touches_path, parse_dates=['timestamp'], index_col=0).reset_index(drop=True)

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
        exp_df['reward_accuracy'] = to_percent(group(res_df.query('is_reward_bug==1'))['is_reward_bug'].count() / num_strikes)

        exp_df['time'] = [date_parser.parse(z[0].split('_')[-1]) for z in exp_group.count().index.to_list()]
        trial_ids = exp_df.index.get_level_values(1)
        first_strikes = (exp_group['timestamp'].first() - exp_df['time']).astype('timedelta64[s]')
        trial_start = exp_group['trial_duration'].first() * (trial_ids - 1) + exp_group['iti'].first() * (trial_ids - 1)
        exp_df['time_to_first_strike'] = first_strikes - trial_start

        exp_cols = [x for x in experiment_arguments() if x in res_df.columns and x != 'animal_id']
        exp_df = pd.concat([exp_df, exp_group[exp_cols].first()], axis=1)
        exp_df = exp_df.sort_values(by=['time', 'trial'])
        exp_df.drop(columns=['num_trials'], inplace=True, errors='ignore')

        return exp_df

    def is_in_date_range(self, exp_path):
        try:
            exp_date = str(exp_path).split('_')[-1]
            exp_date = date_parser.parse(exp_date)
            return self.start_date <= exp_date <= self.end_date
        except Exception as exc:
            print(f'Error parsing experiment {exp_path}; {exc}')

    def is_match_conditions(self, info):
        attrs2check = ['bug_types', 'animal_id']
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


def experiment_arguments():
    """Get the arguments of the Experiment class"""
    sig = inspect.signature(Experiment.__init__)
    return sig.parameters.keys()


def main():
    signature = inspect.signature(Experiment.__init__)
    arg_parser = argparse.ArgumentParser(description='Experiments')

    for k, v in signature.parameters.items():
        if k == 'self':
            continue
        default = v.default if v.default is not signature.empty else None
        annotation = v.annotation if v.annotation is not signature.empty else None
        arg_parser.add_argument('--' + k, type=annotation, default=default)

    args = arg_parser.parse_args()
    kwargs = dict(args._get_kwargs())

    e = Experiment(**kwargs)
    e.start()
    print(e)


if __name__ == "__main__":
    main()
