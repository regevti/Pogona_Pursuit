from utils import get_datetime_string, mkdir, is_debug_mode
from arena import record
from cache import CacheColumns, RedisCache
from mqtt import MQTTClient, LOG_TOPICS, SUBSCRIPTION_TOPICS
from pathlib import Path
import pandas as pd
import time
import json

mqtt_client = MQTTClient()
EXPERIMENTS_DIR = 'experiments'
REWARD_TYPES = ['end_trial', 'always']


class Experiment:
    def __init__(self, name: str, animal_id: str, cache: RedisCache, cameras, trial_duration=60, num_trials=1, iti=10,
                 bug_type=None, bug_speed=None, movement_type=None, is_use_predictions=False, time_between_bugs=None,
                 reward_type='end_trial'):
        self.experiment_name = f'{name}_{get_datetime_string()}'
        self.animal_id = animal_id
        self.cache = cache
        self.num_trials = num_trials
        self.trial_duration = trial_duration
        self.iti = iti
        self.current_trial = 1
        self.cameras = cameras
        self.is_use_predictions = is_use_predictions
        self.bug_type = bug_type
        self.bug_speed = bug_speed
        self.movement_type = movement_type
        self.time_between_bugs = time_between_bugs
        self.reward_type = reward_type

    def __str__(self):
        output = ''
        for obj in ['experiment_name', 'animal_id', 'num_trials', 'cameras', 'trial_duration', 'iti', 'reward_type',
                    'bug_type', 'bug_speed', 'movement_type', 'is_use_predictions', 'time_between_bugs']:
            output += f'{obj}: {getattr(self, obj)}\n'
        return output

    def start(self):
        mkdir(self.experiment_path)
        self.save_experiment_log()
        self.init_experiment_cache()
        mqtt_client.publish_command('led_light', 'on')
        mqtt_client.publish_command('hide_bugs')
        for i in range(self.num_trials):
            if not self.cache.get(CacheColumns.EXPERIMENT_NAME):
                print('experiment was stopped')
                break
            self.current_trial = i + 1
            if i != 0:
                time.sleep(self.iti)
            self.run_trial()

            yield self.trial_summary

        self.end_experiment()
        yield str(self)

    def run_trial(self):
        mkdir(self.trial_path)
        mqtt_client.publish_command('init_bugs', self.bug_options)
        self.cache.set(CacheColumns.EXPERIMENT_TRIAL_ON, True, timeout=self.trial_duration)
        self.cache.set(CacheColumns.EXPERIMENT_TRIAL_PATH, self.trial_path, timeout=self.trial_duration)
        if not is_debug_mode():
            acquire_stop = {'record_time': self.trial_duration}
            if self.is_always_reward:
                acquire_stop.update({'trial_alive': True})
            record(cameras=self.cameras, output=self.videos_path, is_auto_start=True, cache=self.cache,
                   is_use_predictions=self.is_use_predictions, **acquire_stop)
        else:
            time.sleep(self.trial_duration)
        mqtt_client.publish_command('hide_bugs')
        self.cache.delete(CacheColumns.EXPERIMENT_TRIAL_ON)

    def end_experiment(self):
        self.cache.delete(CacheColumns.EXPERIMENT_NAME)
        self.cache.delete(CacheColumns.EXPERIMENT_PATH)
        if self.is_always_reward:
            self.cache.delete(CacheColumns.ALWAYS_REWARD)

        mqtt_client.publish_command('led_light', 'off')

    def save_experiment_log(self):
        with open(f'{self.experiment_path}/experiment.log', 'w') as f:
            f.write(str(self))

    def init_experiment_cache(self):
        self.cache.set(CacheColumns.EXPERIMENT_NAME, self.experiment_name, timeout=self.experiment_duration)
        self.cache.set(CacheColumns.EXPERIMENT_PATH, self.experiment_path, timeout=self.experiment_duration)
        if self.is_always_reward:
            self.cache.set(CacheColumns.ALWAYS_REWARD, True, timeout=self.experiment_duration)

    @property
    def trial_summary(self):
        log = f'Trial {self.current_trial}:\n'
        touches_file = Path(self.trial_path) / LOG_TOPICS.get("touch", '')
        num_hits = 0
        if touches_file.exists() and touches_file.is_file():
            touches_df = pd.read_csv(touches_file, parse_dates=['timestamp'], index_col=0).reset_index(drop=True)
            log += f'Number of touches on the screen: {len(touches_df)}\n'
            num_hits = len(touches_df.query("is_hit == True"))
            log += f'Number of successful hits: {num_hits}'
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
            'bugType': self.bug_type,
            'movementType': self.movement_type,
            'timeBetweenBugs': self.time_between_bugs if not self.is_always_reward else 0
        })

    @property
    def experiment_duration(self):
        return round((self.num_trials * self.trial_duration + (self.num_trials - 1) * self.iti) * 1.5)

    @property
    def experiment_path(self):
        return f'{EXPERIMENTS_DIR}/{self.experiment_name}'

    @property
    def trial_path(self):
        return f'{self.experiment_path}/trial{self.current_trial}'

    @property
    def videos_path(self):
        return f'{self.trial_path}/videos'

    @property
    def is_always_reward(self):
        return self.reward_type == 'always'
