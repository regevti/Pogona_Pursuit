from dotenv import load_dotenv
load_dotenv()
from utils import get_datetime_string, mkdir, is_debug_mode, turn_display_on, turn_display_off, async_call_later
from arena import record
from cache import CacheColumns, RedisCache
from mqtt import MQTTPublisher, LOG_TOPICS, SUBSCRIPTION_TOPICS, EXPERIMENT_LOG
from pathlib import Path
import pandas as pd
import asyncio
import argparse
import inspect
import time
import json

mqtt_client = MQTTPublisher()
EXPERIMENTS_DIR = 'experiments'
REWARD_TYPES = ['always', 'end_trial']
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


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
        sig = inspect.signature(Experiment.__init__)
        for obj in sig.parameters.keys():
            if obj in ['self', 'cache']:
                continue
            output += f'{obj}: {getattr(self, obj)}\n'
        return output

    @staticmethod
    def log(msg):
        print(msg)
        mqtt_client.publish_event(EXPERIMENT_LOG, msg)

    def start(self):
        self.log('>> Experiment started\n')
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

        mqtt_client.publish_command('end_experiment')
        return str(self)

    def run_trial(self):
        mkdir(self.trial_path)
        turn_display_on()
        mqtt_client.publish_command('led_light', 'on')
        mqtt_client.publish_command('init_bugs', self.bug_options)
        self.cache.set(CacheColumns.EXPERIMENT_TRIAL_ON, True, timeout=self.trial_duration)
        self.cache.set(CacheColumns.EXPERIMENT_TRIAL_PATH, self.trial_path, timeout=self.trial_duration + self.iti)
        self.log(f'>> Trial {self.current_trial} started')
        if not is_debug_mode():
            acquire_stop = {'record_time': self.trial_duration}
            if self.is_always_reward:
                acquire_stop.update({'trial_alive': True})
            record(cameras=self.cameras, output=self.videos_path, is_auto_start=True, cache=self.cache,
                   is_use_predictions=self.is_use_predictions, **acquire_stop)
        else:
            time.sleep(self.trial_duration)
        mqtt_client.publish_command('hide_bugs')
        mqtt_client.publish_command('end_trial')

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
            touches_df = pd.read_csv(touches_file, parse_dates=['timestamp'], index_col=0).reset_index(drop=True)
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


if __name__ == "__main__":
    signature = inspect.signature(Experiment.__init__)
    parser = argparse.ArgumentParser(description='Experiments')

    for k, v in signature.parameters.items():
        if k == 'self':
            continue
        default = v.default if v.default is not signature.empty else None
        annotation = v.annotation if v.annotation is not signature.empty else None
        parser.add_argument('--' + k, type=annotation, default=default)

    args = parser.parse_args()
    kwargs = dict(args._get_kwargs())

    e = Experiment(**kwargs)
    e.start()
    print(e)
