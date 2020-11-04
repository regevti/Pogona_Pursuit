import argparse
import inspect
import json
import time
import re
from dataclasses import dataclass, field
from multiprocessing.pool import ThreadPool
from threading import Event
from pathlib import Path

import pandas as pd

import config
from arena import record
from cache import CacheColumns, RedisCache
from mqtt import MQTTPublisher
from utils import datetime_string, mkdir, turn_display_on, turn_display_off, Serializer

mqtt_client = MQTTPublisher()


@dataclass
class Experiment:
    name: str
    animal_id: str
    bug_types: list
    cameras: str
    cache: RedisCache = field(default_factory=RedisCache, repr=False)
    trial_duration: int = 60
    num_trials: int = 1
    iti: int = 10
    bug_speed: int = None
    movement_type: str = None
    is_use_predictions: bool = False
    time_between_bugs: int = None
    extra_time_recording: int = config.extra_time_recording
    reward_type: str = 'end_trial'
    reward_bugs: list = None
    is_anticlockwise: bool = False
    current_trial: int = field(default=1, repr=False)
    pool: ThreadPool = field(default=None, repr=False)
    threads_event: Event = field(default_factory=Event, repr=False)

    def __post_init__(self):
        self.name = f'{self.name}_{datetime_string()}'
        if isinstance(self.bug_types, str):
            self.bug_types = self.bug_types.split(',')
        if self.reward_bugs and isinstance(self.reward_bugs, str):
            self.reward_bugs = self.reward_bugs.split(',')
        else:
            self.reward_bugs = self.bug_types

    def __str__(self):
        output = ''
        for obj in experiment_arguments():
            if obj in ['self', 'cache', 'pool', 'current_trial', 'threads_event']:
                continue
            value = getattr(self, obj)
            if isinstance(value, list):
                value = ','.join(value)
            output += f'{obj}: {value}\n'
        return output

    def start(self):
        """Main Function for starting an experiment"""
        self.log(f'>> Experiment {self.name} started\n')
        mkdir(self.experiment_path)
        self.save_experiment_log()
        self.init_experiment_cache()
        mqtt_client.publish_command('hide_bugs')
        for i in range(self.num_trials):
            try:
                self.current_trial = i + 1
                if i != 0:
                    self.wait(self.iti)
                self.run_trial()
            except EndExperimentException:
                self.hide_bugs()
                self.end_trial()
                self.log('>> experiment was stopped externally')
                return str(self)

            self.log(self.trial_summary)

        time.sleep(3)
        mqtt_client.publish_command('end_experiment')
        return str(self)

    def run_trial(self):
        """Run trial flow"""
        self.init_trial()
        self.pool = self.start_threads()
        self.wait(self.extra_time_recording)

        self.start_bugs()
        self.wait(self.trial_duration, check_bugs_on=True)

        self.hide_bugs()
        self.wait(self.extra_time_recording)
        self.end_trial()

    def start_threads(self) -> ThreadPool:
        """Start cameras recording and temperature recording on a separate processes"""
        self.threads_event.set()

        def _start_recording():
            self.trial_log('recording started')
            if not config.is_debug_mode:
                acquire_stop = {'record_time': self.overall_trial_duration} #, 'thread_event': self.threads_event
                record(cameras=self.cameras, output=self.videos_path, cache=self.cache,
                       is_use_predictions=self.is_use_predictions, **acquire_stop)
            else:
                self.wait(self.overall_trial_duration)
            self.trial_log('recording ended')

        def _read_temp():
            ser = Serializer()
            print('read_temp started')
            while self.threads_event.is_set():
                try:
                    line = ser.read_line()
                    if line and isinstance(line, bytes):
                        m = re.search(r'Temperature is: ([\d.]+)', line.decode())
                        if m:
                            mqtt_client.publish_event(config.subscription_topics['temperature'], m[1])
                except Exception as exc:
                    print(f'Error in read_temp: {exc}')
                time.sleep(5)

        pool = ThreadPool(processes=2)
        pool.apply_async(_start_recording)
        pool.apply_async(_read_temp)

        return pool

    def save_experiment_log(self):
        with open(f'{self.experiment_path}/experiment.log', 'w') as f:
            f.write(str(self))
        with open(f'{self.experiment_path}/config.log', 'w') as f:
            f.write(config_string())

    def init_experiment_cache(self):
        self.cache.set(CacheColumns.EXPERIMENT_NAME, self.name, timeout=self.experiment_duration)
        self.cache.set(CacheColumns.EXPERIMENT_PATH, self.experiment_path, timeout=self.experiment_duration)
        if self.is_always_reward:
            self.cache.set(CacheColumns.ALWAYS_REWARD, True, timeout=self.experiment_duration)

    def init_trial(self):
        mkdir(self.trial_path)
        turn_display_on()
        mqtt_client.publish_command('led_light', 'on')
        self.cache.set(CacheColumns.EXPERIMENT_TRIAL_ON, True, timeout=self.overall_trial_duration)
        self.cache.set(CacheColumns.EXPERIMENT_TRIAL_PATH, self.trial_path,
                       timeout=self.overall_trial_duration + self.iti)

    def end_trial(self):
        mqtt_client.publish_command('led_light', 'off')
        self.terminate_pool()
        self.cache.delete(CacheColumns.EXPERIMENT_TRIAL_ON)
        self.cache.delete(CacheColumns.EXPERIMENT_TRIAL_PATH)

    def start_bugs(self):
        mqtt_client.publish_command('init_bugs', self.bug_options)
        self.cache.set(CacheColumns.BUGS_ON, True)
        self.trial_log('bugs initiated')

    def hide_bugs(self):
        mqtt_client.publish_command('hide_bugs')
        mqtt_client.publish_command('end_bugs_wait')
        self.trial_log('bugs stopped')
        turn_display_off()

    def wait(self, duration, check_bugs_on=False):
        """Sleep while checking for experiment end"""
        t0 = time.time()
        while time.time() - t0 < duration:
            if not self.cache.get(CacheColumns.EXPERIMENT_NAME):
                raise EndExperimentException()
            if check_bugs_on and not self.cache.get(CacheColumns.BUGS_ON):
                self.trial_log('Reward Bug catch')
                return
            time.sleep(2)

    def terminate_pool(self):
        self.threads_event.clear()
        time.sleep(3)
        self.pool.close()
        self.pool.join()

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
            'isLogTrajectory': True,
            'isAntiClockWise': self.is_anticlockwise
        })

    @property
    def trial_summary(self):
        log = f'Summary of Trial {self.current_trial}:\n'
        touches_file = Path(self.trial_path) / config.logger_files.get("touch", '')
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
            mqtt_client.publish_event(config.subscription_topics['reward'], '')

        return log

    @staticmethod
    def log(msg):
        print(msg)
        mqtt_client.publish_event(config.experiment_topic, msg)

    def trial_log(self, msg):
        self.log(f'>> Trial {self.current_trial} {msg}')

    @property
    def experiment_duration(self):
        return round((self.num_trials * self.overall_trial_duration + (self.num_trials - 1) * self.iti) * 1.5)

    @property
    def overall_trial_duration(self):
        return self.trial_duration + 2 * self.extra_time_recording

    @property
    def experiment_path(self):
        return f'{config.experiments_dir}/{self.name}'

    @property
    def trial_path(self):
        return f'{self.experiment_path}/trial{self.current_trial}'

    @property
    def videos_path(self):
        return f'{self.trial_path}/videos'

    @property
    def is_always_reward(self):
        return self.reward_type == 'always'


class EndExperimentException(Exception):
    """End Experiment"""


def config_string():
    """Get a printable string of config"""
    drop_config_fields = ['Env', 'env']
    config_dict = config.__dict__
    for k in config_dict.copy():
        if k.startswith('__') or k in drop_config_fields:
            config_dict.pop(k)
    s = ''
    for k, v in config_dict.items():
        s += f'{k}: {v}\n'
    return s


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
