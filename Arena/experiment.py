import argparse
import inspect
import json
import threading
import time
import re
from datetime import datetime
from dataclasses import dataclass, field
from multiprocessing.pool import ThreadPool
from threading import Event
from pathlib import Path
import yaml
import requests
import pandas as pd

import config
from loggers import get_logger
from cache import RedisCache, CacheColumns as cc
from utils import mkdir, Serializer, to_integer

cache = RedisCache()


@dataclass
class Experiment:
    animal_id: str
    cameras: str
    num_blocks: int = 1
    name: str = ''
    blocks: list = field(default_factory=list, repr=False)
    time_between_blocks: int = config.time_between_blocks
    is_use_predictions: bool = False
    extra_time_recording: int = config.extra_time_recording

    def __post_init__(self):
        self.name = str(self)
        blocks_ids = range(self.first_block, self.first_block + len(self.blocks))
        self.blocks = [Block(i, self.cameras, str(self), self.experiment_path,
                             extra_time_recording=self.extra_time_recording, **kwargs)
                       for i, kwargs in zip(blocks_ids, self.blocks)]
        self.start_time = datetime.now()
        self.day = self.start_time.strftime('%Y%m%d')
        self.logger = get_logger(f'Experiment {str(self)}')
        self.run_thread = None

    def __str__(self):
        return self.day

    @property
    def first_block(self):
        mkdir(self.experiment_path)
        blocks = Path(self.experiment_path).glob('block*')
        blocks = [to_integer(x.name.split('block')[-1]) for x in blocks]
        blocks = sorted([b for b in blocks if isinstance(b, int)])
        return blocks[-1] + 1 if blocks else 1

    @property
    def info(self):
        non_relevant_fields = ['self', 'blocks', 'cache']
        info = {k: getattr(self, k) for k in get_arguments(self) if k not in non_relevant_fields}
        for block in self.blocks:
            info[f'block{block.block_id}'] = block.info

        return info

    def start(self):
        """Main Function for starting an experiment"""
        def _start():
            self.logger.info('Experiment started')
            self.init_experiment_cache()
            self.turn_screen('on')

            try:
                for i, block in enumerate(self.blocks):
                    if i > 0:
                        time.sleep(self.time_between_blocks)
                    block.start()
            except EndExperimentException as exc:
                self.logger.warning('Experiment stopped externally')

            self.turn_screen('off')
            time.sleep(3)
            cache.publish_command('end_experiment')

        self.run_thread = threading.Thread(target=_start)
        self.run_thread.start()
        return str(self)

    def save(self, data):
        """Save experiment arguments"""

    def turn_screen(self, val):
        """val must be on or off"""
        try:
            requests.get(f'http://localhost:5000/display/{val}')
        except Exception as exc:
            self.logger.exception(f'Error turning off screen: {exc}')

    def init_experiment_cache(self):
        cache.set(cc.EXPERIMENT_NAME, self.name, timeout=self.experiment_duration)
        cache.set(cc.EXPERIMENT_PATH, self.experiment_path, timeout=self.experiment_duration)

    @property
    def experiment_path(self):
        return f'{config.experiments_dir}/{self.animal_id}/{self.day}'

    @property
    def experiment_duration(self):
        return sum(b.overall_block_duration for b in self.blocks) + self.time_between_blocks * (len(self.blocks) - 1)


@dataclass
class Block:
    block_id: int
    cameras: str
    experiment_name: str
    experiment_path: str
    extra_time_recording: int = config.extra_time_recording

    num_trials: int = 1
    trial_duration: int = 10
    iti: int = 10
    block_type: str = 'bugs'
    bug_types: list = field(default_factory=list)
    bug_speed: int = None
    bug_size: int = None
    is_default_bug_size: bool = True
    exit_hole: str = None
    reward_type: str = 'always'
    reward_bugs: list = None
    reward_any_touch_prob: float = 0.0

    media_url: str = ''

    movement_type: str = None
    is_anticlockwise: bool = False
    target_drift: str = ''
    bug_height: int = None
    time_between_bugs: int = None
    is_use_predictions: bool = False
    background_color: str = ''

    pool: ThreadPool = field(default=None, repr=False)
    threads_event: Event = field(default_factory=Event, repr=False)

    def __post_init__(self):
        self.logger = get_logger(f'{self.experiment_name} - Block {self.block_id}')
        if isinstance(self.bug_types, str):
            self.bug_types = self.bug_types.split(',')
        if isinstance(self.reward_bugs, str):
            self.reward_bugs = self.reward_bugs.split(',')
        elif not self.reward_bugs:
            self.logger.warning(f'No reward bugs were given, using all bug types as reward; {self.reward_bugs}')
            self.reward_bugs = self.bug_types

    @property
    def info(self):
        non_relevant_fields = ['self', 'cache', 'pool', 'threads_event', 'is_use_predictions',
                               'cameras', 'experiment_path']
        for block_type, block_type_fields in config.experiment_types.items():
            if block_type != self.block_type:
                non_relevant_fields += block_type_fields
        info = {k: getattr(self, k) for k in get_arguments(self) if k not in non_relevant_fields}
        info['start_time'] = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        info['version'] = config.version
        return info

    def start(self):
        mkdir(self.block_path)
        if self.is_always_reward:
            cache.set(cc.IS_ALWAYS_REWARD, True, timeout=self.block_duration)
        self.hide_visual_app_content()
        try:
            self.run_block()
        except EndExperimentException as exc:
            self.hide_visual_app_content()
            self.end_block()
            self.logger.warning('block stopped externally')
            raise exc

        self.logger.info(self.block_summary)

    def run_block(self):
        """Run block flow"""
        self.init_block()
        cache.publish_command('start_recording')
        self.wait(self.extra_time_recording)

        self.start_trials()
        self.wait(self.block_duration, check_visual_app_on=True)

        self.end_trials()
        self.wait(self.extra_time_recording)
        self.end_block()
        cache.publish_command('stop_recording')

    # def start_threads(self) -> ThreadPool:
    #     """Start cameras recording and temperature recording on a separate processes"""
    #     self.threads_event.set()
    #
    #     def _start_recording():
    #         self.logger.info('recording started')
    #         cache.publish_command('gaze_external', 'on')
    #         if not config.is_debug_mode:
    #             acquire_stop = {'record_time': self.block_duration, 'thread_event': self.threads_event}
    #             try:
    #                 record(cameras=self.cameras, output=self.videos_path, cache=cache,
    #                        is_use_predictions=self.is_use_predictions, **acquire_stop)
    #             except Exception as exc:
    #                 print(f'Error in start_recording: {exc}')
    #         else:
    #             self.wait(self.block_duration)
    #         cache.publish_command('gaze_external', 'off')
    #         self.logger.info('recording ended')
    #
    #     def _read_temp():
    #         ser = Serializer()
    #         print('read_temp started')
    #         while self.threads_event.is_set():
    #             try:
    #                 line = ser.read_line()
    #                 if line and isinstance(line, bytes):
    #                     m = re.search(r'Temperature is: ([\d.]+)', line.decode())
    #                     if m:
    #                         cache.publish(config.subscription_topics['temperature'], m[1])
    #             except Exception as exc:
    #                 print(f'Error in read_temp: {exc}')
    #             time.sleep(5)
    #
    #     pool = ThreadPool(processes=2)
    #     pool.apply_async(_start_recording)
    #     pool.apply_async(_read_temp)
    #
    #     return pool

    def init_block(self):
        mkdir(self.block_path)
        self.save_block_log()
        cache.publish_command('led_light', 'on')
        cache.set(cc.EXPERIMENT_BLOCK_ID, self.block_id, timeout=self.overall_block_duration)
        cache.set(cc.EXPERIMENT_BLOCK_PATH, self.block_path, timeout=self.overall_block_duration + self.iti)

    @staticmethod
    def end_block():
        cache.publish_command('led_light', 'off')
        cache.delete(cc.EXPERIMENT_BLOCK_ID)
        cache.delete(cc.EXPERIMENT_BLOCK_PATH)

    def start_trials(self):
        if self.is_media_experiment:
            cache.publish_command('init_media', self.media_options)
        else:
            cache.publish_command('init_bugs', self.bug_options)

        cache.set(cc.IS_VISUAL_APP_ON, True)
        self.logger.info(f'{self.block_type} initiated')

    def end_trials(self):
        self.hide_visual_app_content()

    def hide_visual_app_content(self):
        if self.is_media_experiment:
            cache.publish_command('hide_media')
        else:
            cache.publish_command('hide_bugs')

    def save_block_log(self):
        with open(f'{self.block_path}/info.yaml', 'w') as f:
            yaml.dump(self.info, f)
        with open(f'{self.block_path}/config.yaml', 'w') as f:
            yaml.dump(config_log(), f)

    def wait(self, duration, check_visual_app_on=False):
        """Sleep while checking for experiment end"""
        self.logger.info(f'waiting for {duration} seconds...')
        t0 = time.time()
        while time.time() - t0 < duration:
            # check for external stop of the experiment
            if not cache.get(cc.EXPERIMENT_NAME):
                raise EndExperimentException()
            # check for visual app finish (due to bug catch, etc...)
            if check_visual_app_on and not cache.get(cc.IS_VISUAL_APP_ON):
                self.logger.info('Trials ended')
                return
            time.sleep(0.1)

    @property
    def media_options(self):
        return json.dumps({
            'url': f'{config.management_url}/media/{self.media_url}'
        })

    @property
    def bug_options(self):
        return json.dumps({
            'numOfBugs': 1,
            'numTrials': self.num_trials,
            'iti': self.iti,
            'trialDuration': self.trial_duration,
            'speed': self.bug_speed,
            'bugTypes': self.bug_types,
            'rewardBugs': self.reward_bugs,
            'movementType': self.movement_type,
            'timeBetweenBugs': self.time_between_bugs,
            'isStopOnReward': self.is_always_reward,
            'isLogTrajectory': True,
            'bugSize': self.bug_size,
            'backgroundColor': self.background_color,
            'exitHole': self.exit_hole,
            'rewardAnyTouchProb': self.reward_any_touch_prob
        })

    @property
    def block_summary(self):
        log_string = f'Summary of Block {self.block_id}:\n'
        touches_file = Path(self.block_path) / config.metrics_files.get("touch", '')
        num_hits = 0
        if touches_file.exists() and touches_file.is_file():
            touches_df = pd.read_csv(touches_file, parse_dates=['time'], index_col=0).reset_index(drop=True)
            log_string += f'  Number of touches on the screen: {len(touches_df)}\n'
            num_hits = len(touches_df.query("is_hit == True"))
            log_string += f'  Number of successful hits: {num_hits}\n'
            num_hits_rewarded = len(touches_df.query("is_hit == True & is_reward_bug == True"))
            log_string += f'  Number of Rewarded hits: {num_hits_rewarded}'
        else:
            log_string += 'No screen strikes were recorded.'

        log_string += 2 * '\n'

        if num_hits and self.reward_type == 'end_trial':
            cache.publish(config.subscription_topics['reward'], '')

        return log_string

    @property
    def is_media_experiment(self):
        return self.block_type == 'media'

    @property
    def is_always_reward(self):
        return self.reward_type == 'always'

    @property
    def overall_block_duration(self):
        return self.block_duration + 2 * self.extra_time_recording

    @property
    def block_duration(self):
        return round((self.num_trials * self.trial_duration + (self.num_trials - 1) * self.iti) * 1.5)

    @property
    def block_path(self):
        return f'{self.experiment_path}/block{self.block_id}'

    @property
    def videos_path(self):
        return f'{self.block_path}/videos'


class ExperimentCache:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or config.experiment_cache_path
        mkdir(self.cache_dir)
        self.saved_caches = self.get_saved_caches()

    def load(self, name):
        path = Path(self.get_cache_path(name))
        assert path.exists(), f'experiment {name} does not exist'
        with path.open('r') as f:
            data = json.load(f)
        return data

    def save(self, data):
        name = data.get('name')
        with Path(self.get_cache_path(name)).open('w') as f:
            json.dump(data, f)

    def get_saved_caches(self):
        return list(Path(self.cache_dir).glob('*.json'))

    def get_cache_path(self, name):
        return f"{self.cache_dir}/{name}.json"


class EndExperimentException(Exception):
    """End Experiment"""


def config_log():
    """Get the config for logging"""
    drop_config_fields = ['Env', 'env']
    config_dict = config.__dict__
    for k in config_dict.copy():
        if k.startswith('__') or k in drop_config_fields:
            config_dict.pop(k)
    return config_dict


def get_arguments(cls):
    """Get the arguments of the Experiment class"""
    sig = inspect.signature(cls.__init__)
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
