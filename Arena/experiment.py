import argparse
import inspect
import random
import websocket
import json
import threading
import time
from typing import Union
import humanize
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
import utils
from loggers import get_logger
from cache import RedisCache, CacheColumns as cc
from utils import mkdir, to_integer, turn_display_on, turn_display_off, run_command, get_hdmi_xinput_id, get_psycho_files
from subscribers import Subscriber, start_experiment_subscribers
from periphery_integration import PeripheryIntegrator
from db_models import ORM, Experiment as Experiment_Model


@dataclass
class Experiment:
    cam_units: dict
    animal_id: str
    cameras: dict
    exit_hole: str
    bug_types: list = field(default_factory=list)
    reward_bugs: list = field(default_factory=list)
    reward_any_touch_prob: float = 0.0
    background_color: str = ''
    num_blocks: int = 1
    name: str = ''
    blocks: list = field(default_factory=list, repr=False)
    time_between_blocks: int = config.time_between_blocks
    extra_time_recording: int = config.extra_time_recording
    is_identical_blocks: bool = False
    is_test: bool = False
    cache = RedisCache()

    def __post_init__(self):
        self.start_time = datetime.now()
        if self.is_test:
            self.animal_id = 'test'
        self.day = self.start_time.strftime('%Y%m%d')
        self.name = str(self)
        self.orm = ORM()
        blocks_ids = range(self.first_block, self.first_block + len(self.blocks))
        self.blocks = [Block(i, self.cameras, str(self), self.experiment_path, self.animal_id, self.cam_units, self.orm,
                             self.cache, bug_types=self.bug_types, reward_bugs=self.reward_bugs, exit_hole=self.exit_hole,
                             background_color=self.background_color, reward_any_touch_prob=self.reward_any_touch_prob,
                             extra_time_recording=self.extra_time_recording, **kwargs)
                       for i, kwargs in zip(blocks_ids, self.blocks)]
        self.logger = get_logger('Experiment')
        self.threads = {}
        self.experiment_stop_flag = threading.Event()
        self.init_experiment_cache()

    def __str__(self):
        return f'EXP{self.day}'

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
            self.logger.info(f'Experiment started for {humanize.precisedelta(self.experiment_duration)}'
                             f' with cameras: {",".join(self.cameras.keys())}')
            self.orm.commit_experiment(self)
            self.turn_screen('on', board=self.get_board())
            if not config.IS_CHECK_SCREEN_MAPPING:
                # if you don't check the screen's mapping, map it again in each experiment
                self.map_touchscreen_to_hdmi(is_display_on=True)

            try:
                for i, block in enumerate(self.blocks):
                    if i > 0:
                        time.sleep(self.time_between_blocks)
                    block.start()
            except EndExperimentException as exc:
                self.logger.warning(f'Experiment stopped externally; {exc}')

            self.turn_screen('off')
            time.sleep(3)
            self.orm.update_experiment_end_time()
            self.cache.publish_command('end_experiment')

        if not self.is_ready_for_experiment():
            self.stop_experiment()
            return
        self.threads['experiment_stop'] = Subscriber(self.experiment_stop_flag,
                                                     channel=config.subscription_topics['end_experiment'],
                                                     callback=self.stop_experiment)

        self.threads['main'] = threading.Thread(target=_start)
        [t.start() for t in self.threads.values()]
        return str(self)

    def is_ready_for_experiment(self):
        checks = {
            'websocket_server_on': self.is_websocket_server_on(),
            'pogona_hunter_app_up': self.is_pogona_hunter_up(),
            'reward_left': is_reward_left(self.cache),
            'touchscreen_mapped': self.is_touchscreen_mapped_to_hdmi()
        }
        if all(checks.values()):
            return True
        else:
            msg = f'Aborting experiment due to violation of {", ".join([k for k, v in checks.items() if not v])}.'
            utils.send_telegram_message(msg)
            self.logger.error(msg)

    def is_websocket_server_on(self):
        try:
            ws = websocket.WebSocket()
            ws.connect(config.websocket_url)
            return True
        except Exception:
            self.logger.error(f'Websocket server on {config.websocket_url} is dead')

    def is_pogona_hunter_up(self):
        try:
            res = requests.get(f'http://0.0.0.0:{config.POGONA_HUNTER_PORT}')
            return res.ok
        except Exception:
            self.logger.error('pogona hunter app is down')

    @staticmethod
    def get_touchscreen_device_id():
        touchscreen_device_id = get_hdmi_xinput_id()
        if not touchscreen_device_id:
            raise Exception('unable to find touch USB')
        return touchscreen_device_id

    def map_touchscreen_to_hdmi(self, is_display_on=False):
        touchscreen_device_id = self.get_touchscreen_device_id()
        cmd = f'DISPLAY="{config.ARENA_DISPLAY}" xinput map-to-output {touchscreen_device_id} HDMI-0'
        if not is_display_on:
            turn_display_on()
            time.sleep(5)
        next(run_command(cmd))

    def is_touchscreen_mapped_to_hdmi(self):
        if not config.IS_CHECK_SCREEN_MAPPING:
            return True

        touchscreen_device_id = self.get_touchscreen_device_id()

        def _check_mapped():
            # if the matrix under "Coordinate Transformation Matrix" has values different from 0,1 - that means
            # that the mapping is working
            cmd = f'DISPLAY=":0"  xinput list-props {touchscreen_device_id} | grep "Coordinate Transformation Matrix"'
            res = next(run_command(cmd)).decode()
            return any(z not in [0.0, 1.0] for z in [float(x) for x in re.findall(r'\d\.\d+', res)])

        try:
            is_mapped = _check_mapped()
            if not is_mapped:
                self.logger.info('Fixing mapping of touchscreen output')
                cmd = f'DISPLAY="{config.ARENA_DISPLAY}" xinput map-to-output {touchscreen_device_id} HDMI-0'
                self.map_touchscreen_to_hdmi()
                time.sleep(1)
                is_mapped = _check_mapped()
                if not is_mapped:
                    self.logger.error(
                        f'Touch detection is not mapped to HDMI screen\nFix by running: {cmd}')
            return is_mapped
        except Exception:
            self.logger.exception('Error in is_touchscreen_mapped_to_hdmi')

    def stop_experiment(self, *args):
        self.logger.debug('closing experiment...')
        self.experiment_stop_flag.set()
        self.cache.delete(cc.IS_VISUAL_APP_ON)
        self.cache.delete(cc.EXPERIMENT_BLOCK_ID)
        self.cache.delete(cc.EXPERIMENT_BLOCK_PATH)
        self.cache.delete(cc.IS_ALWAYS_REWARD)
        self.cache.delete(cc.EXPERIMENT_NAME)
        self.cache.delete(cc.EXPERIMENT_PATH)
        self.cache.set(cc.IS_EXPERIMENT_CONTROL_CAMERAS, False)
        self.cache.set(cc.IS_REWARD_TIMEOUT, False)
        if 'main' in self.threads:
            self.threads['main'].join()
        time.sleep(0.2)
        self.logger.info('Experiment ended')

    def save(self, data):
        """Save experiment arguments"""

    def turn_screen(self, val, board='holes'):
        """val must be on or off"""
        if config.DISABLE_ARENA_SCREEN or board == 'blank':
            return
        assert val in ['on', 'off'], 'val must be either "on" or "off"'
        try:
            if val.lower() == 'on':
                turn_display_on(board, is_test=self.is_test)
            else:
                turn_display_off(app_only=self.is_test)
            self.logger.debug(f'screen turned {val}')
        except Exception as exc:
            self.logger.exception(f'Error turning off screen: {exc}')

    def init_experiment_cache(self):
        self.cache.set(cc.EXPERIMENT_NAME, str(self), timeout=self.experiment_duration)
        self.cache.set(cc.EXPERIMENT_PATH, self.experiment_path, timeout=self.experiment_duration)
        self.cache.set(cc.IS_EXPERIMENT_CONTROL_CAMERAS, True)
        if self.is_test:
            # cancel reward in test experiments
            self.cache.set(cc.IS_REWARD_TIMEOUT, True, timeout=self.experiment_duration)
        else:
            self.cache.set(cc.IS_REWARD_TIMEOUT, False)

    @property
    def experiment_path(self):
        return f'{config.EXPERIMENTS_DIR}/{self.animal_id}/{self.day}'

    @property
    def experiment_duration(self):
        return sum(b.overall_block_duration for b in self.blocks) + self.time_between_blocks * (len(self.blocks) - 1)

    def get_board(self):
        with open('../pogona_hunter/src/config.json', 'r') as f:
            app_config = json.load(f)

        block_type = self.blocks[0].block_type
        if block_type in ['psycho', 'blank']:
            return block_type
        curr_mt = self.blocks[0].movement_type
        curr_board = None
        for board, boards_mts in app_config['boards'].items():
            if curr_mt in boards_mts:
                curr_board = board
        if not curr_board:
            raise EndExperimentException(f'unable to find board for movement type: {curr_mt}')
        return curr_board


@dataclass
class Block:
    block_id: int
    cameras: dict
    experiment_name: str
    experiment_path: str
    animal_id: str
    cam_units: dict
    orm: ORM
    cache: RedisCache
    extra_time_recording: int = config.extra_time_recording
    start_time = None
    bug_types: list = field(default_factory=list)
    num_trials: int = 1
    trial_duration: int = 10
    iti: int = 10
    block_type: str = 'bugs'
    bug_speed: int = None
    bug_size: int = None
    is_default_bug_size: bool = True
    exit_hole: str = None
    reward_type: str = 'always'
    reward_bugs: list = None
    reward_any_touch_prob: float = 0.0

    media_url: str = ''

    psycho_file: str = ''

    movement_type: str = None
    is_anticlockwise: bool = False
    target_drift: str = ''
    bug_height: int = None
    time_between_bugs: int = None
    background_color: str = ''
    periphery: PeripheryIntegrator = PeripheryIntegrator()

    def __post_init__(self):
        self.logger = get_logger(f'{self.experiment_name}-Block {self.block_id}')
        if isinstance(self.bug_types, str):
            self.bug_types = self.bug_types.split(',')
        if isinstance(self.reward_bugs, str):
            self.reward_bugs = self.reward_bugs.split(',')
        elif not self.reward_bugs:
            self.logger.debug(f'No reward bugs were given, using all bug types as reward; {self.reward_bugs}')
            self.reward_bugs = self.bug_types

        if self.is_random_low_horizontal:
            self.init_random_low_horizontal()

    @property
    def info(self):
        non_relevant_fields = ['self', 'cache', 'is_use_predictions', 'cameras', 'experiment_path', 'orm', 'cam_units',
                               'periphery']
        for block_type, block_type_fields in config.experiment_types.items():
            if block_type != self.block_type:
                non_relevant_fields += block_type_fields
        info = {k: getattr(self, k) for k in get_arguments(self) if k not in non_relevant_fields}
        info['start_time'] = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        info['version'] = config.version
        return info

    def start(self):
        self.start_time = datetime.now()
        self.logger.debug('block start')
        self.orm.commit_block(self)

        mkdir(self.block_path)
        if self.is_always_reward:
            self.cache.set(cc.IS_ALWAYS_REWARD, True, timeout=self.block_duration)
        self.hide_visual_app_content()
        try:
            self.run_block()
        except EndExperimentException as exc:
            self.hide_visual_app_content()
            self.end_block()
            self.logger.warning('block stopped externally')
            raise exc
        finally:
            self.orm.update_block_end_time()

        self.logger.info(self.block_summary)

    def run_block(self):
        """Run block flow"""
        self.init_block()
        self.wait(self.extra_time_recording, label='Extra Time Rec')

        for trial_id in range(1, self.num_trials + 1):
            if not is_reward_left(self.cache):
                utils.send_telegram_message('No reward left in feeder; stopping experiment')
                raise EndExperimentException('No reward left; stopping experiment')
            self.start_trial(trial_id)
            self.wait(self.trial_duration, check_visual_app_on=True, label=f'Trial {trial_id}')
            self.end_trial()
            self.wait(self.iti, label='ITI')

        self.wait(self.extra_time_recording, label='Extra Time Rec')
        self.end_block()

    def init_block(self):
        mkdir(self.block_path)
        self.save_block_log_files()
        self.cache.publish_command('led_light', 'on')
        self.cache.set(cc.EXPERIMENT_BLOCK_ID, self.block_id, timeout=self.overall_block_duration)
        self.cache.set(cc.EXPERIMENT_BLOCK_PATH, self.block_path, timeout=self.overall_block_duration + self.iti)
        # start cameras for experiment with their predictors and set the output dir for videos
        self.turn_cameras('on')
        # screencast
        if config.IS_RECORD_SCREEN_IN_EXPERIMENT:
            threading.Thread(target=self.record_screen).start()

        self.hold_triggers()
        for cam_name in self.cameras.keys():
            output_dir = mkdir(f'{self.block_path}/videos')
            self.cache.set_cam_output_dir(cam_name, output_dir)

    def end_block(self):
        self.cache.publish_command('led_light', 'off')
        self.cache.delete(cc.EXPERIMENT_BLOCK_ID)
        self.cache.delete(cc.EXPERIMENT_BLOCK_PATH)
        for cam_name in self.cameras.keys():
            self.cache.set_cam_output_dir(cam_name, '')
        self.hold_triggers()
        time.sleep(8)
        self.turn_cameras('off')

    def turn_cameras(self, required_state):
        """Turn on cameras if needed, and load the experiment predictors"""
        assert required_state in ['on', 'off']
        for cam_name, cu in self.cam_units.items():
            # If there are no predictors configured and camera is on
            # or writing_fps is 0 - do nothing.
            configured_predictors = cu.get_conf_predictors()
            if (not configured_predictors and cu.is_on()) or \
                    (cu.cam_config.get('writing_fps') == 0):
                continue

            t0 = time.time()
            # wait maximum 10 seconds if CU is starting or stopping
            while (cu.is_starting or cu.is_stopping) and (time.time() - t0 < 10):
                time.sleep(0.1)

            if required_state == 'on':
                if not cu.is_on():
                    cu.start(is_experiment=True, movement_type=self.movement_type)
                else:
                    cu.reload_predictors(is_experiment=True, movement_type=self.movement_type)
            else:
                cu.reload_predictors(is_experiment=False, movement_type=self.movement_type)

        t0 = time.time()
        # wait maximum 30 seconds for cameras to finish start / stop
        while any(cu.is_starting or cu.is_stopping for cu in self.cam_units.values()) and \
                (time.time() - t0 < 30):
            time.sleep(0.1)

    def start_trial(self, trial_id):
        trial_db_id = self.orm.commit_trial({
            'start_time': datetime.now(),
            'in_block_trial_id': trial_id})
        if self.block_type == 'psycho':
            self.run_psycho()
        if not self.is_blank_block:
            if self.is_media_block:
                command, options = 'init_media', self.media_options
            else:
                command, options = 'init_bugs', self.bug_options
                if self.is_random_low_horizontal:
                    options = self.set_random_low_horizontal_trial(options)
            options['trialID'] = trial_id
            options['trialDBId'] = trial_db_id
            self.cache.publish_command(command, json.dumps(options))
            self.cache.set(cc.IS_VISUAL_APP_ON, True)
            time.sleep(1)  # wait for data to be sent
        self.logger.info(f'Trial #{trial_id} started')

    def end_trial(self):
        self.hide_visual_app_content()
        time.sleep(1)

    def hide_visual_app_content(self):
        if self.is_blank_block:
            return
        if self.is_media_block:
            self.cache.publish_command('hide_media')
        else:
            self.cache.publish_command('hide_bugs')

    def save_block_log_files(self):
        with open(f'{self.block_path}/info.yaml', 'w') as f:
            yaml.dump(self.info, f)
        with open(f'{self.block_path}/config.yaml', 'w') as f:
            yaml.dump(config_log(), f)

    def wait(self, duration, check_visual_app_on=False, label=''):
        """Sleep while checking for experiment end"""
        if label:
            label = f'({label}): '
        self.logger.info(f'{label}waiting for {duration} seconds...')
        t0 = time.time()
        while time.time() - t0 < duration:
            # check for external stop of the experiment
            if not self.cache.get(cc.EXPERIMENT_NAME):
                raise EndExperimentException()
            # check for visual app finish (due to bug catch, etc...)
            if check_visual_app_on and not self.is_blank_block and not self.cache.get(cc.IS_VISUAL_APP_ON):
                self.logger.debug('Trial ended')
                return
            time.sleep(0.1)

    def run_psycho(self):
        psycho_files = get_psycho_files()
        cmd = f'cd {psycho_files[self.psycho_file]} && DISPLAY="{config.ARENA_DISPLAY}" {config.PSYCHO_PYTHON_INTERPRETER} {self.psycho_file}.py'
        self.logger.info(f'Running the following psycho cmd: {cmd}')
        next(run_command(cmd))

    def hold_triggers(self):
        try:
            self.logger.info(f'holding triggers for {config.HOLD_TRIGGERS_TIME} sec')
            self.cache.set(cc.CAM_TRIGGER_DISABLE, True, timeout=self.block_duration)
            self.periphery.cam_trigger(0)
            time.sleep(config.HOLD_TRIGGERS_TIME)
            self.periphery.cam_trigger(1)
            time.sleep(1)
            self.cache.set(cc.CAM_TRIGGER_DISABLE, False)
        except Exception as exc:
            self.logger.error(f'Error holding triggers: {exc}')

    def record_screen(self):
        filename = f'{self.block_path}/screen_record.mp4'
        next(run_command(
            f'ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab '
            f'-i :0.0+1920+0 -f pulse -i default -ac 2 -t {int(self.block_duration)} '
            f'''-vf "drawtext=fontfile=/Windows/Fonts/Arial.ttf: 
            text='%{{localtime}}':x=30:y=30:fontcolor=red:fontsize=30" {filename}''', is_debug=False)
        )

    def init_random_low_horizontal(self, max_strikes=30):
        speeds = [2, 4, 6, 8]
        speed_strikes_count = {k: 0 for k in speeds}
        with self.orm.session() as s:
            exps = s.query(Experiment_Model).filter_by(animal_id=self.animal_id).all()
            for e in exps:
                for b in e.blocks:
                    if b.movement_type != 'random_low_horizontal' or b.bug_speed not in speeds:
                        continue
                    speed_strikes_count[b.bug_speed] = speed_strikes_count.get(b.bug_speed, 0) + len(b.strikes)
        available_speeds = [s for s in speeds if speed_strikes_count[s] < max_strikes]
        self.bug_speed = random.choice(available_speeds)
        self.logger.info(f'random_low_horizontal starts with bug_speed={self.bug_speed}; '
                         f'speeds strikes: {speed_strikes_count}')

    @staticmethod
    def set_random_low_horizontal_trial(options):
        options['exitHole'] = random.choice(['bottomLeft', 'bottomRight'])
        options['movementType'] = 'low_horizontal'
        return options

    @property
    def media_options(self) -> dict:
        return {
            'trialID': 1,  # default value, changed in init_media,
            'url': f'{config.management_url}/media/{self.media_url}'
        }

    @property
    def bug_options(self) -> dict:
        return {
            'numOfBugs': 1,
            'trialID': 1,  # default value, changed in init_bugs
            'trialDBId': 1, # default value, changed in init_bugs
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
        }

    @property
    def block_summary(self):
        log_string = f'Summary of Block {self.block_id}:\n'
        touches_file = Path(self.block_path) / config.experiment_metrics.get("touch", {}).get('csv_file', 'touch.csv')
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
            self.cache.publish(config.subscription_topics['reward'], '')

        return log_string

    @property
    def is_media_block(self):
        return self.block_type == 'media'

    @property
    def is_blank_block(self):
        return self.block_type == 'blank'

    @property
    def is_random_low_horizontal(self):
        return self.movement_type == 'random_low_horizontal'

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


def is_reward_left(cache):
    is_left = False
    try:
        is_left = sum([int(x) for x in cache.get(cc.REWARD_LEFT)]) > 0
    except Exception:
        pass
    return is_left


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


def start_trial():
    arg_parser = argparse.ArgumentParser(description='Experiments')
    arg_parser.add_argument('-m', '--movement_type', default='random')
    arg_parser.add_argument('--exit_hole', default='bottomLeft', choices=['bottomLeft', 'bottomRight'])
    arg_parser.add_argument('--speed', type=int, default=5)
    args = arg_parser.parse_args()
    cache_ = RedisCache()
    options = {
        'numOfBugs': 1,
        'trialID': 1,  # default value, changed in init_bugs
        'trialDBId': 1, # default value, changed in init_bugs
        'numTrials': 1,
        'iti': 30,
        'trialDuration': 30,
        'speed': args.speed,
        'bugTypes': ['cockroach'],
        'rewardBugs': [],
        'movementType': args.movement_type,
        # 'timeBetweenBugs': self.time_between_bugs,
        # 'isStopOnReward': self.is_always_reward,
        'isLogTrajectory': True,
        # 'bugSize': self.bug_size,
        # 'backgroundColor': self.background_color,
        'exitHole': args.exit_hole,
        'rewardAnyTouchProb': 0
    }
    cache_.publish_command('init_bugs', json.dumps(options))


if __name__ == "__main__":
    start_trial()
