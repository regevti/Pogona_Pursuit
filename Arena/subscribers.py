import re
import json
import time
import inspect
import threading
import pandas as pd
from datetime import datetime
from pathlib import Path
from parallel_port import ParallelPort

from cache import CacheColumns as cc, RedisCache
import config
from loggers import get_logger, get_process_logger
from utils import Serializer
from db_models import ORM


class Subscriber(threading.Thread):
    sub_name = ''

    def __init__(self, stop_event: threading.Event, log_queue=None, channel=None, callback=None):
        super().__init__()
        self.cache = RedisCache()
        self.channel = channel or config.subscription_topics[self.sub_name]
        self.name = self.sub_name or self.channel.split('/')[-1]
        if log_queue is None:
            self.logger = get_logger(str(self))
        else:
            self.logger = get_process_logger(str(self), log_queue)
        self.stop_event = stop_event
        self.callback = callback

    def __str__(self):
        return self.name

    def run(self):
        try:
            p = self.cache._redis.pubsub()
            p.psubscribe(self.channel)
            self.logger.debug(f'start listening on {self.channel}')
            while not self.stop_event.is_set():
                message_dict = p.get_message(ignore_subscribe_messages=True, timeout=1)
                if message_dict:
                    channel, data = self.parse_message(message_dict)
                    self._run(channel, data)
                    if self.name == 'arena_shutdown':
                        return
                time.sleep(0.01)
            p.punsubscribe()
        except:
            self.logger.exception(f'Error in subscriber {self.name}')

    def _run(self, channel, data):
        if self.callback is not None:
            self.callback(channel, data)

    @staticmethod
    def parse_message(msg: dict):
        channel = msg.get('channel')
        payload = msg.get('data')
        if isinstance(channel, bytes):
            channel = channel.decode()
        if isinstance(payload, bytes):
            payload = payload.decode()
        return channel, payload


class ExperimentLogger(Subscriber):
    def __init__(self, stop_event: threading.Event, log_queue, channel=None, callback=None):
        super().__init__(stop_event, log_queue, channel, callback)
        self.config = config.experiment_metrics[self.name]
        self.orm = ORM()

    def __str__(self):
        return f'{self.name}-Logger'

    def _run(self, channel, data):
        try:
            payload = json.loads(data)
            payload = self.convert_time_fields(payload)
            self.payload_action(payload)

            if self.config.get('is_write_csv'):
                self.save_to_csv(payload)
            if self.config.get('is_write_db'):
                self.commit_to_db(payload)
        except Exception as exc:
            self.logger.exception(f'Unable to parse log payload of {self.name}: {exc}')

    def payload_action(self, payload):
        pass

    def ms2datetime(self, x, to_string=True):
        try:
            x = pd.to_datetime(x, unit='ms').tz_localize('utc').tz_convert('Asia/Jerusalem')
            if to_string:
                x = x.isoformat()
        except Exception as exc:
            self.logger.exception(f'Unable to convert ms time to local; {exc}')
        return x

    @property
    def time_fields(self):
        return ['time', 'start_time', 'end_time']

    def convert_time_fields(self, payload: dict) -> dict:
        for k, v in payload.copy().items():
            if k in self.time_fields:
                payload[k] = self.ms2datetime(v)
            elif isinstance(v, list):
                payload[k] = [self.convert_time_fields(x) for x in v]
            elif isinstance(v, dict):
                payload[k] = self.convert_time_fields(v)

        return payload

    def commit_to_db(self, payload):
        pass

    def save_to_csv(self, payload, filename=None):
        df = self.to_dataframe(payload)
        try:
            filename = self.get_csv_filename(filename)
            if filename.exists():
                df.to_csv(filename, mode='a', header=False)
            else:
                df.to_csv(filename)
                self.logger.debug(f'Creating analysis log: {filename}')
        except Exception as exc:
            self.logger.exception(f'ERROR saving event to csv; {exc}')

    def get_csv_filename(self, filename=None) -> Path:
        if self.cache.get_current_experiment():
            if self.config.get('is_overall_experiment'):
                parent = self.cache.get(cc.EXPERIMENT_PATH)
            else:
                parent = self.cache.get(cc.EXPERIMENT_BLOCK_PATH)
        else:
            parent = f'events/{datetime.today().strftime("%Y%m%d")}'
            Path(parent).mkdir(parents=True, exist_ok=True)

        return Path(f'{parent}/{filename or self.config["csv_file"]}')

    @staticmethod
    def to_dataframe(payload) -> pd.DataFrame:
        if not isinstance(payload, (list, tuple)):
            payload = [payload]
        return pd.DataFrame(payload)


class TouchLogger(ExperimentLogger):
    def payload_action(self, payload):
        self.handle_hit(payload)

    def handle_hit(self, payload):
        if (self.cache.get(cc.IS_ALWAYS_REWARD) and (payload.get('is_hit')) or
           (payload.get('is_reward_any_touch')) and payload.get('is_reward_bug')):
            self.cache.publish_command('reward')
            return True

    def commit_to_db(self, payload):
        self.orm.commit_strike(payload)


class TrialDataLogger(ExperimentLogger):
    def save_to_csv(self, payload, filename=None):
        for key in ['bug_trajectory', 'video_frames']:
            payload_ = payload.get(key)
            if payload_:
                super().save_to_csv(payload_, filename=self.config["csv_file"][key])

    def commit_to_db(self, payload):
        self.orm.update_trial_data(payload)


class TemperatureLogger(Subscriber):
    def __init__(self, stop_event: threading.Event, log_queue, **kwargs):
        super().__init__(stop_event, log_queue, channel=config.subscription_topics['temperature'])
        self.n_tries = 5
        self.orm = ORM()

    def run(self):
        ser = Serializer(logger=self.logger)
        self.logger.debug('read_temp started')
        grace_count = 0
        while not self.stop_event.is_set() and grace_count < self.n_tries:
            try:
                line = ser.read_line()
                if line and isinstance(line, bytes):
                    m = re.search(r'Temperature is: ([\d.]+)', line.decode())
                    if m:
                        self.cache.publish(config.subscription_topics['temperature'], m[1])
                        self.commit_to_db(m[1])
            except Exception as exc:
                self.logger.exception(f'Error in read_temp: {exc}')
                grace_count += 1
            time.sleep(config.temperature_logging_delay_sec)

    def commit_to_db(self, payload):
        try:
            self.orm.commit_temperature(payload)
        except:
            self.logger.exception('Error committing temperature to DB')


class ArenaOperations(Subscriber):
    sub_name = 'arena_operations'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the parallel port
        self.parport = None
        self.last_reward_time = None
        if config.is_use_parport:
            try:
                self.parport = ParallelPort()
                self.logger.debug('Parallel port is ready')
            except Exception as exc:
                self.logger.exception(f'Error loading feeder: {exc}')
        else:
            self.logger.warning('Parallel port is not configured. some arena operation cannot work')

    def _run(self, channel, data):
        if self.parport is None:
            return
        channel = channel.split('/')[-1]
        if channel == 'reward':
            ret = self.reward()
            if ret:
                self.logger.info('Manual reward was given')
            else:
                self.logger.warning('Could not give manual reward')

        elif channel == 'led_light':
            if self.parport:
                self.logger.debug(f'LED lights turned {data}')
                self.parport.led_lighting(data)

        elif channel == 'heat_light':
            if self.parport:
                self.logger.debug(f'Heat lights turned {data}')
                self.parport.heat_lighting(data)

    def reward(self):
        if self.parport and not self.cache.get(cc.IS_REWARD_TIMEOUT):
            self.parport.feed()
            self.last_reward_time = time.time()
            self.cache.set(cc.IS_REWARD_TIMEOUT, True)
            self.cache.publish('cmd/visual_app/reward_given')
            return True


class AppHealthCheck(Subscriber):
    sub_name = 'healthcheck'

    def run(self):
        try:
            p = self.cache._redis.pubsub()
            p.psubscribe(self.channel)
            self.logger.debug(f'start listening on {self.channel}')
            while not self.stop_event.is_set():
                self.cache.publish_command('healthcheck')
                time.sleep(0.01)
                open_apps_hosts = set()
                for _ in range(3):
                    message_dict = p.get_message(ignore_subscribe_messages=True, timeout=1)
                    if message_dict and message_dict.get('data'):
                        try:
                            message_dict = json.loads(message_dict.get('data').decode('utf-8'))
                            open_apps_hosts.add(message_dict['host'])
                        except:
                            pass

                if open_apps_hosts:
                    if len(open_apps_hosts) > 1:
                        self.logger.warning(f'more than 1 pogona hunter apps are open: {open_apps_hosts}')
                    self.cache.set(cc.OPEN_APP_HOST, list(open_apps_hosts)[0])
                else:
                    if self.cache.get(cc.OPEN_APP_HOST):
                        self.cache.delete(cc.OPEN_APP_HOST)
                time.sleep(2)
            p.punsubscribe()
        except:
            self.logger.exception(f'Error in subscriber {self.name}')


def start_management_subscribers(arena_shutdown_event, log_queue, subs_dict):
    """Start all subscribers that must listen as long as an arena management instance initiated"""
    threads = {}
    for topic, callback in subs_dict.items():
        threads[topic] = Subscriber(arena_shutdown_event, log_queue,
                                    config.subscription_topics[topic], callback)
        threads[topic].start()

    threads['arena_operations'] = ArenaOperations(arena_shutdown_event, log_queue)
    threads['arena_operations'].start()
    threads['temperature'] = TemperatureLogger(arena_shutdown_event, log_queue)
    threads['temperature'].start()
    threads['healthcheck'] = AppHealthCheck(arena_shutdown_event, log_queue)
    threads['healthcheck'].start()
    return threads


def start_experiment_subscribers(arena_shutdown_event, log_queue):
    """Start the subscribers for a running experiment"""
    threads = {}
    for channel_name, d in config.experiment_metrics.items():
        thread_name = f'metric_{channel_name}'
        if channel_name == 'touch':
            logger_cls = TouchLogger
        elif channel_name == 'trial_data':
            logger_cls = TrialDataLogger
        else:
            logger_cls = ExperimentLogger

        threads[thread_name] = logger_cls(arena_shutdown_event, log_queue,
                                          channel=config.subscription_topics[channel_name])
        threads[thread_name].start()
    return threads


# def block_log(data):
#     try:
#         block_path = Path(cache.get(cc.EXPERIMENT_BLOCK_PATH))
#         if block_path.exists():
#             with (block_path / 'block.log').open('a') as f:
#                 f.write(f'{datetime.now().isoformat()} - {data}\n')
#     except Exception as exc:
#         print(f'Error writing block_log; {exc}')
