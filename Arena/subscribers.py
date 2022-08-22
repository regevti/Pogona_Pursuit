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


class Subscriber(threading.Thread):
    sub_name = ''

    def __init__(self, stop_event: threading.Event, log_queue, channel=None, callback=None):
        super().__init__()
        self.cache = RedisCache()
        self.channel = channel or config.subscription_topics[self.sub_name]
        self.name = self.sub_name or self.channel.split('/')[-1]
        self.logger = get_process_logger(str(self), log_queue)
        self.stop_event = stop_event
        self.callback = callback

    def __str__(self):
        return self.name

    def run(self):
        try:
            p = self.cache._redis.pubsub()
            p.psubscribe(self.channel)
            self.logger.info(f'start listening on {self.channel}')
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
            self.callback()

    @staticmethod
    def parse_message(msg: dict):
        channel = msg.get('channel')
        payload = msg.get('payload')
        if isinstance(channel, bytes):
            channel = channel.decode()
        if isinstance(payload, bytes):
            payload = payload.decode()
        return channel, payload


class MetricsLogger(Subscriber):
    def __init__(self, stop_event: threading.Event, log_queue, channel=None, callback=None):
        super().__init__(stop_event, log_queue, channel, callback)
        self.config = config.experiment_metrics[self.name]

    def __str__(self):
        return f'{self.name} Logger'

    def _run(self, channel, data):
        try:
            payload = json.loads(data)
            if self.name == 'touch':
                ret = self.handle_hit(payload)
                if ret:
                    self.logger.info('Reward was given after successful hit')

            if self.config.get('is_write_csv'):
                self.save_to_csv(payload)
            if self.config.get('is_write_db'):
                self.commit_to_db(payload)
        except Exception as exc:
            self.logger.exception(f'Unable to parse log payload of {self.name}: {exc}')

    def ms2datetime(self, x):
        try:
            x = pd.to_datetime(x, unit='ms').tz_localize('utc').tz_convert('Asia/Jerusalem')
        except Exception as exc:
            self.logger.exception(f'Unable to convert ms time to local; {exc}')
        return x

    def parse_time_columns(self, payload_df) -> pd.DataFrame:
        # Time conversions
        try:
            if self.name == 'trials_times':
                payload_df[['start', 'end']] = payload_df[['start', 'end']].applymap(self.ms2datetime)
                payload_df.set_index('trial', inplace=True)
            elif 'time' in payload_df.columns:
                payload_df['time'] = payload_df['time'].map(self.ms2datetime)
            else:
                payload_df['time'] = datetime.now()
        except Exception as exc:
            self.logger.exception(f'Error parsing metric {self.name} payload; {exc}')
        return payload_df

    def parse_payload(self, payload) -> pd.DataFrame:
        if not isinstance(payload, (list, tuple)):
            payload = [payload]
        return self.parse_time_columns(pd.DataFrame(payload))

    def commit_to_db(self, payload):
        pass

    def save_to_csv(self, payload):
        df = self.parse_payload(payload)
        try:
            filename = self.get_csv_filename()
            if filename.exists():
                df.to_csv(filename, mode='a', header=False)
            else:
                df.to_csv(filename)
                self.logger.info(f'Creating analysis log: {filename}')
        except Exception as exc:
            self.logger.exception(f'ERROR saving event to csv; {exc}')

    def get_csv_filename(self) -> Path:
        if self.cache.get_current_experiment():
            if self.config.get('is_overall_experiment'):
                parent = self.cache.get(cc.EXPERIMENT_PATH)
            else:
                parent = self.cache.get(cc.EXPERIMENT_BLOCK_PATH)
        else:
            parent = f'events/{datetime.today().strftime("%Y%m%d")}'
            Path(parent).mkdir(parents=True, exist_ok=True)

        return Path(f'{parent}/{self.config["csv_file"]}')

    def handle_hit(self, payload):
        if self.cache.get(cc.IS_ALWAYS_REWARD) and (payload.get('is_hit') or payload.get('is_reward_any_touch')) \
                and payload.get('is_reward_bug'):
            self.cache.publish_command('reward')


class TemperatureLogger(MetricsLogger):
    def __init__(self, stop_event: threading.Event, log_queue, **kwargs):
        super().__init__(stop_event, log_queue, channel=config.subscription_topics['temperature'])
        self.n_tries = 5

    def run(self):
        ser = Serializer()
        self.logger.info('read_temp started')
        grace_count = 0
        while self.cache.get_current_experiment() and self.can_loop() and grace_count < self.n_tries:
            try:
                line = ser.read_line()
                if line and isinstance(line, bytes):
                    m = re.search(r'Temperature is: ([\d.]+)', line.decode())
                    if m:
                        self.cache.publish(config.subscription_topics['temperature'], m[1])
            except Exception as exc:
                self.logger.exception(f'Error in read_temp: {exc}')
                grace_count += 1
            time.sleep(5)


class ArenaOperations(Subscriber):
    sub_name = 'arena_operations'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the parallel port
        self.parport = None
        if config.is_use_parport:
            try:
                self.parport = ParallelPort()
                self.logger.info('Parallel port is ready')
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
                self.logger.info(f'LED lights turned {data}')
                self.parport.led_lighting(data)

    def reward(self):
        if self.parport and not self.cache.get(cc.IS_REWARD_TIMEOUT):
            self.parport.feed()
            self.cache.set(cc.IS_REWARD_TIMEOUT, True)
            self.cache.publish('cmd/visual_app/reward_given')
            return True


def get_experiment_subscribers(stop_event, log_queue):
    threads = {}
    for channel_name, d in config.experiment_metrics.items():
        thread_name = f'metric_{channel_name}'
        if channel_name == 'temperature':
            threads[thread_name] = TemperatureLogger(stop_event, log_queue)
        else:
            threads[thread_name] = MetricsLogger(stop_event, log_queue,
                                                 channel=config.subscription_topics[channel_name])
    return threads


# def block_log(data):
#     try:
#         block_path = Path(cache.get(cc.EXPERIMENT_BLOCK_PATH))
#         if block_path.exists():
#             with (block_path / 'block.log').open('a') as f:
#                 f.write(f'{datetime.now().isoformat()} - {data}\n')
#     except Exception as exc:
#         print(f'Error writing block_log; {exc}')
