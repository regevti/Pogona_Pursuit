import re
import json
import time
import inspect
import queue
import threading
import pandas as pd
from datetime import datetime
from pathlib import Path

import redis.exceptions

from cache import CacheColumns as cc, RedisCache
import config
from loggers import get_logger, get_process_logger
from utils import Serializer, run_in_thread, run_command
from db_models import ORM
from periphery_integration import PeripheryIntegrator, TemperatureListener, MQTTListener


class DoubleEvent(Exception):
    """Double event"""


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
        except DoubleEvent:
            pass
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
    def __init__(self, *args, **kwargs):
        super(TouchLogger, self).__init__(*args, **kwargs)
        self.periphery = PeripheryIntegrator()
        self.touches_queue = queue.Queue()
        self.start_touches_receiver_thread()

    def payload_action(self, payload):
        try:
            self.touches_queue.put_nowait(payload)
        except queue.Full:
            pass
        except Exception as exc:
            self.logger.error(f'Error in image sink; {exc}')

    def start_touches_receiver_thread(self):
        def loop(q):
            self.logger.info('touch listener has started')
            last_touch_ts = None
            while not self.stop_event.is_set():
                try:
                    payload = q.get_nowait()
                    ts = payload.get('time')
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)

                    dt = (ts - last_touch_ts).total_seconds() if last_touch_ts else 0
                    if last_touch_ts and ts and dt < 0.2:
                        continue

                    last_touch_ts = ts
                    self.logger.info(f'Received touch event; timestamp={ts}; '
                                     f'time passed from last reward: {dt:.1f} seconds')
                    self.handle_hit(payload)
                except queue.Empty:
                    pass
            self.logger.debug('touches receiver thread is closed')

        t = threading.Thread(target=loop, args=(self.touches_queue,))
        t.start()

    def handle_hit(self, payload):
        if self.cache.get(cc.IS_ALWAYS_REWARD) and payload.get('is_reward_bug') and \
                (payload.get('is_hit') or payload.get('is_reward_any_touch')):
            self.periphery.feed()
            return True

    @run_in_thread
    def commit_to_db(self, payload):
        self.orm.commit_strike(payload)


class TrialDataLogger(ExperimentLogger):
    def save_to_csv(self, payload, filename=None):
        for key, csv_path in self.config["csv_file"].items():
            payload_ = payload.get(key)
            if payload_:
                super().save_to_csv(payload_, filename=csv_path)

    def commit_to_db(self, payload):
        self.orm.update_trial_data(payload)


class TemperatureLogger(Subscriber):
    def __init__(self, stop_event: threading.Event, log_queue, **kwargs):
        super().__init__(stop_event, log_queue, channel=config.subscription_topics['temperature'])
        self.n_tries = 5
        self.orm = ORM()

    def run(self):
        def callback(payload):
            self.commit_to_db(payload)
            # self.cache.publish(config.subscription_topics['temperature'], payload)

        try:
            listener = TemperatureListener(is_debug=False, stop_event=self.stop_event, callback=callback)
        except Exception as exc:
            self.logger.error(f'Error loading temperature listener; {exc}')
            return
        self.logger.debug('read_temp started')
        listener.loop()

    def commit_to_db(self, payload):
        try:
            self.orm.commit_temperature(payload)
        except:
            self.logger.exception('Error committing temperature to DB')


class AppHealthCheck(Subscriber):
    sub_name = 'app_healthcheck'

    def run(self):
        try:
            p = self.cache._redis.pubsub()
            p.psubscribe(self.channel)
            self.logger.debug(f'start listening on {self.channel}')
            while not self.stop_event.is_set():
                self.cache.publish_command(self.sub_name)
                time.sleep(0.01)
                open_apps_hosts = set()
                for _ in range(3):
                    try:
                        message_dict = p.get_message(ignore_subscribe_messages=True, timeout=1)
                    except redis.exceptions.ConnectionError:
                        message_dict = None
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


class PeripheryHealthCheck(Subscriber):
    sub_name = 'periphery_healthcheck'

    def __init__(self, stop_event: threading.Event, log_queue=None, channel=None, callback=None):
        super().__init__(stop_event, log_queue, channel, callback)
        self.last_health_check_time = time.time()
        self.last_publish_error_time = None
        self.last_action_time = None
        self.max_check_delay = 20  # if there's no new healthcheck message for 10 seconds log error
        self.max_publish_delay = 120
        self.max_action_delay = 60 * 60

    def run(self):
        try:
            def hc_callback(payload):
                self.last_health_check_time = time.time()

            listener = MQTTListener(topics=['healthcheck'], is_debug=False, callback=hc_callback)
            self.logger.debug('periphery_healthcheck started')
            while not self.stop_event.is_set():
                listener.loop()
                time.sleep(0.1)
                if time.time() - self.last_health_check_time > self.max_check_delay:
                    if self.last_publish_error_time and time.time() - self.last_publish_error_time < self.max_publish_delay:
                        continue
                    self.logger.error('Arena periphery MQTT bridge is down')
                    if not self.last_action_time or time.time() - self.last_action_time > self.max_action_delay:
                        self.logger.warning('Running restart for arena periphery process')
                        next(run_command('supervisorctl restart reptilearn_arena'))
                        self.last_action_time = time.time()
                        time.sleep(4)

                    self.last_publish_error_time = time.time()

        except:
            self.logger.exception(f'Error in subscriber {self.name}')


def start_management_subscribers(arena_shutdown_event, log_queue, subs_dict):
    """Start all subscribers that must listen as long as an arena management instance initiated"""
    threads = {}
    for topic, callback in subs_dict.items():
        threads[topic] = Subscriber(arena_shutdown_event, log_queue,
                                    config.subscription_topics[topic], callback)
        threads[topic].start()

    threads['app_healthcheck'] = AppHealthCheck(arena_shutdown_event, log_queue)
    threads['app_healthcheck'].start()
    if not config.DISABLE_PERIPHERY:
        threads['temperature'] = TemperatureLogger(arena_shutdown_event, log_queue)
        threads['temperature'].start()
        # threads['periphery_healthcheck'] = PeripheryHealthCheck(arena_shutdown_event, log_queue, channel='mqtt')
        # threads['periphery_healthcheck'].start()
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
