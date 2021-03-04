import sys
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from celery import Celery
from celery.utils.log import get_task_logger
from parallel_port import ParallelPort
from cache import CacheColumns, RedisCache
import config
import paho.mqtt.client as mqtt


mqtt = mqtt.Client()
# if config.is_logger and config.is_debug_mode and not config.is_use_parport:
#     config.redis_host = 'cache'
cache = RedisCache()
logger = get_task_logger(__name__)
h = logging.StreamHandler(sys.stdout)
logger.addHandler(h)

parport = None
if config.is_use_parport:
    try:
        parport = ParallelPort()
    except Exception as exc:
        logger.warning(f'Error loading feeder: {exc}')


def is_always_reward():
    return cache.get(CacheColumns.ALWAYS_REWARD)


def log(msg):
    logger.info(msg)
    if mqtt is not None:
        mqtt.connect(config.mqtt_host)
        mqtt.publish(config.experiment_topic, msg)


app = Celery('logger', broker=f'redis://{config.redis_host}:6379/0')


@app.task
def handle_hit(payload):
    if is_always_reward() and payload.get('is_hit') and payload.get('is_reward_bug'):
        # end_app_wait()
        return reward()


@app.task
def end_experiment():
    cache.delete(CacheColumns.APP_ON)
    cache.delete(CacheColumns.EXPERIMENT_BLOCK_ON)
    cache.delete(CacheColumns.EXPERIMENT_BLOCK_PATH)
    cache.delete(CacheColumns.ALWAYS_REWARD)
    cache.delete(CacheColumns.EXPERIMENT_NAME)
    cache.delete(CacheColumns.EXPERIMENT_PATH)
    log('>> experiment finished\n')


@app.task
def end_app_wait():
    cache.delete(CacheColumns.APP_ON)


@app.task
def gaze_external(state):
    if parport:
        if state == 'on':
            parport.turn_on(0x04)
        else:
            parport.turn_off(0x04)


@app.task
def reward(is_force=False):
    if parport and (is_force or is_always_reward()):
        parport.feed()
        log('>> Reward was given')
        return True


@app.task
def led_light(state):
    if parport:
        log(f'>> LED lights turned {state}')
        parport.led_lighting(state)


@app.task
def block_log(msg):
    try:
        block_path = Path(cache.get(CacheColumns.EXPERIMENT_BLOCK_PATH))
        if block_path.exists():
            with (block_path / 'block.log').open('a') as f:
                f.write(f'{datetime.now().isoformat()} - {msg}\n')
    except Exception as exc:
        print(f'Error writing block_log; {exc}')


def ms2datetime(x):
    try:
        x = pd.to_datetime(x, unit='ms').tz_localize('utc').tz_convert('Asia/Jerusalem')
    except Exception as exc:
        logger.warning(f'Unable to convert ms time to local; {exc}')
    return x

@app.task
def save_to_csv(topic, payload):
    try:
        if topic not in ['trajectory', 'video_frames']:
            payload = [payload]
        df = pd.DataFrame(payload)
        # Time conversions
        if topic == 'trials_times':
            df[['start', 'end']] = df[['start', 'end']].applymap(ms2datetime)
            df.set_index('trial', inplace=True)
        elif 'time' in df.columns:
            df['time'] = df['time'].map(ms2datetime)
        else:
            df['time'] = datetime.now()

        filename = get_csv_filename(topic)
        if filename.exists():
            df.to_csv(filename, mode='a', header=False)
        else:
            df.to_csv(filename)
        logger.info(f'saved to {filename}')
    except Exception as exc:
        logger.warning(f'ERROR saving event to csv; {exc}')


def get_csv_filename(topic) -> Path:
    if cache.get(CacheColumns.EXPERIMENT_NAME):
        parent = cache.get(CacheColumns.EXPERIMENT_BLOCK_PATH)
    else:
        parent = f'events/{datetime.today().strftime("%Y%m%d")}'
        Path(parent).mkdir(parents=True, exist_ok=True)

    return Path(f'{parent}/{config.logger_files[topic]}')
