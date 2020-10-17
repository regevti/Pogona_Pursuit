import os
import logging
from pathlib import Path
import numpy as np
import asyncio
from io import StringIO
from datetime import datetime
from subprocess import Popen, PIPE


def run_command(cmd):
    """Execute shell command"""
    process = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    if stderr:
        print(f'Error running cmd: "{cmd}"; {stderr}')
    yield stdout


DISPLAY_CMD = 'DISPLAY=":0" xrandr --output HDMI-0 --{}'


def turn_display_on():
    return run_command(DISPLAY_CMD.format('auto'))


def turn_display_off():
    return run_command(DISPLAY_CMD.format('off'))


def async_call_later(seconds, callback):
    async def schedule():
        await asyncio.sleep(seconds)
        if asyncio.iscoroutinefunction(callback):
            await callback()
        else:
            callback()
    asyncio.ensure_future(schedule())


def get_log_stream():
    return StringIO()


def get_logger(device_id: str, dir_path: str, log_stream=None) -> logging.Logger:
    """
    Create file and stream logger for camera
    :param device_id: Camera device id
    :param dir_path: The path of the dir in which logger file should be saved
    :param log_stream: Log stream for string logging
    :return: Logger
    """
    logger = logging.getLogger(device_id)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - <CAM:{device_id: >8}> - %(message)s')
    if logger.hasHandlers():
        logger.handlers.clear()

    if dir_path:
        fh = logging.FileHandler(f'{dir_path}/output.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_stream:
        sh = logging.StreamHandler(log_stream)
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def is_debug_mode():
    return os.environ.get('DEBUG', False)


def is_predictor_experiment():
    return os.environ.get('PREDICTOR_EXPERIMENT', False)


def get_predictor_model():
    return os.environ.get('PREDICTOR_MODEL', 'lstm')


def calculate_fps(frame_times):
    diffs = [j - i for i, j in zip(frame_times[:-1], frame_times[1:])]
    fps = 1 / np.mean(diffs)
    std = fps - (1 / (np.mean(diffs) + np.std(diffs) / np.sqrt(len(diffs))))
    return fps, std


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_datetime_string():
    return datetime.now().strftime('%Y%m%dT%H%M%S')


def titlize(s: str):
    return s.replace('_', ' ').title()


def to_integer(x):
    try:
        return int(x)
    except Exception:
        return x
