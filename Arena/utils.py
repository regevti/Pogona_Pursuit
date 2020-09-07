import os
import logging
from pathlib import Path
import numpy as np
from io import StringIO
from datetime import datetime


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