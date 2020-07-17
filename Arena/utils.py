import logging
from pathlib import Path
from numpy import mean
from io import StringIO

log_stream = StringIO()


def clear_log_stream():
    log_stream.truncate(0)
    log_stream.seek(0)


def get_logger(device_id: str, dir_path: str) -> logging.Logger:
    """
    Create file and stream logger for camera
    :param device_id: Camera device id
    :param dir_path: The path of the dir in which logger file should be saved
    :return: Logger
    """
    logger = logging.getLogger(device_id)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - <CAM:{device_id: >8}> - %(message)s')

    if dir_path:
        fh = logging.FileHandler(f'{dir_path}/output.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    sh = logging.StreamHandler(log_stream)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def calculate_fps(frame_times):
    diffs = [j - i for i, j in zip(frame_times[:-1], frame_times[1:])]
    return 1 / mean(diffs)


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path