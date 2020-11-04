import logging
import serial
import time
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


def get_logger(device_id: str, dir_path: str, log_stream=None, is_only_file=False) -> logging.Logger:
    """
    Create file and stream logger for camera
    :param device_id: Camera device id
    :param dir_path: The path of the dir in which logger file should be saved
    :param log_stream: Log stream for string logging
    :param is_only_file: Log to file only, not to stdout or string
    :return: Logger
    """
    logger = logging.getLogger(device_id + '_file' if is_only_file else '')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - <CAM:{device_id: >8}> - %(message)s')
    if logger.hasHandlers():
        logger.handlers.clear()

    if dir_path:
        fh = logging.FileHandler(f'{dir_path}/output.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if not is_only_file:
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


def calculate_fps(frame_times):
    diffs = [j - i for i, j in zip(frame_times[:-1], frame_times[1:])]
    fps = 1 / np.mean(diffs)
    std = fps - (1 / (np.mean(diffs) + np.std(diffs) / np.sqrt(len(diffs))))
    return fps, std


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def datetime_string():
    return datetime.now().strftime('%Y%m%dT%H%M%S')


def titlize(s: str):
    return s.replace('_', ' ').title()


def to_integer(x):
    try:
        return int(x)
    except Exception:
        return x


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


SERIAL_PORT_TEMP = '/dev/ttyACM1'
SERIAL_BAUD = 9600


class Serializer:
    """Serializer for connecting the TTL Arduino"""
    def __init__(self, port=SERIAL_PORT_TEMP, baud=SERIAL_BAUD):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(0.5)

    def read_line(self):
        return self.ser.read_until()

    def start_acquisition(self):
        self.ser.write(b'H')

    def stop_acquisition(self):
        self.ser.write(b'L')