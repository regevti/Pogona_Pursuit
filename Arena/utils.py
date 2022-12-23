import logging
import os
import re

import serial
import time
import threading
from pathlib import Path
import numpy as np
import asyncio
from io import StringIO
from datetime import datetime
from subprocess import Popen, PIPE
import config


def run_command(cmd):
    """Execute shell command"""
    process = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, executable='/bin/bash')
    stdout, stderr = process.communicate()
    if stderr:
        print(f'Error running cmd: "{cmd}"; {stderr}')
    yield stdout


DISPLAY = f'DISPLAY="{config.ARENA_DISPLAY}"'


def turn_display_on():
    cmds = [
        'pkill chrome || true',  # kill all existing chrome processes
        f'{DISPLAY} xrandr --output HDMI-0 --auto --right-of DP-0',  # turn touch screen on
        f'{DISPLAY} xinput enable {config.TOUCHSCREEN_DEVICE_ID}',  # enable touch
        f'{DISPLAY} xinput map-to-output {config.TOUCHSCREEN_DEVICE_ID} HDMI-0',
        'sleep 1',
        'scripts/start_pogona_hunter.sh'  # start chrome with the bug application
    ]
    return os.system(' && '.join(cmds))


def turn_display_off():
    cmds = [
        'pkill chrome || true',
        f'{DISPLAY} xrandr --output HDMI-0 --off',  # turn touchscreen off
        f'{DISPLAY} xinput disable {config.TOUCHSCREEN_DEVICE_ID}',  # disable touch
    ]
    return os.system(' && '.join(cmds))


def async_call_later(seconds, callback):
    async def schedule():
        await asyncio.sleep(seconds)
        if asyncio.iscoroutinefunction(callback):
            await callback()
        else:
            callback()
    asyncio.ensure_future(schedule())


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


class Serializer:
    """Serializer for connecting the TTL Arduino"""
    def __init__(self, logger=None):
        self.logger = logger
        self.device_id = config.SERIAL_PORT_TEMP
        self.find_temperature_sensor_device()
        self.ser = serial.Serial(self.device_id, config.SERIAL_BAUD, timeout=1)
        time.sleep(0.5)

    def read_line(self):
        return self.ser.read_until()

    def start_acquisition(self):
        self.ser.write(b'H')

    def stop_acquisition(self):
        self.ser.write(b'L')

    def find_temperature_sensor_device(self):
        """Get the device ID of the temperature sensor"""
        try:
            out = next(run_command('scripts/usb_scan.sh | grep Seeed')).decode()
            m = re.search(r'(/dev/tty\w+) - ', out)
            if not m:
                raise Exception('unable to find the temperature arduino')
            self.device_id = m.group(1)
            self.logger.debug(f'Temperature sensor device ID was set to {self.device_id}')
        except Exception:
            self.logger.exception(f'ERROR in finding the temperature arduino;')


class MetaArray(np.ndarray):
    """Array with metadata."""

    def __new__(cls, array, dtype=None, order=None, **kwargs):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)
        obj.metadata = kwargs
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', None)


def run_in_thread(func):
    def wrapper(*args):
        t = threading.Thread(target=func, args=args)
        t.start()
    return wrapper


def get_sys_metrics():
    gpu_usage, cpu_usage, memory_usage, storage = None, None, None, None
    try:
        cmd = 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits'
        res = next(run_command(cmd)).decode().replace(' ', '').replace('\n', '')
        res = [int(x) for x in res.split(',')]
        gpu_usage = res[0] / res[1]
    except Exception:
        pass
    try:
        # cmd = "cat /proc/stat |grep cpu |tail -1|awk '{print ($5*100)/($2+$3+$4+$5+$6+$7+$8+$9+$10)}'|awk '{print 100-$1}'"
        cmd = "awk '{u=$2+$4; t=$2+$4+$5; if (NR==1){u1=u; t1=t;} else print ($2+$4-u1) * 100 / (t-t1); }' <(grep 'cpu ' /proc/stat) <(sleep 1;grep 'cpu ' /proc/stat)"
        res = next(run_command(cmd)).decode().replace(' ', '').replace('\n', '')
        cpu_usage = float(res)
    except Exception:
        pass
    try:
        cmd = "free -mt | grep Total"
        res = next(run_command(cmd)).decode()
        m = re.search(r'Total:\s+(?P<total>\d+)\s+(?P<used>\d+)\s+(?P<free>\d+)', res)
        memory_usage = int(m.group('used')) / int(m.group('total'))
    except Exception:
        pass
    try:
        cmd = 'df -h /data | grep -oP "\d+%"'
        res = next(run_command(cmd)).decode()
        if res and isinstance(res, str):
            storage = float(res.replace('%', ''))
    except Exception:
        pass
    return {'gpu_usage': gpu_usage, 'cpu_usage': cpu_usage, 'memory_usage': memory_usage, 'storage': storage}


class KalmanFilter:
    def __init__(self):
        """
        Parameters:
        x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
        P: initial uncertainty convariance matrix
        measurement: observed position
        R: measurement noise
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        """
        self.x = None
        self.P = np.matrix(np.eye(4)) * 1000  # initial uncertainty
        self.R = 0.01**2
        self.F = np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      ''')
        self.H = np.matrix('''
                      1. 0. 0. 0.;
                      0. 1. 0. 0.''')
        self.motion = np.matrix('0. 0. 0. 0.').T
        self.Q = np.matrix(np.eye(4))

    def get_filtered(self, pos):
        if self.x is None:
            self.x = np.matrix([*pos, 0, 0]).T
        else:
            if pos is not None:
                self.update(pos)
            self.predict()

        return np.array(self.x).ravel()[:2]

    def update(self, measurement):
        '''
        Parameters:
        x: initial state
        P: initial uncertainty convariance matrix
        measurement: observed position (same shape as H*x)
        R: measurement noise (same shape as H)
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        F: next state function: x_prime = F*x
        H: measurement function: position = H*x

        Return: the updated and predicted new values for (x, P)

        See also http://en.wikipedia.org/wiki/Kalman_filter

        This version of kalman can be applied to many different situations by
        appropriately defining F and H
        '''
        # UPDATE x, P based on measurement m
        # distance between measured and current position-belief
        y = np.matrix(measurement).T - self.H * self.x
        S = self.H * self.P * self.H.T + self.R  # residual convariance
        K = self.P * self.H.T * S.I  # Kalman gain
        self.x = self.x + K * y
        I = np.matrix(np.eye(self.F.shape[0]))  # identity matrix
        self.P = (I - K * self.H) * self.P

    def predict(self):
        # PREDICT x, P based on motion
        self.x = self.F * self.x + self.motion
        self.P = self.F * self.P * self.F.T + self.Q
