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


DISPLAY_CMD = 'DISPLAY=":0" xrandr --output HDMI-0 --{}'


def turn_display_on():
    return os.system('pkill chrome; ' + DISPLAY_CMD.format('auto') + ' --right-of DP-0 && sleep 1 && ' +
                       '/home/regev/scripts/start_pogona_hunter.sh')


def turn_display_off():
    return os.system('pkill chrome; ' + DISPLAY_CMD.format('off'))


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
    gpu_usage, cpu_usage, memory_usage = None, None, None
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
    return {'gpu_usage': gpu_usage, 'cpu_usage': cpu_usage, 'memory_usage': memory_usage}
