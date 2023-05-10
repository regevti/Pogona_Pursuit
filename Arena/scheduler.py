import re
import time
from functools import wraps
from datetime import datetime, timedelta
import threading
import multiprocessing
from loggers import get_logger
import config
from cache import RedisCache, CacheColumns as cc
from compress_videos import get_videos_ids_for_compression, compress
from periphery_integration import PeripheryIntegrator
from analysis.pose import convert_all_videos

env = config.env
TIME_TABLE = {
    'cameras_on': (env('CAMERAS_ON_TIME', '07:00'), env('CAMERAS_OFF_TIME', '19:00')),
    'lights_sunrise': env('LIGHTS_SUNRISE', '07:00'),
    'lights_sunset': env('LIGHTS_SUNSET', '19:00'),
    'dwh_commit_time': env('DWH_COMMIT_TIME', '00:00')
}
ALWAYS_ON_CAMERAS_RESTART_DURATION = 30 * 60  # seconds
cache = RedisCache()


def schedule_method(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            args[0].logger.error(f'Error in {func.__name__}: {exc}')
    return wrapper


class Scheduler(threading.Thread):
    def __init__(self, arena_mgr):
        super().__init__()
        self.logger = get_logger('Scheduler')
        self.logger.debug('Scheduler started...')
        self.arena_mgr = arena_mgr
        self.periphery = PeripheryIntegrator()
        self.next_experiment_time = None
        self.dlc_on = multiprocessing.Event()
        self.compress_threads = {}
        self.lights_state = 0  # 0 - off, 1 - on

    def run(self):
        time.sleep(10)  # let all other arena processes and threads to start
        t0 = None  # every minute
        t1 = None  # every 10 minutes
        while not self.arena_mgr.arena_shutdown_event.is_set():
            if not t0 or time.time() - t0 >= 60:  # every minute
                t0 = time.time()
                self.check_lights()
                self.check_camera_status()
                self.check_scheduled_experiments()
                self.arena_mgr.update_upcoming_schedules()

            if not t1 or time.time() - t1 >= 60 * 5:  # every 5 minutes
                t1 = time.time()
                self.compress_videos()
                if config.IS_RUN_NIGHTLY_POSE_ESTIMATION:
                    self.run_pose()

    @schedule_method
    def check_lights(self):
        """Check that during the day LEDs are on and IR is off, and vice versa during the night"""
        if self.is_in_range('lights_sunrise'):
            self.turn_light(config.IR_LIGHT_NAME, 0)
            self.turn_light(config.DAY_LIGHT_NAME, 1)
        elif self.is_in_range('lights_sunset'):
            self.turn_light(config.IR_LIGHT_NAME, 1)
            self.turn_light(config.DAY_LIGHT_NAME, 0)

    def turn_light(self, name, state):
        if not name:
            return
        self.periphery.switch(name, state)
        self.logger.info(f'turn {name} {"on" if state else "off"}')

    @schedule_method
    def check_scheduled_experiments(self):
        """Check if a scheduled experiment needs to be executed and run it"""
        for schedule_id, schedule_string in self.arena_mgr.schedules.items():
            m = re.search(r'(?P<date>.{16}) - (?P<name>.*)', schedule_string)
            if m:
                schedule_date = datetime.strptime(m.group('date'), config.schedule_date_format)
                if (schedule_date - datetime.now()).total_seconds() <= 0:
                    self.arena_mgr.start_cached_experiment(m.group('name'))

    @staticmethod
    def is_in_range(label):
        now = datetime.now()
        val = TIME_TABLE[label]
        if isinstance(val, tuple):
            start, end = [datetime.combine(now, datetime.strptime(t, '%H:%M').time()) for t in val]
        elif isinstance(val, str):
            dt = datetime.combine(now, datetime.strptime(val, '%H:%M').time())
            start, end = dt, dt + timedelta(minutes=1)
        else:
            raise Exception(f'bad value for {label}: {val}')

        return ((now - start).total_seconds() >= 0) and ((now - end).total_seconds() <= 0)

    @schedule_method
    def check_camera_status(self):
        """turn off cameras outside working hours, and restart predictors. Does nothing during
        an active experiment"""
        if cache.get(cc.IS_EXPERIMENT_CONTROL_CAMERAS):
            return

        for cam_name, cu in self.arena_mgr.units.copy().items():
            if cu.is_starting or cu.is_stopping:
                continue

            if not self.is_in_range('cameras_on'):
                # outside the active hours
                self.stop_camera(cu)
            else:
                # in active hours
                if cu.cam_config.get('is_manual'):
                    # camera will be stopped only if min_duration was reached
                    self.stop_camera(cu)
                else:
                    self.start_camera(cu)
                    # if there are any alive_predictors on, restart every x minutes.
                    if cu.is_on() and cu.get_alive_predictors() and cu.preds_start_time and \
                            time.time() - cu.preds_start_time > ALWAYS_ON_CAMERAS_RESTART_DURATION:
                        self.logger.info(f'restarting camera unit of {cu.cam_name}')
                        cu.stop()
                        if cam_name == self.arena_mgr.get_streaming_camera():
                            self.arena_mgr.stop_stream()
                        time.sleep(1)
                        cu.start()

    def stop_camera(self, cu):
        if cu.is_on() and cu.time_on > config.camera_on_min_duration:
            self.logger.info(f'stopping CU {cu.cam_name}')
            cu.stop()

    def start_camera(self, cu):
        if not cu.is_on():
            self.logger.debug(f'starting CU {cu.cam_name}')
            cu.start()
            time.sleep(5)

    @schedule_method
    def compress_videos(self):
        if self.is_in_range('cameras_on') or not self.is_compression_thread_available():
            return

        videos = get_videos_ids_for_compression(sort_by_size=True)
        if not videos:
            return

        while self.is_compression_thread_available():
            currently_compressed_vids = [v for _, v in self.compress_threads.values()]
            vids_ = [v for v in videos if v not in currently_compressed_vids]
            if not vids_:
                return
            t = threading.Thread(target=compress, args=(vids_[0],))
            t.start()
            self.compress_threads[t.name] = (t, vids_[0])

    def is_compression_thread_available(self):
        for thread_name in list(self.compress_threads.keys()):
            t, _ = self.compress_threads[thread_name]
            if not t.is_alive():
                self.compress_threads.pop(thread_name)

        return len(self.compress_threads) < config.MAX_COMPRESSION_THREADS

    @schedule_method
    def run_pose(self):
        if self.is_in_range('cameras_on') or cache.get(cc.IS_BLANK_CONTINUOUS_RECORDING) or self.dlc_on.is_set():
            return

        multiprocessing.Process(target=_run_pose_callback, args=(self.dlc_on,)).start()
        self.dlc_on.set()


def _run_pose_callback(dlc_on):
    try:
        convert_all_videos(max_videos=20)
    finally:
        dlc_on.clear()

