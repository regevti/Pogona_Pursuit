import re
import time
from functools import wraps
from datetime import datetime, timedelta
import threading
from loggers import get_logger
import config
from cache import RedisCache, CacheColumns as cc
from compress_videos import get_videos_for_compression, compress

TIME_TABLE = {
    'cameras_on': ('07:10', '18:45')
}
ALWAYS_ON_CAMERAS_RESTART_DURATION = 15 * 60  # seconds
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
        self.arena_mgr = arena_mgr
        self.next_experiment_time = None

    def run(self):
        time.sleep(10)  # let all other arena processes and threads to start
        t0 = None  # every minute
        t1 = None  # every 10 minutes
        while not self.arena_mgr.arena_shutdown_event.is_set():
            if not t0 or time.time() - t0 >= 60:
                t0 = time.time()
                self.check_camera_status()
                self.check_scheduled_experiments()
                self.arena_mgr.update_upcoming_schedules()

            if not t1 or time.time() - t1 >= 60 * 10:
                t1 = time.time()
                self.compress_videos()

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
        start, end = [datetime.combine(now, datetime.strptime(t, '%H:%M').time())
                      for t in TIME_TABLE[label]]
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
                self.start_camera(cu)

                # if there are any alive_predictors on, restart every x minutes.
                if cu.is_on() and cu.get_alive_predictors() and cu.preds_start_time and \
                        time.time() - cu.preds_start_time > ALWAYS_ON_CAMERAS_RESTART_DURATION:
                    self.logger.info(f'restarting predictors of {cu.cam_name}')
                    cu.reload_predictors(is_experiment=False)

    def stop_camera(self, cu):
        if cu.is_on():
            self.logger.info(f'stopping CU {cu.cam_name}')
            cu.stop()

    def start_camera(self, cu):
        if not cu.is_on():
            self.logger.info(f'starting CU {cu.cam_name}')
            cu.start()
            time.sleep(5)

    @schedule_method
    def compress_videos(self):
        if self.is_in_range('cameras_on'):
            return

        videos = get_videos_for_compression()
        if not videos:
            return

        t = threading.Thread(target=compress, args=(videos[0],))
        t.start()
