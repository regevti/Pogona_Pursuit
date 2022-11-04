import time
from datetime import datetime
import threading
from loggers import get_logger

time_table = {
    'cameras_on': (7, 19)
}


class Scheduler(threading.Thread):
    def __init__(self, arena_mgr):
        super().__init__()
        self.logger = get_logger('Scheduler')
        self.arena_mgr = arena_mgr

    def run(self):
        time.sleep(10)  # let all other arena processes and threads to start
        t_last = None
        while not self.arena_mgr.arena_shutdown_event.is_set():
            if not t_last or time.time() - t_last >= 60:
                t_last = time.time()
                self.check_camera_on()
            time.sleep(0.5)

    def check_camera_on(self):
        current_hour = datetime.now().hour
        start, end = time_table['cameras_on']
        if current_hour < start or current_hour >= end:
            for cam_name, cu in self.arena_mgr.units.copy().items():
                if cu.is_on():
                    self.logger.info(f'stopping CU {cam_name}')
                    cu.stop()
        else:
            for cam_name, cu in self.arena_mgr.units.copy().items():
                if not cu.is_on():
                    self.logger.info(f'starting CU {cam_name}')
                    cu.start()
                    time.sleep(5)