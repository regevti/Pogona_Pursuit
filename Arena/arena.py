#!/usr/bin/env python
import sys
import time
import cv2
import re
import importlib
import atexit
import signal
import threading
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager, SyncManager
import numpy as np
import pandas as pd
from dataclasses import dataclass

import config
from cache import RedisCache, CacheColumns as cc
from utils import mkdir, datetime_string
from loggers import get_logger, get_process_logger, logger_thread, _loggers
from experiment import Experiment
from subscribers import Subscriber, MetricsLogger, ArenaOperations, AppHealthCheck

cache = RedisCache()


def signal_handler(signum, frame):
    print('signal detected!')
    cache.publish_command('arena_shutdown')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class ArenaProcess(mp.Process):
    def __init__(self,
                 cam_name: str,
                 shm: SharedMemory,
                 cam_config: dict,
                 log_queue: mp.Queue,
                 lock: mp.Lock,
                 stop_signal: mp.Event,
                 cam_image_unloaded: mp.Event,
                 cam_ready: mp.Event,
                 start_signal: mp.Event,
                 n_frames: mp.Value,
                 cam_frame_timestamp: mp.Value):
        self.cam_name = cam_name
        self.shm = shm
        self.cam_config = cam_config
        self.log_queue = log_queue
        self.lock = lock
        self.stop_signal = stop_signal
        self.cam_image_unloaded = cam_image_unloaded
        self.cam_ready = cam_ready
        self.start_signal = start_signal
        self.n_frames = n_frames
        self.cam_frame_timestamp = cam_frame_timestamp
        super().__init__()
        self.log = get_process_logger(str(self), self.log_queue)

    def run(self):
        raise NotImplemented('')


class Camera(ArenaProcess):
    def __str__(self):
        return f'Cam-{self.cam_name}'

    def is_streaming(self):
        return cache.get(cc.STREAM_CAMERA) == self.cam_name


class ImageHandler(ArenaProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.durations = []

    def __str__(self):
        return f'ImageHandler {self.cam_name}'

    def run(self):
        if self.start_signal is not None:
            self.start_signal.wait()
        self.cam_ready.wait(timeout=3)
        self.log.info('Start frame handling')
        while True:
            if self.stop_signal.is_set():
                self.log.info('stop event detected')
                break

            if self.cam_image_unloaded.is_set():
                continue

            with self.lock:
                t0 = time.time()
                frame = np.frombuffer(self.shm.buf, dtype=config.shm_buffer_dtype).reshape(self.cam_config['image_size'])
                timestamp = self.cam_frame_timestamp.value
                self.durations.append(time.time() - t0)
                self.cam_image_unloaded.set()

            self.handle(frame, timestamp)

        self._on_end()

    def handle(self, frame, timestamp):
        raise NotImplemented('No handle method')

    def _on_start(self):
        pass

    def _on_end(self):
        pass


@dataclass
class CameraUnit:
    name: str
    cam_cls: Camera
    global_start: mp.Event
    global_stop: mp.Event
    cam_config: dict
    num_frames: int
    log_queue: mp.Queue

    def __post_init__(self):
        self.shm = None
        self.is_on = False
        self.processes = []
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.mp_utils = {
            'lock': mp.Lock(),
            'stop_signal': mp.Event(),
            'start_signal': self.global_start,
            'cam_image_unloaded': mp.Event(),
            'cam_ready': mp.Event(),
            'n_frames': mp.Value('i', 0),
            'cam_frame_timestamp': mp.Value("d", 0.0)
        }
        self.logger = get_process_logger(str(self), self.log_queue)
        self.processes_cls = self.get_process_classes()
        self.listen_stop_events()

    def __str__(self):
        return f'CU-{self.name}'

    def __del__(self):
        if self.is_on:
            self.stop()
        self.shm_manager.shutdown()

    def start(self):
        self.logger.info('start camera unit')
        self.mp_utils['stop_signal'].clear()
        self.shm = self.shm_manager.SharedMemory(size=int(np.prod(self.cam_config['image_size'])))

        for proc_cls in self.processes_cls:
            proc = proc_cls(self.name, self.shm, self.cam_config, self.log_queue, **self.mp_utils)
            proc.start()
            self.processes.append(proc)

        self.listen_stop_events()
        cache.append_to_list(cc.ACTIVE_CAMERAS, self.name)
        self.is_on = True

    def stop(self):
        cache.remove_from_list(cc.ACTIVE_CAMERAS, self.name)
        self.mp_utils['stop_signal'].set()
        [proc.join() for proc in self.processes]
        self.is_on = False
        self.logger.info('unit stopped')

    def listen_stop_events(self):
        def listener():
            while not self.mp_utils['stop_signal'].is_set():
                if self.global_stop.is_set():
                    self.stop()
                    break
                time.sleep(0.01)

        t = threading.Thread(target=listener)
        t.start()

    def stream(self):
        while not self.mp_utils['stop_signal'].is_set():
            img = np.frombuffer(self.shm.buf, dtype=config.shm_buffer_dtype).reshape(self.cam_config['image_size'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            (flag, encodedImage) = cv2.imencode(".jpg", img)

            if not flag:
                continue

            time.sleep(0.03)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n\r\n')

    def is_stream_camera(self):
        self.logger.info(f'{cache.get(cc.STREAM_CAMERA)} - {self.name}')
        return cache.get(cc.STREAM_CAMERA) == self.name

    def get_process_classes(self):
        listeners = [self.cam_cls]
        self.logger.info(self.is_stream_camera())
        if not self.is_stream_camera() or 1:
            for lst in self.cam_config['listeners']:
                module_path, class_name = config.arena_modules[lst]
                listeners.append(getattr(importlib.import_module(module_path), class_name))
        return listeners


class ArenaManager(SyncManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) #address=(config.arena_manager_address, config.arena_manager_port),
                         #authkey=config.arena_manager_password.encode('utf-8'), **kwargs)
        self.start()
        self.detected_cameras = {}
        self.camera_modules = []
        self.units = {}  # activated camera units
        self.threads = {}
        self.experiment_threads = []
        self.reset_cache()

        self.global_stop_event = self.Event()
        self.global_start_event = self.Event()
        self.arena_shutdown_event = self.Event()
        self.stop_logging_event = self.Event()
        self.log_queue = self.Queue(-1)
        self.logging_thread = logger_thread(self.log_queue, self.stop_logging_event)
        self.logger = get_process_logger('ArenaManagement', self.log_queue)
        # self.log = get_logger('Arena Management')
        self.cache_lock = self.Lock()
        self._streaming_camera = None

        self.start_management_listeners()
        self.camera_init()
        self.logger.info('Arena manager created.')
        atexit.register(self.arena_shutdown)

    def __del__(self):
        sys.exit(0)

    def camera_init(self):
        """Detect connected cameras"""
        configured_modules = list(set([cam_config['module'] for cam_config in config.cameras.values()]))
        self.logger.info(f'Configured camera modules: {configured_modules}')
        for module_name in configured_modules:
            try:
                cam_module, cam_class = config.arena_modules['cameras'][module_name]
                cam_module = importlib.import_module(cam_module)
                self.camera_modules.append(cam_module)
                # each camera module must include an init function,
                # which returns the Camera Processes according to the config
                info_df = cam_module.scan_cameras()
                self.detected_cameras.update({cam_name: getattr(cam_module, cam_class) for cam_name in info_df.index})
            except Exception as exc:
                raise Exception(f'unable to load camera module: {module_name}; {exc}')

    def start_experiment(self, **kwargs):
        exposure = kwargs.get('exposure')
        cameras = kwargs.get('cameras')
        e = Experiment(**kwargs)
        self.start_experiment_listeners()
        self.record(exposure=exposure, cameras=cameras)
        time.sleep(0.1)
        e.start()

    def start_management_listeners(self):
        subs_dict = {
            'start_recording': self.start_recording,
            'stop_recording': self.stop_recording,
            'arena_shutdown': self.arena_shutdown
        }
        for topic, callback in subs_dict.items():
            self.threads[topic] = Subscriber(self.arena_shutdown_event, self.log_queue,
                                             config.subscription_topics[topic], callback)
            self.threads[topic].start()

        self.threads['arena_operations'] = ArenaOperations(self.arena_shutdown_event, self.log_queue)
        self.threads['arena_operations'].start()

    def start_experiment_listeners(self):
        # start experiment listen   ers
        for channel_name, d in config.experiment_metrics.items():
            thread_name = f'metric_{channel_name}'
            self.threads[thread_name] = MetricsLogger(self.arena_shutdown_event, self.log_queue,
                                                      channel=config.subscription_topics[channel_name])
        # self.threads['app_healthcheck'] = AppHealthCheck(self.arena_shutdown_event, self.log_queue)
        # self.threads['app_healthcheck'].start()

    def reset_cache(self):
        for name, col in cc.__dict__.items():
            if name.startswith('_'):
                continue
            cache.delete(col)

    def record(self, exposure=None, cameras=None, output_dir=None, folder_prefix=None,
               num_frames=None, rec_time=None, is_streamer=False):
        """
        Record videos from Arena's cameras
        :param exposure: (int) Exposure time to be set to cameras
        :param cameras: (str) Cameras to be used. You can specify last digits of p/n or name. (for more than 1 use ',')
        :param output_dir: Output dir for videos and timestamps, if not exist save into a timestamp folder in default output dir.
        :param folder_prefix: Prefix to be added to folder name. Not used if output is given.
        :param num_frames: Limit number of frames to be taken by each camera
        :param rec_time: Limit the recording time (seconds)
        """
        assert not (num_frames and rec_time), 'you can not set num_frames and rec_time together'
        if cache.get(cc.IS_RECORDING):
            self.logger.exception('Another recording is happening, can not initiate a new record')
            return

        self.global_start_event.clear()
        self.global_stop_event.clear()
        record_cameras = {cn: ccls for cn, ccls in self.detected_cameras.items() if (not cameras or cn in cameras)}
        for cam_name, cam_class in record_cameras.items():
            cam_config = config.cameras[cam_name].copy()
            if exposure:
                cam_config['exposure'] = exposure
            cam_config['output_dir'] = self.check_output_dir(output_dir, folder_prefix)
            cu = CameraUnit(cam_name, cam_class, self.global_start_event, self.global_stop_event,
                            cam_config, num_frames, self.log_queue)
            cu.start()
            self.units[cam_name] = cu

        if rec_time or num_frames or is_streamer:
            time.sleep(0.1)
            self.start_recording()

        if rec_time:
            threading.Timer(rec_time, self.stop_recording).start()

    def start_recording(self):
        if cache.get(cc.IS_RECORDING):
            self.logger.warning('recording in progress. can not start a new record')
            return
        cache.set(cc.IS_RECORDING, True)
        self.global_start_event.set()
        self.logger.info('start recording signal was sent.')

    def stop_recording(self):
        self.logger.warning('stop recording signal was sent.')
        if cache.get(cc.IS_RECORDING):
            self.global_stop_event.set()
            cache.set(cc.IS_RECORDING, False)
            self.logger.info('closing record...')

    def arena_shutdown(self) -> None:
        self.logger.warning('shutdown start')
        self.logger.info(f'NUm of threads: {len(self.threads)}')
        self.arena_shutdown_event.set()
        self.stop_recording()
        [cu.stop() for cu in self.units.values()]
        for name, t in self.threads.items():
            if threading.current_thread().name != t.name:
                try:
                    t.join()
                    self.logger.info(f'thread {name} is down')
                except:
                    self.logger.exception(f'Error joining thread {name}')
        self.units, self.threads = {}, {}
        self.logger.info('closing logging thread')
        self.stop_logging_event.set()
        self.logging_thread.join()
        self.shutdown()
        print('shutdown finished')

    def display_info(self, return_string=False):
        if not self.camera_modules:
            return
        info_df = self.camera_modules[0].scan_cameras()
        with pd.option_context('display.max_colwidth', None,
                               'display.max_columns', None,
                               'display.max_rows', None):
            self.logger.info(f'\n{info_df}')
        if return_string:
            return f'\nCameras Info:\n\n{info_df.to_string()}\n\n'
        return info_df

    def set_streaming_camera(self, cam_name):
        cache.set(cc.STREAM_CAMERA, cam_name)
        if cam_name not in self.units:
            self.record(cameras=[cam_name], is_streamer=True)
        if not self.units[cam_name].is_on:
            self.units[cam_name].start()
        self._streaming_camera = cam_name
        self.logger.info(f'Set streaming camera to {cam_name}')

    def stream(self):
        return self.units[self._streaming_camera].stream()

    def stop_stream(self):
        if self._streaming_camera is not None:
            self.units[self._streaming_camera].stop()
        cache.delete(cc.STREAM_CAMERA)
        self._streaming_camera = None
        self.logger.info('streaming stopped')

    @staticmethod
    def check_output_dir(output, folder_prefix):
        if not output:
            folder_name = datetime_string()
            if folder_prefix:
                folder_name = f'{folder_prefix}_{folder_name}'
            output = f"{config.output_dir}/{folder_name}"
        return mkdir(output)


# def main():
#     """Main function for Arena capture"""
#     ap = argparse.ArgumentParser(description="Tool for capturing multiple cameras streams in the arena.")
#     ap.add_argument("-n", "--num_frames", type=int, help=f"Specify Number of Frames.")
#     ap.add_argument("-t", "--record_time", type=int, help=f"Specify record duration in seconds.")
#     ap.add_argument("-m", "--manual_stop", action="store_true", default=False,
#                     help=f"Stop record using cache key MANUAL_RECORD_STOP.")
#     ap.add_argument("--experiment_alive", action="store_true", default=False,
#                     help=f"Stop record if the experiment ended")
#     ap.add_argument("-o", "--output", type=str, default=config.output_dir,
#                     help=f"Specify output directory path. Default={config.output_dir}")
#     ap.add_argument("-e", "--exposure", type=int, default=config.exposure_time,
#                     help=f"Specify cameras exposure time. Default={config.exposure_time}")
#     ap.add_argument("-c", "--camera", type=str, required=False,
#                     help=f"filter cameras by last digits or according to CAMERA_NAMES (for more than one use ',').")
#     ap.add_argument("-i", "--info", action="store_true", default=False,
#                     help=f"Show cameras information")
#
#     args = vars(ap.parse_args())
#
#     if args.get('info'):
#         print(display_info())
#     else:
#         acquire_stop = {}
#         for key in config.acquire_stop_options:
#             if key in args:
#                 acquire_stop[key] = args[key]
#         record(args.get('exposure'), args.get('camera'), args.get('output'), **acquire_stop)


if __name__ == '__main__':
    mgr = ArenaManager()
    mgr.record()
