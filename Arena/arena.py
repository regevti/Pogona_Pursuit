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
from arrayqueues.shared_arrays import TimestampedArrayQueue, Empty
import numpy as np
import pandas as pd
from typing import Type

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
    def __init__(self, cam_name: str, frames_queue: TimestampedArrayQueue, cam_config: dict, log_queue: mp.Queue,
                 stop_signal: mp.Event):
        self.cam_name = cam_name
        self.frames_queue = frames_queue
        self.cam_config = cam_config
        self.log_queue = log_queue
        self.stop_signal = stop_signal
        self.cam_config = cache.get_cam_dict(self.cam_name)
        super().__init__()
        self.logger = get_process_logger(str(self), self.log_queue)

    def run(self):
        raise NotImplemented('')


class Camera(ArenaProcess):
    def __str__(self):
        return f'Cam-{self.cam_name}'


class ImageSink(ArenaProcess):
    def __init__(self, *args, shm: SharedMemory = None):
        super().__init__(*args)
        self.shm = shm
        self.video_out = None
        self.write_video_counter = 0
        self.write_first_frame_timestamp = None
        self.write_video_path = None
        self.mean_duration = 0
        self.mean_frames_offset = 0
        self.prev_timestamp = None

    def __str__(self):
        return f'Sink-{self.cam_name}'

    def run(self):
        self.logger.info('Start frame handling in ImageSink')
        while not self.stop_signal.is_set():
            try:
                t0 = time.time()
                self.cam_config = cache.get_cam_dict(self.cam_name)
                timestamp, frame = self.frames_queue.get(timeout=2)
                if self.prev_timestamp is None:
                    self.prev_timestamp = timestamp

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.write_to_shm(frame)
                self.write_to_video_file(frame, timestamp)
                self.mean_duration += (time.time() - t0)
                self.mean_frames_offset += (timestamp - self.prev_timestamp)
                self.prev_timestamp = timestamp

            except Empty:
                self.logger.warning('Empty queue')
                self.stop_signal.set()
                break

            except Exception:
                self.logger.exception('Error in sink')
                self.stop_signal.set()
                break

        self.logger.info('sink stopped')

    def write_to_shm(self, frame):
        buf_np = np.frombuffer(self.shm.buf, dtype=config.shm_buffer_dtype).reshape(self.cam_config['image_size'])
        np.copyto(buf_np, frame)

    def write_to_video_file(self, frame, timestamp):
        if not self.cam_config[config.output_dir_key]:
            if self.video_out is not None:
                self.close_video_out()
            return

        if self.video_out is None:
            self.init_video_out(frame, timestamp)
        self.video_out.write(frame)
        self.write_video_counter += 1

        if self.cam_config.get('num_frames') and self.write_video_counter >= int(self.cam_config['num_frames']):
            cache.set_cam_output_dir(self.cam_name, '')
        elif self.cam_config.get('rec_time') and \
                (timestamp - self.write_first_frame_timestamp) >= int(self.cam_config['rec_time']):
            cache.set_cam_output_dir(self.cam_name, '')

    def init_video_out(self, frame, timestamp):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        h, w = frame.shape[:2]
        self.write_video_path = self.cam_config[config.output_dir_key]
        self.logger.info(f'start video writing to {self.video_path}')
        self.video_out = cv2.VideoWriter(self.video_path, fourcc, int(self.cam_config['fps']), (w, h), True)
        self.write_first_frame_timestamp = timestamp
        self.mean_duration = 0
        self.mean_frames_offset = 0

    def close_video_out(self):
        self.video_out.release()
        wr = int(1 / (self.mean_duration / self.write_video_counter))
        fr = int(1 / (self.mean_frames_offset / self.write_video_counter))
        self.logger.info(f'Video with {self.write_video_counter} frames (frame rate: {fr}fps, writing rate: {wr}fps) '
                         f'saved into {self.video_path}')
        self.video_out = None
        self.write_video_counter = 0
        self.write_first_frame_timestamp = None
        self.write_video_path = None


    @property
    def video_path(self):
        return f'{self.write_video_path}/{self.cam_name}_{datetime_string()}.avi'


class CameraUnit:

    def __init__(self, name: str, cam_cls: Type[Camera], global_start: mp.Event, global_stop: mp.Event, cam_config: dict,
                 log_queue: mp.Queue):
        self.cam_name = name
        self.cam_cls = cam_cls
        self.global_start = global_start
        self.global_stop = global_stop
        self.cam_config = cam_config
        self.log_queue = log_queue
        self.processes = []
        self.frames_queue = TimestampedArrayQueue(config.array_queue_size_mb)
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.shm = self.shm_manager.SharedMemory(size=int(np.prod(self.cam_config['image_size'])))
        self.stop_signal = mp.Event()
        self.stop_signal.set()
        self.logger = get_process_logger(str(self), self.log_queue)
        self.listen_stop_events()
        self.is_stopping = False

    def __str__(self):
        return f'CU-{self.cam_name}'

    def __del__(self):
        if not self.stop_signal.is_set():
            self.stop()
        self.shm_manager.shutdown()

    def start(self):
        if self.processes:
            return
        self.logger.info('start camera unit')
        self.stop_signal.clear()
        cache.delete_cam_dict(self.cam_name)
        cache.update_cam_dict(self.cam_name, **self.cam_config)
        args = (self.cam_name, self.frames_queue, self.cam_config, self.log_queue, self.stop_signal)
        cam = self.cam_cls(*args)
        sink = ImageSink(*args, shm=self.shm)
        self.processes.extend([cam, sink])

        self.listen_stop_events()
        [proc.start() for proc in self.processes]
        cache.append_to_list(cc.ACTIVE_CAMERAS, self.cam_name)

    def stop(self):
        if self.is_stopping or not self.is_on():
            return
        self.is_stopping = True
        cache.remove_from_list(cc.ACTIVE_CAMERAS, self.cam_name)
        self.stop_signal.set()
        [proc.join() for proc in self.processes]
        self.processes = []
        self.logger.info('unit stopped')
        self.is_stopping = False

    def listen_stop_events(self):
        def listener():
            while True:
                if self.global_stop.is_set() or self.stop_signal.is_set():
                    self.stop()
                    break
                time.sleep(0.01)

        t = threading.Thread(target=listener)
        t.start()

    def stream(self):
        """iterator for streaming frames to web UI"""
        while not self.stop_signal.is_set():
            img = self.get_frame()
            (flag, encodedImage) = cv2.imencode(".jpg", img)

            if not flag:
                continue

            time.sleep(0.01)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n\r\n')

    def get_frame(self):
        img = np.frombuffer(self.shm.buf, dtype=config.shm_buffer_dtype).reshape(self.cam_config['image_size'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def is_on(self):
        return not self.stop_signal.is_set()

    def is_recording(self):
        d = cache.get_cam_dict(self.cam_name)
        return bool(d.get(config.output_dir_key))


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
        self.arena_start()

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

    def arena_start(self):
        record_cameras = {cn: ccls for cn, ccls in self.detected_cameras.items() if cn in config.cameras}
        for cam_name, cam_class in record_cameras.items():
            cam_config = config.cameras[cam_name].copy()
            cu = CameraUnit(cam_name, cam_class, self.global_start_event, self.global_stop_event,
                            cam_config, self.log_queue)
            self.units[cam_name] = cu
        self.logger.info('Arena started')

    def arena_shutdown(self) -> None:
        self.logger.warning('shutdown start')
        self.logger.info(f'NUm of threads: {len(self.threads)}')
        self.arena_shutdown_event.set()
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

    def start_experiment(self, **kwargs):
        if not kwargs.get('cameras'):
            self.logger.error('unable to start experiment with no cameras specified')
            return
        e = Experiment(**kwargs)
        self.start_experiment_listeners()
        time.sleep(0.1)
        e.start()

    def start_management_listeners(self):
        subs_dict = {
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
            self.threads[thread_name].start()
        # self.threads['app_healthcheck'] = AppHealthCheck(self.arena_shutdown_event, self.log_queue)
        # self.threads['app_healthcheck'].start()

    def reset_cache(self):
        for name, col in cc.__dict__.items():
            if name.startswith('_'):
                continue
            cache.delete(col)

    def record(self, cameras=None, folder_prefix=None, output_dir=None, num_frames=None, rec_time=None,
               is_use_predictions=None):
        """
        Record videos from Arena's cameras
        :param cameras: (str) Cameras to be used. You can specify last digits of p/n or name. (for more than 1 use ',')
        :param output_dir: Output dir for videos and timestamps, if not exist save into a timestamp folder in default output dir.
        :param folder_prefix: Prefix to be added to folder name. Not used if output is given.
        :param num_frames: Limit number of frames to be taken by each camera
        :param rec_time: Limit the recording time (seconds)
        """
        assert not (num_frames and rec_time), 'you can not set num_frames and rec_time together'

        for cam_name, cu in self.units.items():
            if not cu.is_on() or (cameras and cam_name not in cameras):
                continue

            cache.set_cam_output_dir(cam_name, self.get_output_dir(output_dir, folder_prefix))
            cache.update_cam_dict(cam_name, num_frames=num_frames)
            cache.update_cam_dict(cam_name, rec_time=rec_time)

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

    def get_frame(self, cam_name):
        return self.units[cam_name].get_frame()

    def set_streaming_camera(self, cam_name):
        self._streaming_camera = cam_name

    def stop_stream(self):
        self._streaming_camera = None

    def stream(self):
        return self.units[self._streaming_camera].stream()

    def stop_recording(self):
        for cam_name in self.units.keys():
            cache.set_cam_output_dir(cam_name, '')

    @staticmethod
    def get_output_dir(output_dir, folder_prefix):
        folder_name = datetime_string()
        if folder_prefix != '':
            folder_name = f'{folder_prefix}_{folder_name}'
        output = f"{output_dir or config.output_dir}/{folder_name}"
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
