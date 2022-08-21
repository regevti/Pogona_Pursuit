#!/usr/bin/env python
import time
import cv2
import re
import importlib
import threading
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager, SyncManager
import numpy as np

import config
from cache import RedisCache, CacheColumns as cc
from utils import mkdir, datetime_string, Serializer
from loggers import get_logger
from experiment import Experiment
from subscribers import Subscriber, MetricsLogger, ArenaOperations

cache = RedisCache()


class ArenaProcess(mp.Process):
    def __init__(self, shm: SharedMemory, lock: mp.Lock, name: str,
                 cam_ready: mp.Event, image_unloaded: mp.Event, start_event: mp.Event,
                 stop_event: mp.Event, frame_timestamp: mp.Value, cam_config: dict,
                 output_dir: str):
        super().__init__()
        self.shm = shm
        self.lock = lock
        self.name = name
        self.log = get_logger(str(self))
        self.stop_event = stop_event
        self.start_event = start_event
        self.cam_ready = cam_ready
        self.image_unloaded = image_unloaded
        self.frame_timestamp = frame_timestamp
        self.cam_config = cam_config
        self.output_dir = output_dir

    def run(self):
        raise NotImplemented('')


class Camera(ArenaProcess):
    def __str__(self):
        return f'Camera {self.name}'


class ImageHandler(ArenaProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.durations = []

    def __str__(self):
        return f'Image Handler {self.name}'

    def run(self):
        if self.start_event is not None:
            self.start_event.wait()
        self.cam_ready.wait(timeout=3)
        self.log.info('Start frame handling')
        while True:
            if self.stop_event.is_set():
                self.log.info('stop event detected')
                break

            if self.image_unloaded.is_set():
                continue

            with self.lock:
                t0 = time.time()
                frame = np.frombuffer(self.shm.buf, dtype=config.shm_buffer_dtype).reshape(self.cam_config['image_size'])
                timestamp = self.frame_timestamp.value
                self.durations.append(time.time() - t0)
                self.image_unloaded.set()

            self.handle(frame, timestamp)

        self._on_end()

    def handle(self, frame, timestamp):
        raise NotImplemented('No handle method')

    def _on_start(self):
        pass

    def _on_end(self):
        pass


class CameraUnit:
    def __init__(self, name, cam_cls, start_event, global_stop, cam_config, output_dir, num_frames):
        self.name = name
        self.start_event = start_event
        self.global_stop = global_stop
        self.processes = []
        self.shm_manager = SharedMemoryManager()
        self.cam_image_unloaded = mp.Event()
        self.cam_ready = mp.Event()
        self.shm = None
        self.stop_signal = mp.Event()
        self.n_frames = mp.Value('i', 0)
        self.listen_stop_events()
        self.cam_config = cam_config
        self.processes_cls = self.get_process_classes(cam_cls)
        self.output_dir = output_dir
        self.num_frames = num_frames

    def get_process_classes(self, cam_cls):
        listeners = [cam_cls]
        for lst in self.cam_config['listeners']:
            module_path, class_name = config.arena_modules[lst]
            listeners.append(getattr(importlib.import_module(module_path), class_name))
        return listeners

    def __del__(self):
        self.stop()

    def start(self):
        self.shm_manager.start()
        self.shm = self.shm_manager.SharedMemory(size=int(np.prod(self.cam_config['image_size'])))
        lock = mp.Lock()

        cam_frame_timestamp = mp.Value("d", 0.0)

        for proc_cls in self.processes_cls:
            proc = proc_cls(self.shm, lock, self.name, self.cam_ready, self.cam_image_unloaded,
                            self.start_event, self.stop_signal, cam_frame_timestamp, self.cam_config, self.output_dir)
            proc.start()
            self.processes.append(proc)

    def stop(self):
        self.stop_signal.set()
        [proc.join() for proc in self.processes]
        self.shm_manager.shutdown()

    def listen_stop_events(self):
        def listener():
            while not self.stop_signal.is_set():
                if self.global_stop.is_set():
                    self.stop()
                    break
                time.sleep(0.01)

        t = threading.Thread(target=listener)
        t.start()

    def stream(self):
        while not self.stop_signal.is_set():
            img = np.frombuffer(self.shm.buf, dtype=config.shm_buffer_dtype).reshape(self.cam_config['image_size'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            (flag, encodedImage) = cv2.imencode(".jpg", img)

            if not flag:
                continue

            time.sleep(0.03)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n\r\n')


class ArenaManager(SyncManager):
    def __init__(self, **kwargs):
        super().__init__(address=(config.arena_manager_address, config.arena_manager_port),
                         authkey=config.arena_manager_password.encode('utf-8'), **kwargs)
        self.start()
        self.detected_cameras = {}
        self.camera_modules = []
        self.units = {}  # activated camera units
        self.threads = {}

        self.log = get_logger('Arena Management')
        self.global_stop_event = self.Event()
        self.global_start_event = self.Event()
        self.arena_shutdown_event = self.Event()
        self.cache_lock = self.Lock()
        self._streaming_camera = None

        self.start_management_listeners()
        self.camera_init()
        self.log.info('Arena manager created.')

    def camera_init(self):
        """Detect connected cameras"""
        configured_modules = list(set([cam_config['module'] for cam_config in config.cameras.values()]))
        self.log.info(f'Configured camera modules: {configured_modules}')
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

    def start_management_listeners(self):
        subs_dict = {
            'start_recording': self.start_recording,
            'stop_recording': self.stop_recording,
            'arena_shutdown': self.shutdown
        }
        for topic, callback in subs_dict.items():
            self.threads[topic] = Subscriber(self.arena_shutdown_event, config.subscription_topics[topic], callback)
            self.threads[topic].start()

        self.threads['arena_operations'] = ArenaOperations(self.arena_shutdown_event)
        self.threads['metrics_logger'] = MetricsLogger(self.arena_shutdown_event)

    def log_temperature(self):
        if self.threads.get('temp') is not None and self.threads['temp'].is_alive():
            self.log.warning('Another temperature logger is already running')
            return

        def _rec_temp():
            ser = Serializer()
            self.log.info('read_temp started')
            while cache.get_current_experiment() and not self.arena_shutdown_event.is_set():
                try:
                    line = ser.read_line()
                    if line and isinstance(line, bytes):
                        m = re.search(r'Temperature is: ([\d.]+)', line.decode())
                        if m:
                            cache.publish(config.subscription_topics['temperature'], m[1])
                except Exception as exc:
                    self.log.exception(f'Error in read_temp: {exc}')
                time.sleep(5)

        self.threads['temp'] = threading.Thread(target=_rec_temp)
        self.threads['temp'].start()

    def record(self, exposure=None, cameras=None, output_dir=None, folder_prefix=None,
               num_frames=None, rec_time=None):
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
            self.log.exception('Another recording is happening, can not initiate a new record')
            return

        self.global_start_event.clear()
        self.global_stop_event.clear()
        record_cameras = {cn: ccls for cn, ccls in self.detected_cameras.items() if (not cameras or cn in cameras)}
        for cam_name, cam_class in record_cameras.items():
            cam_config = config.cameras[cam_name].copy()
            if exposure:
                cam_config['exposure'] = exposure
            output_dir = self.check_output_dir(output_dir, folder_prefix)
            cu = CameraUnit(cam_name, cam_class, self.global_start_event, self.global_stop_event,
                            cam_config, output_dir, num_frames)
            cu.start()
            self.units[cam_name] = cu

        if rec_time or num_frames:
            time.sleep(0.1)
            self.start_recording()

        if rec_time:
            threading.Timer(rec_time, self.stop_recording).start()

    def start_recording(self):
        if cache.get(cc.IS_RECORDING):
            self.log.warning('recording in progress. can not start a new record')
            return
        cache.set(cc.IS_RECORDING, True)
        self.global_start_event.set()
        self.log.info('start recording signal was sent.')

    def stop_recording(self):
        if cache.get(cc.IS_RECORDING):
            self.global_stop_event.set()
            cache.set(cc.IS_RECORDING, False)
            self.log.info('stop recording signal was sent.')

    def shutdown(self) -> None:
        self.arena_shutdown_event.set()
        self.stop_recording()
        [cu.stop() for cu in self.units.values()]
        [lst.join() for lst in self.threads]
        self.units = {}
        self.log.info('shutdown')
        super().shutdown()

    def start_experiment(self, **kwargs):
        exposure = kwargs.get('exposure')
        cameras = kwargs.get('cameras')
        e = Experiment(**kwargs)
        self.record(exposure=exposure, cameras=cameras)
        time.sleep(0.1)
        e.start()
        self.log_temperature()

    def display_info(self, return_string=False):
        info_df = self.camera_modules[0].scan_cameras()
        if return_string:
            return f'\nCameras Info:\n\n{info_df.to_string()}\n'
        return info_df

    def set_streaming_camera(self, cam_name):
        if cam_name not in self.units:
            self.record(cameras=[cam_name])
        
        if not self.units[cam_name].cam_ready.is_set():
            self.units[cam_name].start()
        self._streaming_camera = cam_name
        self.log.info(f'Set streaming camera to {cam_name}')

    def stream(self):
        return self.units[self._streaming_camera].stream()

    def stop_stream(self):
        if self._streaming_camera is not None:
            self.units[self._streaming_camera].stop()
        self._streaming_camera = None

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
