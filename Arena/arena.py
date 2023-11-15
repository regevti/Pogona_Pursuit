#!/usr/bin/env python
import datetime
import sys
import time
import cv2
import importlib
import atexit
import signal
import queue
import threading
import sentry_sdk
from pathlib import Path
import torch.multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager, SyncManager

from arrayqueues.shared_arrays import TimestampedArrayQueue, Empty
import numpy as np
import pandas as pd
from typing import Type

import config
from cache import RedisCache, CacheColumns as cc
from utils import mkdir, datetime_string, run_in_thread
from loggers import get_process_logger, logger_thread
from experiment import Experiment, ExperimentCache
from subscribers import start_management_subscribers, start_experiment_subscribers
from db_models import ORM
from scheduler import Scheduler
from image_handlers.video_writers import OpenCVWriter, ImageIOWriter

cache = RedisCache()


def signal_handler(signum, frame):
    print('signal detected!')
    cache.publish_command('arena_shutdown')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class ArenaException(Exception):
    """"""


class QueueException(ArenaException):
    """"""


class ArenaProcess(mp.Process):
    #  change that to either "cam_fps", "sink_fps" or "pred_fps" to make the process to log to the specified fps value
    calc_fps_name = None

    def __init__(self, cam_name: str, frames_queue: TimestampedArrayQueue, cam_config: dict, log_queue: mp.Queue,
                 stop_signal: mp.Event, mp_metadata: dict):
        self.cam_name = cam_name
        self.frames_queue = frames_queue
        self.cam_config = cam_config
        self.log_queue = log_queue
        self.stop_signal = stop_signal
        self.mp_metadata = mp_metadata
        self.cam_config = cache.get_cam_dict(self.cam_name)
        self.timestamps_stack = []
        self.delays_stack = []
        super().__init__()
        self.logger = None

    def run(self):
        self.logger = get_process_logger(str(self), self.log_queue)
        try:
            self._run()
        except ArenaException as exc:
            self.logger.error(str(exc))
        except Exception as exc:
            sentry_sdk.capture_exception(exc)
            self.logger.exception(exc)
        finally:
            if self.calc_fps_name:
                self.clear_fps()

    def _run(self):
        raise NotImplemented('_run is not implemented')

    def calc_fps(self, timestamp):
        fps = 0.0
        self.timestamps_stack.append(timestamp)
        if len(self.timestamps_stack) > config.count_timestamps_for_fps_calc:
            fps = 1 / np.diff(self.timestamps_stack).mean()
            self.timestamps_stack.pop(0)

        self.mp_metadata[self.calc_fps_name].value = fps

    def calc_pred_delay(self, timestamp, pred_timestamp):
        dt = (pred_timestamp - timestamp) * 1000
        self.delays_stack.append(dt)
        if len(self.delays_stack) > config.count_timestamps_for_fps_calc:
            self.delays_stack.pop(0)
        dt = np.mean(self.delays_stack)
        self.mp_metadata['pred_delay'].value = dt

    def clear_fps(self):
        self.mp_metadata[self.calc_fps_name].value = 0.0

    @property
    def is_streaming(self):
        return self.mp_metadata['is_streaming'].is_set()


class Camera(ArenaProcess):
    calc_fps_name = 'cam_fps'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_time_delta = 0.0
        self.last_queue_warning_time = None

    def __str__(self):
        return f'Cam-{self.cam_name}'

    def update_time_delta(self, cam):
        cam.TimestampLatch.run()
        cam_time = cam.TimestampLatchValue.get()  # in nanosecs
        server_time = time.time_ns()
        self.camera_time_delta = (server_time - cam_time) / 1e9


class ImageSink(ArenaProcess):
    calc_fps_name = 'sink_fps'

    def __init__(self, *args, shm: SharedMemory = None):
        super().__init__(*args)
        self.shm = shm
        self.orm = None
        self.video_out = None
        self.write_video_timestamps = []
        self.write_output_dir = None
        self.video_path = None
        self.db_video_id = None
        self.writing_queue = None
        self.writing_thread = None
        self.writing_stop_event = mp.Event()

    def __str__(self):
        return f'Sink-{self.cam_name}'

    def _run(self):
        self.logger.debug('Start frame handling in ImageSink')
        self.orm = ORM()
        self.writing_queue = queue.Queue(maxsize=config.writing_video_queue_maxsize)
        frames_counter = 0
        while not self.stop_signal.is_set():
            try:
                self.cam_config = cache.get_cam_dict(self.cam_name)
                t0 = time.time()
                timestamp, frame = self.frames_queue.get(timeout=config.SINK_QUEUE_TIMEOUT)
                frames_counter += 1
                if time.time() - t0 > 1:
                    self.logger.info(f'got frame in sink after waiting {time.time() - t0:.3f} sec (# frames: {frames_counter})')
                    frames_counter = 0
                if self.cam_config.get('crop'):
                    x, y, w, h = [int(c) for c in self.cam_config['crop']]
                    frame = frame[y:y+h, x:x+w]

                if self.cam_config.get('is_color') and self.cam_config.get('module') != 'flir':
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.write_to_shm(frame, timestamp)
                self.write_to_video_file(frame, timestamp)
                self.calc_fps(time.time())

            except Empty:
                self.logger.error('Empty queue')
                self.stop_signal.set()
                break

            except Exception:
                self.logger.exception('Error in sink')
                self.stop_signal.set()
                break

        if self.video_out is not None:
            self.close_video_out()
        self.logger.debug('sink stopped')

    def write_to_shm(self, frame, timestamp):
        datetime_text = datetime.datetime.fromtimestamp(timestamp).strftime('%d/%m/%Y %H:%M:%S')
        frame = cv2.putText(frame, datetime_text, (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 0, 255), 2, cv2.LINE_AA)
        buf_np = np.frombuffer(self.shm.buf, dtype=config.shm_buffer_dtype).reshape(self.cam_config['image_size'])
        np.copyto(buf_np, frame)
        self.mp_metadata['shm_frame_timestamp'].value = timestamp

    def write_to_video_file(self, frame, timestamp):
        if self.cam_config.get('writing_fps') == '0':
            return
        output_path = self.cam_config[config.output_dir_key]
        if not output_path or (self.write_output_dir and self.write_output_dir != output_path):
            if self.video_out is not None:
                self.close_video_out()
            return

        if self.video_out is None:
            self.init_video_out(frame)

        try:
            if self.check_writing_fps(timestamp):
                self.writing_queue.put_nowait(frame)
                self.write_video_timestamps.append(timestamp)
        except queue.Full:
            pass
        except Exception as exc:
            self.logger.error(f'Error in image sink; {exc}')
            self.logger.info(f'writing FPS: {1/np.diff(self.write_video_timestamps).mean():.1f}')
            raise exc

        # check the n_frames and rec_time conditions
        n_frames = len(self.write_video_timestamps)
        rec_time = timestamp - self.write_video_timestamps[0] if self.write_video_timestamps else 0
        if (self.cam_config.get('num_frames') and n_frames >= int(self.cam_config['num_frames'])) or \
               (self.cam_config.get('rec_time') and rec_time >= int(self.cam_config['rec_time'])):
            cache.set_cam_output_dir(self.cam_name, '')
        # check if video exceeds maximum video duration
        if config.MAX_VIDEO_TIME_SEC and rec_time > config.MAX_VIDEO_TIME_SEC:
            self.close_video_out()

    def start_writing_thread(self):
        def loop(q):
            while not self.stop_signal.is_set() and not self.writing_stop_event.is_set():
                try:
                    frame = q.get_nowait()
                    self.video_out.write(frame)
                except queue.Empty:
                    pass
            self.logger.debug('video writing thread is closed')

        if self.writing_stop_event.is_set():
            self.writing_stop_event.clear()
        self.writing_thread = threading.Thread(target=loop, args=(self.writing_queue,))
        self.writing_thread.start()

    def init_video_out(self, frame):
        self.write_output_dir = self.cam_config[config.output_dir_key]
        is_color = self.cam_config.get('is_color', False)
        if cache.get(cc.IS_BLANK_CONTINUOUS_RECORDING) or self.cam_config.get('mode') == 'tracking':
            self.video_out = ImageIOWriter(frame, self.writing_fps, self.write_output_dir, self.cam_name, is_color)
        else:
            self.video_out = OpenCVWriter(frame, self.writing_fps, self.write_output_dir, self.cam_name, is_color)
        self.video_path = self.video_out.video_path
        self.logger.info(f'start video writing to {self.video_path} frame size: {frame.shape}')
        self.db_video_id = self.orm.commit_video(path=self.video_path, fps=self.writing_fps,
                                                 cam_name=self.cam_name, start_time=datetime.datetime.now())
        if self.db_video_id is not None:
            self.mp_metadata['db_video_id'].value = self.db_video_id
        self.write_video_timestamps = []
        self.start_writing_thread()

    def close_video_out(self):
        if self.writing_thread is not None:
            self.writing_stop_event.set()
            self.writing_thread.join()
            self.writing_thread = None
        self.video_out.close()
        calc_fps = 1 / np.diff(self.write_video_timestamps).mean()
        self.logger.info(f'Video with {len(self.write_video_timestamps)} frames and calc_fps={calc_fps:.1f} '
                         f'saved into {self.video_path}')
        if self.write_video_timestamps:
            self.commit_video_frames_to_db(self.video_path)
        self.video_out = None
        self.video_path = None
        self.write_output_dir = None
        self.db_video_id = None
        self.mp_metadata['db_video_id'].value = 0

    def check_writing_fps(self, timestamp):
        if not self.cam_config.get('writing_fps'):
            return True

        return not self.write_video_timestamps or \
               (timestamp - self.write_video_timestamps[-1]) >= (1 / float(self.cam_config['writing_fps'])) * 0.9

    @run_in_thread
    def commit_video_frames_to_db(self, video_path: str):
        self.orm.commit_video_frames(self.write_video_timestamps, self.db_video_id)
        # frames timestamp artifact
        frames_output_dir = Path(video_path).parent / config.frames_timestamps_dir
        frames_output_dir.mkdir(exist_ok=True)
        csv_path = frames_output_dir / Path(video_path).with_suffix('.csv').name
        pd.DataFrame(self.write_video_timestamps).to_csv(csv_path)
        self.logger.debug(f'Saved frames timestamps to: {csv_path}')

    @property
    def writing_fps(self):
        wf = self.cam_config.get('writing_fps')
        return int(wf) if wf is not None else None


class ImageHandler(ArenaProcess):
    """Parent class to all image handlers or predictors. Run all imports in the new process"""
    calc_fps_name = 'pred_fps'

    def __init__(self, predictor_name, *args, shm=None, pred_shm=None, pred_image_size=None):
        super().__init__(*args)
        self.proc_args = args
        self.shm = shm
        self.pred_shm = pred_shm
        self.predictor_name = predictor_name
        self.pred_image_size = pred_image_size
        self.predictor = None
        self.orm = None

    def __str__(self):
        return f'Image-Handler-{self.cam_name}'

    def _run(self):
        prd_module, prd_class = config.arena_modules['image_handlers'][self.predictor_name]
        prd_module = importlib.import_module(prd_module)
        prd_class = getattr(prd_module, prd_class)
        try:
            self.predictor = prd_class(self.predictor_name, *self.proc_args, shm=self.shm, pred_shm=self.pred_shm,
                                       pred_image_size=self.pred_image_size, logger=self.logger)
            self.predictor.loop()
        except RuntimeError as exc:
            self.logger.error(f'GPU out of memory. Close other GPU processes; {exc}')
            raise exc
        finally:
            self.logger.info('Process is down')

    def is_on(self):
        return self.mp_metadata['is_pred_on'].is_set()


class CameraUnit:

    def __init__(self, name: str, cam_cls: Type[Camera], global_start: mp.Event, global_stop: mp.Event, cam_config: dict,
                 log_queue: mp.Queue):
        self.cam_name = name
        self.cam_cls = cam_cls
        self.global_start = global_start
        self.global_stop = global_stop
        self.cam_config = cam_config
        self.log_queue = log_queue
        self.processes = {}
        self.is_stopping = False
        self.is_starting = False
        self.frames_queue = TimestampedArrayQueue(config.array_queue_size_mb)
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.shm = self.shm_manager.SharedMemory(size=int(np.prod(self.cam_config['image_size'])))
        self.pred_shm = {}
        self.stop_signal = mp.Event()
        self.mp_metadata = {  # Multiprocessing Metadata
            'cam_fps': mp.Value('d', 0.0),
            'sink_fps': mp.Value('d', 0.0),
            'pred_fps': mp.Value('d', 0.0),
            'is_pred_on': mp.Event(),
            'shm_frame_timestamp': mp.Value('d', 0.0),
            'db_video_id': mp.Value('i', 0),
            'pred_delay': mp.Value('d', 0.0),
            'is_streaming': mp.Event(),
            'predictors_stop': mp.Event()
        }
        self.stop_signal.set()
        self.logger = get_process_logger(str(self), self.log_queue)
        self.listen_stop_events()
        self.start_time = time.time()
        self.preds_start_time = None

    def __str__(self):
        return f'CU-{self.cam_name}'

    def __del__(self):
        if not self.stop_signal.is_set():
            self.stop()
        self.shm_manager.shutdown()

    def start(self, is_experiment=False, **kwargs):
        if self.is_starting:
            return
        self.is_starting = True
        try:
            if self.processes:
                alive_procs = [proc_name for proc_name, proc in self.processes.items() if proc.is_alive()]
                if any([pn not in alive_procs for pn in ['cam', 'sink']]):
                    self.stop_signal.set()
                    self.stop()
                else:
                    self.logger.debug(f'cannot stop camera-unit since the following processes are still alive: {alive_procs}')
                    return
            self.logger.debug('start camera unit')
            self.start_time = time.time()
            self.stop_signal.clear()
            cache.delete_cam_dict(self.cam_name)
            cache.update_cam_dict(self.cam_name, **self.cam_config)
            self.processes['cam'] = self.cam_cls(*self.proc_args)
            self.processes['sink'] = ImageSink(*self.proc_args, shm=self.shm)

            self.listen_stop_events()
            [proc.start() for proc in self.processes.values()]
            cache.append_to_list(cc.ACTIVE_CAMERAS, self.cam_name)
            self.start_predictors(is_experiment, **kwargs)
        except Exception as exc:
            self.logger.error(exc)
        finally:
            self.is_starting = False

    def stop(self):
        if self.is_stopping or not self.processes:
            return
        self.is_stopping = True
        cache.remove_from_list(cc.ACTIVE_CAMERAS, self.cam_name)
        self.stop_signal.set()
        [proc.join() for proc in list(self.processes.values())]
        self.processes = {}
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
        self.mp_metadata['is_streaming'].set()
        while not self.stop_signal.is_set():
            img = self.get_stream_frame()
            (flag, encodedImage) = cv2.imencode(".jpg", img)

            if not flag:
                continue

            time.sleep(0.01)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n\r\n')
        self.mp_metadata['is_streaming'].clear()

    def get_frame(self):
        img = np.frombuffer(self.shm.buf, dtype=config.shm_buffer_dtype).reshape(self.cam_config['image_size'])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_stream_frame(self):
        if not self.pred_shm or not self.get_alive_predictors():
            return self.get_frame()

        stream_pred = self.cam_config.get('stream_predictor')
        if not stream_pred:
            stream_pred = list(self.pred_shm.keys())[0]
            if len(self.pred_shm) > 1:
                self.logger.warning(f'More than 2 predictors configured, but no stream_predictor was defined. '
                                    f'using {stream_pred} for streaming')

        conf_preds = self.get_conf_predictors()
        pred_image_size = conf_preds[stream_pred]['image_size']
        img = np.frombuffer(self.pred_shm[stream_pred].buf, dtype=config.shm_buffer_dtype).reshape(pred_image_size)
        return img

    def start_predictors(self, is_experiment=False, **kwargs):
        predictors = self.cam_config.get('predictors', {})
        self.is_starting = True
        self.mp_metadata['predictors_stop'].clear()
        for pred_name, pred_dict in predictors.items():
            try:
                self.logger.debug(f'start predictor {pred_name}')
                mode = pred_dict.get('mode')
                if (mode == 'always' and self.processes.get(pred_name)) or \
                        (mode == 'experiment' and not is_experiment) or \
                        (mode == 'no_experiment' and is_experiment):
                    continue

                # continue if there's something in kwargs that also appears in the predictor dict, and they're different
                if any(k in pred_dict and (v not in pred_dict[k] if isinstance(pred_dict[k], list) else v != pred_dict[k])
                       for k, v in kwargs.items()):
                    continue

                self._start_predictor(pred_dict, pred_name)
            except Exception:
                self.logger.exception(f'Error in starting predictor: {pred_name}')
        self.preds_start_time = time.time()
        self.is_starting = False

    def _start_predictor(self, pred_dict, pred_name):
        pred_image_size = pred_dict.get('image_size')
        if pred_image_size is not None:
            self.pred_shm[pred_name] = self.shm_manager.SharedMemory(size=int(np.prod(pred_image_size)))
        prd = ImageHandler(pred_name, *self.proc_args, shm=self.shm,
                           pred_shm=self.pred_shm.get(pred_name),
                           pred_image_size=pred_image_size)
        prd.start()
        self.processes[pred_name] = prd
        self.logger.info(f'Predictor {pred_name} is up')

    def stop_predictors(self):
        """stop all alive predictors for this camera unit"""
        preds_to_close = self.get_alive_predictors()
        self.mp_metadata['predictors_stop'].set()
        for pred in preds_to_close:
            if self.processes.get(pred) is not None:
                self.is_stopping = True
                self.processes[pred].join(timeout=5)
                if self.processes[pred].is_alive():
                    self.logger.debug(f'terminating predictor {pred}')
                    self.processes[pred].terminate()
                del self.processes[pred]
                self.is_stopping = False

    def reload_predictors(self, is_experiment, **kwargs):
        """start the configured experiment predictors. If any predictors are already running - stop them"""
        self.stop_predictors()
        self.start_predictors(is_experiment, **kwargs)

    def is_on(self):
        return not self.stop_signal.is_set()
        # return self.processes and all([k in self.processes and self.processes[k].is_alive() for k in ['cam', 'sink']])

    def is_recording(self):
        d = cache.get_cam_dict(self.cam_name)
        return bool(d.get(config.output_dir_key))

    def get_alive_predictors(self) -> list:
        """return the process names of the live predictors"""
        alive_preds = []
        conf_predictors = list(self.get_conf_predictors().keys())
        for predictor_name in set(conf_predictors):
            prd = self.processes.get(predictor_name)
            if prd is not None and prd.is_alive():
                alive_preds.append(predictor_name)
        return alive_preds

    def get_conf_predictors(self) -> dict:
        """Get dictionary with all the configured predictors of the camera unit"""
        conf_preds = self.cam_config.get('predictors', {}).copy()
        # conf_preds = self.cam_config.get('general_predictors', {}).copy()
        # conf_preds.update(self.cam_config.get('experiment_predictors', {}))
        return conf_preds

    @property
    def proc_args(self):
        return self.cam_name, self.frames_queue, self.cam_config, self.log_queue, self.stop_signal, self.mp_metadata

    @property
    def time_on(self):
        return time.time() - self.start_time


class ArenaManager(SyncManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) #address=(config.arena_manager_address, config.arena_manager_port),
                         #authkey=config.arena_manager_password.encode('utf-8'), **kwargs)
        self.start()
        self.detected_cameras = {}
        self.camera_modules = []
        self.units = {}  # activated camera units
        self.threads = {}
        self.schedules = {}  # experiment schedules
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

        self.orm = ORM()
        self.update_upcoming_schedules()
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

    def update_camera_unit(self, cam_name, data):
        if cam_name not in self.units:
            self.logger.warning(f'Aborting cam-unit update; {cam_name} does not exist')
            return

        cam_class, cam_config = self.units[cam_name].cam_cls, self.units[cam_name].cam_config
        cam_config.update(data)
        self.units[cam_name].stop()
        del self.units[cam_name]
        cu = CameraUnit(cam_name, cam_class, self.global_start_event, self.global_stop_event,
                        cam_config, self.log_queue)
        self.units[cam_name] = cu
        cu.start()

    def arena_shutdown(self, *args) -> None:
        self.logger.warning('shutdown start')
        self.logger.debug(f'open threads: {list(self.threads.keys())}')
        self.arena_shutdown_event.set()
        [cu.stop() for cu in self.units.values()]
        for name, t in self.threads.items():
            if threading.current_thread().name != t.name:
                try:
                    t.join()
                    self.logger.debug(f'thread {name} is down')
                except:
                    self.logger.exception(f'Error joining thread {name}')
        self.units, self.threads = {}, {}
        self.logger.info('Closing logging thread; Arena is down')
        self.stop_logging_event.set()
        self.logging_thread.join()
        self.shutdown()
        print('shutdown finished')

    def start_experiment(self, **kwargs):
        cameras_dict = kwargs.get('cameras', {})
        if not cameras_dict:
            self.logger.error('unable to start experiment with no cameras specified')
            return

        e = Experiment(cam_units=self.units, **kwargs)
        self.start_experiment_listeners(e.experiment_stop_flag)
        time.sleep(0.1)
        e.start()

    def start_cached_experiment(self, experiment_name):
        data = ExperimentCache().load(experiment_name)
        data['animal_id'] = cache.get(cc.CURRENT_ANIMAL_ID)
        self.start_experiment(**data)

    def start_management_listeners(self):
        subs_dict = {
            'arena_shutdown': self.arena_shutdown
        }
        management_subs = start_management_subscribers(self.arena_shutdown_event, self.log_queue, subs_dict)
        self.threads.update(management_subs)
        # scheduler
        if not config.DISABLE_SCHEDULER:
            self.threads['scheduler'] = Scheduler(self)
            self.threads['scheduler'].start()

    def start_experiment_listeners(self, exp_stop_flag):
        # start experiment listeners
        exp_subs = start_experiment_subscribers(exp_stop_flag, self.log_queue)
        self.threads.update(exp_subs)

    def update_upcoming_schedules(self):
        if config.DISABLE_DB:
            return
        self.schedules = {}
        for s in self.orm.get_upcoming_schedules().all():
            self.schedules[s.id] = f'{s.date.strftime(config.schedule_date_format)} - {s.experiment_name}'

    def reset_cache(self):
        for name, col in cc.__dict__.items():
            if name.startswith('_') or col.timeout == 'static':
                continue
            cache.delete(col)

    def record(self, cameras=None, folder_prefix=None, output_dir=None, num_frames=None, rec_time=None):
        """
        Record videos from Arena's cameras
        :param cameras: (dict) Cameras to be used. You can specify last digits of p/n or name. (for more than 1 use ',')
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
        if not self.camera_modules or 1:
            return
        info_df = self.camera_modules[0].scan_cameras(is_print=False)
        with pd.option_context('display.max_colwidth', None,
                               'display.max_columns', None,
                               'display.max_rows', None):
            self.logger.info(f'\n{info_df}')
        if return_string:
            return f'\nCameras Info:\n\n{info_df.to_string()}\n\n'
        return info_df

    def get_frame(self, cam_name):
        return self.units[cam_name].get_frame()

    def get_streaming_camera(self):
        return self._streaming_camera

    def set_streaming_camera(self, cam_name):
        self._streaming_camera = cam_name

    def stop_stream(self):
        if self._streaming_camera is not None:
            try:
                self.units[self._streaming_camera].mp_metadata['is_streaming'].clear()
            except Exception:
                self.logger.exception('Error in stop_stream')
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
        output = f"{output_dir or config.recordings_output_dir}/{folder_name}"
        return mkdir(output)

    def is_cam_trigger_setup(self):
        return any(cu.cam_config.get('trigger_source') for cu in self.units.values())

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
