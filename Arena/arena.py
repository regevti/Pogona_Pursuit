#!/usr/bin/env python

import time
import re
import cv2
import json
import argparse

from datetime import datetime
import pandas as pd
import numpy as np
from multiprocessing.dummy import Pool
import PySpin
import config
from cache import CacheColumns
from mqtt import MQTTPublisher
from utils import get_logger, calculate_fps, mkdir, get_log_stream, datetime_string

################################################ Predictor ################################################

IS_PREDICTOR_READY = False
if not config.is_disable_predictor:
    try:
        from Prediction import predictor, detector, seq2seq_predict

        _detector = detector.Detector_v4(conf_thres=config.detector_thresh)

        class PredictModel:
            def __init__(self, weigths, traj_model):
                self.weights = weigths
                self.traj_model = traj_model
                self.input_len = 20
                self.forecast_horizon = 20
                self.seq2seq_predictor = seq2seq_predict.Seq2SeqPredictor(model=self.traj_model,
                                                                          weights_path=self.weights,
                                                                          input_len=self.input_len,
                                                                          forecast_horizon=self.forecast_horizon)
                self.hit_pred = predictor.HitPredictor(trajectory_predictor=self.seq2seq_predictor, detector=_detector)

        _models = {
            'gru': PredictModel('Prediction/traj_models/model_20_20_h64_b64_l1_EncDec_6_best.pth', seq2seq_predict.GRUEncDec()),
            'lstm': PredictModel('Prediction/traj_models/model_20_20_h64_b128_l1_lstmDense_feeding_51_best.pth',
                                 seq2seq_predict.LSTMdense(output_seq_size=20, hidden_size=64, LSTM_layers=1,
                                                           embedding_size=16))
        }
        IS_PREDICTOR_READY = True
    except Exception as exc:
        print(f'Error loading detector: {exc}')

################################################ End Predictor ################################################


class SpinCamera:
    def __init__(self, cam: PySpin.Camera, acquire_stop=None, dir_path=None, cache=None, log_stream=None,
                 is_use_predictions=False):
        self.cam = cam
        self.acquire_stop = acquire_stop or {'num_frames': config.default_num_frames}
        self.dir_path = dir_path
        self.cache = cache
        self.is_use_predictions = is_use_predictions
        self.validate_acquire_stop()

        self.is_ready = False  # ready for acquisition
        self.video_out = None
        self.start_acquire_time = None
        self.mqtt_client = None

        self.cam.Init()
        self.logger = get_logger(self.device_id, dir_path, log_stream=log_stream)
        self.name = self.get_camera_name()
        if self.is_realtime_mode:
            self.logger.info('Working in realtime mode')
            self.predictor_experiment_ids = []
            self.predictor = _models[config.predictor_model].hit_pred
            self.mqtt_client = MQTTPublisher()

    def begin_acquisition(self, exposure):
        """Main function for running camera acquisition in trigger mode"""
        try:
            self.configure_camera(exposure)
            self.cam.BeginAcquisition()
            self.is_ready = True
            self.logger.info('Entering to trigger mode')
        except Exception as exc:
            self.logger.error(f'(run); {exc}')

    def __del__(self):
        if self.is_realtime_mode:
            self.predictor.reset()
        self.cam.DeInit()

    def configure_camera(self, exposure):
        """Configure camera for trigger mode before acquisition"""
        try:
            self.cam.AcquisitionFrameRateEnable.SetValue(False)
            # self.cam.AcquisitionFrameRate.SetValue(FPS)
            self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line1)
            self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
            self.cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
            self.cam.LineSelector.SetValue(PySpin.LineSelector_Line1)
            self.cam.LineMode.SetValue(PySpin.LineMode_Input)
            self.cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
            self.cam.DeviceLinkThroughputLimit.SetValue(self.get_max_throughput())
            self.cam.ExposureTime.SetValue(exposure)
            self.logger.info(f'Finished Configuration')
            self.log_info()

        except PySpin.SpinnakerException as exc:
            self.logger.error(f'(configure_images); {exc}')

    def capture_image(self, exposure):
        """Capture single image"""
        self.begin_acquisition(exposure)
        try:
            image_result = self.cam.GetNextImage()
            img = image_result.GetNDArray()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_result.Release()
            return img
        except PySpin.SpinnakerException as exc:
            self.logger.error(f'(image_capture); {exc}')
        finally:
            self.cam.EndAcquisition()

    def acquire(self):
        """Acquire images and measure FPS"""
        if self.is_ready:
            frame_times = list()
            image_handler_times = list()
            i = 0
            while self.is_acquire_allowed(i):
                try:
                    image_result = self.cam.GetNextImage(2000)  # Retrieve next received image
                    if i == 0:
                        self.start_acquire_time = time.time()
                        self.logger.info('Acquisition Started')

                    if image_result.IsIncomplete():  # Ensure image completion
                        sts = image_result.GetImageStatus()
                        self.logger.warning(f'Image incomplete with image status {sts}')
                        if sts == 9:
                            self.logger.warning(f'Breaking after status 9')
                            image_result.Release()
                            break
                    else:
                        frame_times.append(image_result.GetTimeStamp())
                        t0 = time.time()
                        self.image_handler(image_result, i)
                        image_handler_times.append(time.time() - t0)

                    image_result.Release()  # Release image

                except PySpin.SpinnakerException as exc:
                    self.logger.error(f'(acquire); {exc}')
                    continue

                finally:
                    i += 1

            self.logger.info(f'Number of frames taken: {i}')
            mean_fps, std_fps = self.analyze_timestamps(frame_times)
            self.logger.info(f'Calculated FPS: {mean_fps:.3f} ± {std_fps:.3f}')
            self.logger.info(f'Average image handler time: {np.mean(image_handler_times):.4f} seconds')
            self.save_predictions()

        self.cam.EndAcquisition()  # End acquisition
        if self.video_out:
            self.logger.info(f'Video path: {self.video_path}')
            self.video_out.release()
        self.is_ready = False

    def image_handler(self, image_result: PySpin.ImagePtr, i: int):
        img = image_result.GetNDArray()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_realtime_mode:
            self.handle_prediction(img, i)

        if not self.is_realtime_mode or config.is_predictor_experiment:
            if self.dir_path and self.video_out is None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                h, w = img.shape[:2]
                self.video_out = cv2.VideoWriter(self.video_path, fourcc, config.fps, (w, h), True)

            self.video_out.write(img)

        # img.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

    def validate_acquire_stop(self):
        for key, value in self.acquire_stop.items():
            assert key in config.acquire_stop_options, f'unknown acquire_stop: {key}'
            if config.acquire_stop_options[key] == 'cache':
                assert self.cache is not None
            else:
                assert isinstance(value, int), f'acquire stop {key}: expected type int, received {type(value)}'

    def is_acquire_allowed(self, iteration):
        """Check all given acquire_stop conditions"""
        for stop_key in self.acquire_stop.keys():
            if not getattr(self, f'check_{stop_key}')(iteration):
                return False
        return True

    def check_num_frames(self, iteration):
        return iteration <= self.acquire_stop['num_frames']

    def check_record_time(self, iteration):
        if iteration == 0:
            return True
        return time.time() < self.start_acquire_time + self.acquire_stop['record_time']

    def check_manual_stop(self, iteration):
        return not self.cache.get(CacheColumns.MANUAL_RECORD_STOP)

    def check_trial_alive(self, iteration):
        return self.cache.get(CacheColumns.EXPERIMENT_TRIAL_ON)

    def handle_prediction(self, img, i):
        if config.is_predictor_experiment and not i % 60:
            self.predictor_experiment_ids.append(i)
            self.mqtt_client.publish_command('show_pogona', 3)
        forecast, hit_point, hit_steps = self.predictor.handle_frame(img)
        if hit_point is None or not hit_steps:
            return

        time2hit = (1 / config.fps) * hit_steps  # seconds
        self.mqtt_client.publish_event('event/log/prediction', json.dumps({'hit_point': hit_point.tolist(), 'time2hit': time2hit}))

    def log_info(self):
        """Print into logger the info of the camera"""
        st = '\n'
        for k, v in zip(config.info_fields, self.info()):
            st += f'{k}: {v}\n'
        self.logger.info(st)

    def analyze_timestamps(self, frame_times):
        """Convert camera's timestamp to server time, save server timestamps and calculate FPS"""
        self.cam.TimestampLatch()
        camera_time = self.cam.TimestampLatchValue.GetValue()
        server_time = time.time()
        frame_times = server_time - (camera_time - pd.Series(frame_times)) / 1e9
        mean_fps, std_fps = calculate_fps(frame_times)

        frame_times = pd.to_datetime(frame_times, unit='s')
        frame_times.to_csv(self.timestamp_path)
        if config.is_predictor_experiment and self.is_realtime_mode:
            predictor_times = frame_times[self.predictor_experiment_ids]
            predictor_times.to_csv(f'{self.dir_path}/predictor_times.csv')

        return mean_fps, std_fps

    def save_predictions(self):
        if not self.is_realtime_mode:
            return

        pd.Series(self.predictor.forecasts).to_csv(self.predictions_path)
        pd.DataFrame(self.predictor.history).to_csv(self.predictor_history_path)

    def info(self) -> list:
        """Get All camera values of INFO_FIELDS and return as a list"""
        nan = 'x'
        values = []
        for field in config.info_fields:
            try:
                value = getattr(self.cam, field.replace(' ', ''))
                if not value:
                    raise Exception('No Value')
                else:
                    try:
                        value = value.ToString()
                    except PySpin.SpinnakerException:
                        value = value.GetValue()
            except Exception as exc:
                self.logger.warning(f'{field}: {exc}')
                value = nan
            values.append(value)

        return values

    def get_max_throughput(self):
        try:
            max_throughput = int(self.cam.DeviceMaxThroughput.GetValue())
        except Exception as exc:
            self.logger.warning(exc)
            max_throughput = config.default_max_throughput

        return max_throughput

    def is_firefly(self):
        """Check whether cam is a Firefly camere"""
        nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
        device_name = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceModelName')).GetValue()
        if 'firefly' in device_name.lower():
            return True

    def get_camera_name(self):
        for name, device_id in config.camera_names.items():
            if self.device_id == device_id:
                return name

    @property
    def video_path(self):
        return f'{self.dir_path}/{self.name}_{datetime_string()}.avi'

    @property
    def timestamp_path(self):
        mkdir(f'{self.dir_path}/timestamps')
        return f'{self.dir_path}/timestamps/{self.device_id}.csv'

    @property
    def predictions_path(self):
        return f'{self.dir_path}/forecasts.csv'

    @property
    def predictor_history_path(self):
        return f'{self.dir_path}/predictor_history.csv'

    @property
    def device_id(self):
        return get_device_id(self.cam)

    @property
    def is_realtime_mode(self):
        return IS_PREDICTOR_READY and self.is_use_predictions and self.name == config.realtime_camera


############################################################################################################


def get_device_id(cam) -> str:
    """Get the camera device ID of the cam instance"""
    nodemap_tldevice = cam.GetTLDeviceNodeMap()
    device_id = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceID')).GetValue()
    m = re.search(r'\d{8}', device_id)
    if not m:
        return device_id
    return m[0]


def filter_cameras(cam_list: PySpin.CameraList, cameras_string: str) -> None:
    """Filter cameras according to camera_label, which can be a name or last digits of device ID"""
    current_devices = [get_device_id(cam) for cam in cam_list]
    chosen_devices = []
    for cam_id in cameras_string.split(','):
        if re.match(r'[a-zA-z]+', cam_id):
            device = config.camera_names.get(cam_id)
            if device and device in current_devices:
                chosen_devices.append(device)
        elif re.match(r'[0-9]+', cam_id):
            chosen_devices.extend([d for d in current_devices if d[-len(cam_id):] == cam_id])

    def _remove_from_cam_list(device_id):
        devices = [get_device_id(c) for c in cam_list]
        cam_list.RemoveByIndex(devices.index(device_id))

    for d in current_devices:
        if d not in chosen_devices:
            _remove_from_cam_list(d)


def display_info():
    """Function for displaying info of all FireFly cameras detected"""
    df = []
    index = []
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    for cam in cam_list:
        sc = SpinCamera(cam)
        if not sc.is_firefly():
            continue
        df.append(sc.info())
        index.append(sc.device_id)

    df = pd.DataFrame(df, columns=config.info_fields, index=index)
    del cam, sc
    output = f'\nCameras Info:\n\n{df.to_string()}\n'
    cam_list.Clear()
    # system.ReleaseInstance()
    return output


def start_camera(cam, acquire_stop, dir_path, exposure, cache, log_stream, is_use_predictions):
    """Thread function for configuring and starting spin cameras"""
    sc = SpinCamera(cam, acquire_stop, dir_path, cache=cache, log_stream=log_stream, is_use_predictions=is_use_predictions)
    sc.begin_acquisition(exposure)
    return sc


def wait_for_streaming(results: list, is_auto_start=False):
    """Wait for user approval for start streaming and send serial for Arduino to start TTL.
    If keyboard interrupt turn all is_ready to false, so acquisition will not start"""
    serializer = None
    try:
        if not is_auto_start:
            key = input(f'\nThere are {len([sc for sc in results if sc.is_ready])} cameras ready for streaming.\n'
                        f'Press any key for sending TTL serial to start streaming.\n'
                        f"If you like to start TTL manually press 'm'\n>> ")
            # if not key == 'm':
            #     serializer = Serializer()
            #     serializer.start_acquisition()

    except Exception as exc:
        print(f'Error: {exc}')
        for sc in results:
            sc.is_ready = False

    return results, serializer


def start_streaming(sc: SpinCamera):
    """Thread function for start acquiring frames from camera"""
    sc.acquire()
    del sc


def capture_image(camera: str, exposure=config.exposure_time) -> (np.ndarray, None):
    """
    Capture single image from a camera
    :param camera: The camera name (don't use more than one camera)
    :param exposure: The exposure of the camera
    :return: Image numpy array
    """
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    filter_cameras(cam_list, camera)
    if len(cam_list) < 1:
        print(f'No camera matches name: {camera}')
        return
    cam = SpinCamera(cam_list[0])
    img = cam.capture_image(exposure)
    del cam
    cam_list.Clear()
    return img


def record(exposure=config.exposure_time, cameras=None, output=None, folder_prefix=None,
           is_auto_start=False, cache=None, is_use_predictions=False, **acquire_stop) -> str:
    """
    Record videos from Arena's cameras
    :param exposure: The exposure time to be set to the cameras
    :param cameras: (str) Cameras to be used. You can specify last digits of p/n or name. (for more than 1 use ',')
    :param output: Output dir for videos and timestamps, if not exist save into a timestamp folder in default output dir.
    :param folder_prefix: Prefix to be added to folder name. Not used if output is given.
    :param is_auto_start: Start record automatically or wait for user input
    :param cache: memory cache to be used by the cameras
    :param is_use_predictions: relevant for realtime camera only - using strike prediction
    """
    if config.is_debug_mode:
        return 'DEBUG MODE'
    assert all(k in config.acquire_stop_options for k in acquire_stop.keys())
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    log_stream = get_log_stream()

    if cameras:
        filter_cameras(cam_list, cameras)

    if not output:
        folder_name = datetime_string()
        if folder_prefix:
            folder_name = f'{folder_prefix}_{folder_name}'
        output = f"{config.output_dir}/{folder_name}"
    output = mkdir(output)

    filtered = [(cam, acquire_stop, output, exposure, cache, log_stream, is_use_predictions) for cam in cam_list]
    print(f'\nCameras detected: {len(filtered)}')
    print(f'Acquire Stop: {acquire_stop}')
    if filtered:
        with Pool(len(filtered)) as pool:
            results = pool.starmap(start_camera, filtered)
            results, serializer = wait_for_streaming(results, is_auto_start)
            results = [(sc,) for sc in results]
            pool.starmap(start_streaming, results)
        del filtered, results  # must delete this list in order to destroy all pointers to cameras.

    cam_list.Clear()
    # system.ReleaseInstance()

    return log_stream.getvalue()


def main():
    """Main function for Arena capture"""
    ap = argparse.ArgumentParser(description="Tool for capturing multiple cameras streams in the arena.")
    ap.add_argument("-n", "--num_frames", type=int, help=f"Specify Number of Frames.")
    ap.add_argument("-t", "--record_time", type=int, help=f"Specify record duration in seconds.")
    ap.add_argument("-m", "--manual_stop", action="store_true", default=False,
                    help=f"Stop record using cache key MANUAL_RECORD_STOP.")
    ap.add_argument("--experiment_alive", action="store_true", default=False,
                    help=f"Stop record if the experiment ended")
    ap.add_argument("-o", "--output", type=str, default=config.output_dir,
                    help=f"Specify output directory path. Default={config.output_dir}")
    ap.add_argument("-e", "--exposure", type=int, default=config.exposure_time,
                    help=f"Specify cameras exposure time. Default={config.exposure_time}")
    ap.add_argument("-c", "--camera", type=str, required=False,
                    help=f"filter cameras by last digits or according to CAMERA_NAMES (for more than one use ',').")
    ap.add_argument("-i", "--info", action="store_true", default=False,
                    help=f"Show cameras information")

    args = vars(ap.parse_args())

    if args.get('info'):
        print(display_info())
    else:
        acquire_stop = {}
        for key in config.acquire_stop_options:
            if key in args:
                acquire_stop[key] = args[key]
        record(args.get('exposure'), args.get('camera'), args.get('output'), **acquire_stop)


if __name__ == '__main__':
    main()
