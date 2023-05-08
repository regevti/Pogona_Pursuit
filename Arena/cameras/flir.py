import cv2
import time
import re
import os
import PySpin
import json
import pandas as pd
import numpy as np
from multiprocessing.dummy import Pool
from cache import CacheColumns
from utils import calculate_fps, mkdir, datetime_string
from arena import Camera
import config
from arrayqueues.shared_arrays import Full
from cache import RedisCache, CacheColumns as cc


cache = RedisCache()


class FLIRCamera(Camera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _run(self):
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        cam = self.get_cam(cam_list)
        if cam is None:
            self.logger.error(f'unable to find camera {self.cam_name}')
            return
        cam.Init()
        self.configure_camera(cam)
        self.update_time_delta(cam)
        cam.BeginAcquisition()
        image_result = None
        time.sleep(1)

        try:
            while not self.stop_signal.is_set():
                if image_result is not None:
                    try:
                        image_result.Release()
                    except PySpin.SpinnakerException:
                        pass
                try:
                    image_result = cam.GetNextImage(config.QUEUE_WAIT_TIME * 1000, 0)
                except PySpin.SpinnakerException as e:
                    if e.errorcode == -1011:
                        # restart acquisition due to empty buffer bug
                        # "Spinnaker: Failed waiting for EventData on NEW_BUFFER_DATA event. [-1011]"
                        cam.EndAcquisition()
                        cam.BeginAcquisition()
                        time.sleep(0.1)
                        continue
                else:
                    if image_result.IsIncomplete():  # Ensure image completion
                        sts = image_result.GetImageStatus()
                        self.logger.warning(f'Image incomplete with image status {sts}')
                        if sts == 9:
                            self.logger.warning(f'Breaking after status 9')
                            break
                    else:
                        self.image_handler(image_result)
        except Exception as exc:
            self.logger.error(str(exc))
        finally:
            if cam.IsStreaming():
                cam.EndAcquisition()
            cam.DeInit()
            cam_list.Clear()

    def image_handler(self, image_result: PySpin.ImagePtr):
        t0 = time.time()
        waiting_time = 0.1
        if self.stop_signal.is_set():
            return
        img = image_result.GetNDArray()
        timestamp = image_result.GetTimeStamp() / 1e9 + self.camera_time_delta
        while True:
            try:
                self.frames_queue.put(img, timestamp)
                self.calc_fps(timestamp)
                break
            except Full:
                if (time.time() - t0) > waiting_time:
                    if not self.last_queue_warning_time or (time.time() - self.last_queue_warning_time > 60):
                        self.logger.warning(f'Queue is still full after waiting {waiting_time}')
                        self.last_queue_warning_time = time.time()
                    break

    def configure_camera(self, cam):
        """Configure camera for trigger mode before acquisition"""
        try:
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
            cam.ExposureTime.SetValue(int(self.cam_config['exposure']))
            cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)

            if self.cam_config.get('is_color'):
                self.logger.info(f'setting camera {self.cam_name} to BGR8; '
                                 f'due to is_color={self.cam_config.get("is_color")}')
                cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)

            assert not (self.cam_config.get('trigger_source') and self.cam_config.get('fps')), \
                'must provide either fps or trigger_source'
            if self.cam_config.get('trigger_source'):
                cam.AcquisitionFrameRateEnable.SetValue(False)
                cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
                cam.TriggerSource.SetValue(getattr(PySpin, f"TriggerSource_{self.cam_config['trigger_source']}"))
                cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
                cam.TriggerActivation.SetValue(PySpin.TriggerActivation_FallingEdge)

            elif self.cam_config.get('fps'):
                cam.DeviceLinkThroughputLimit.SetValue(self.get_max_throughput(cam))
                cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
                cam.AcquisitionFrameRateEnable.SetValue(True)
                cam.AcquisitionFrameRate.SetValue(int(self.cam_config['fps']))

            else:
                raise Exception('bad configuration. must provide either trigger_source or fps in cam_config')

            # cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
            # max_throughput = self.get_max_throughput(cam)
            # self.logger.info(f'Max throughput: {max_throughput}')
            # cam.DeviceLinkThroughputLimit.SetValue(max_throughput)

            self.logger.debug(f'Finished Configuration')

        except PySpin.SpinnakerException as exc:
            self.logger.error(f'(configure_images); {exc}')

    def get_cam(self, cam_list):
        cam_id = str(self.cam_config['id'])
        for cam in cam_list:
            if get_device_id(cam) == cam_id:
                return cam

    def get_max_throughput(self, cam):
        try:
            max_throughput = int(cam.DeviceMaxThroughput.GetValue())
        except Exception as exc:
            self.logger.warning(exc)
            max_throughput = 4e8

        return max_throughput

    def update_time_delta(self, cam):
        cam.TimestampLatch.Execute()
        cam_time = cam.TimestampLatchValue.GetValue()  # in nanosecs
        server_time = time.time_ns()
        self.camera_time_delta = (server_time - cam_time) / 1e9


info_fields = [
    'AcquisitionFrameRate',
    'AcquisitionMode',
    'TriggerSource',
    'TriggerMode',
    'TriggerSelector',
    'PayloadSize',
    'EventSelector',
    'LineStatus',
    'ExposureTime',
    'PixelFormat',
    'DeviceLinkCurrentThroughput',
    'DeviceLinkThroughputLimit',
    'DeviceMaxThroughput',
    'DeviceLinkSpeed',
]


def scan_cameras(is_print=True) -> pd.DataFrame:
    df, cam_names = [], []
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    for cam in cam_list:
        try:
            sc = SpinCamera(cam)
            # if not sc.is_firefly():
            #     continue
            d = {'DeviceID': sc.device_id}
            d.update({k: v for k, v in zip(info_fields, sc.info())})
            df.append(d)
            # find camera name from cam_config.yaml
            cam_name = 'unknown'
            for n, cam_config in config.cameras.items():
                if str(cam_config['id']) == sc.device_id:
                    cam_name = n
                    break
            cam_names.append(cam_name)
        except Exception as exc:
            print(f'Unable to load camera; {exc}')

    df = pd.DataFrame(df, index=cam_names)
    del cam, sc
    if is_print:
        output = f'\nCameras Info:\n\n{df.to_string()}\n'
        print(output)
    cam_list.Clear()
    system.ReleaseInstance()
    return df


def get_device_id(cam) -> str:
    """Get the camera device ID of the cam instance"""
    nodemap_tldevice = cam.GetTLDeviceNodeMap()
    device_id = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber')).GetValue()
    m = re.search(r'\d{8}', device_id)
    if not m:
        device_id = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceID')).GetValue()
        m = re.search(r'\d{8}', device_id)
        if m:
            return m[0]
        else:
            return device_id
    return m[0]



################################################ End Predictor ################################################


class SpinCamera:
    def __init__(self, cam, acquire_stop=None, dir_path=None, cache=None, log_stream=None,
                 is_use_predictions=False):
        self.cam = cam
        self.acquire_stop = acquire_stop or {'num_frames': 0}
        self.dir_path = dir_path
        self.cache = cache
        self.is_use_predictions = is_use_predictions
        self.thread_event = None
        # self.validate_acquire_stop()

        self.is_ready = False  # ready for acquisition
        self.video_out = None
        self.start_acquire_time = None
        # self.mqtt_client = MQTTPublisher()
        self.cam.Init()
        # self.logger = get_logger(self.device_id, dir_path, log_stream=log_stream)
        self.name = self.get_camera_name()
        if self.is_realtime_mode:
            self.logger.info('Working in realtime mode')
            self.predictor_experiment_ids = []
            # self.predictor = predictor.gen_hit_predictor(self.logger, dir_path)

    def begin_acquisition(self, exposure):
        """Main function for running camera acquisition in trigger mode"""
        try:
            self.configure_camera(exposure)
            self.cam.BeginAcquisition()
            self.is_ready = True
            self.logger.debug('Entering to trigger mode')
        except Exception as exc:
            self.logger.error(f'(run); {exc}')

    def __del__(self):
        if self.is_realtime_mode:
            self.predictor.reset()
        if self.cam.IsStreaming():
            self.cam.EndAcquisition()
        self.cam.DeInit()

    def configure_camera(self, exposure):
        """Configure camera for trigger mode before acquisition"""
        try:
            # self.cam.AcquisitionFrameRate.SetValue(FPS)
            self.cam.AcquisitionFrameRateEnable.SetValue(False)
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

    def acquire(self):
        """Acquire images and measure FPS"""
        if self.is_ready:
            frame_times = list()
            image_handler_times = list()
            i = 0
            err_count = 0
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
                    self.management_log(f'Acquire Error: {exc}')
                    break

                finally:
                    i += 1

            self.logger.info(f'Number of frames taken: {i}')
            mean_fps, std_fps = self.analyze_timestamps(frame_times)
            self.logger.debug(f'Calculated FPS: {mean_fps:.3f} ± {std_fps:.3f}')
            self.logger.debug(f'Average image handler time: {np.mean(image_handler_times):.4f} seconds')
            self.management_log(f'Number of frames taken: {i}, Calculated FPS: {mean_fps:.3f} ± {std_fps:.3f}')
            if self.is_realtime_mode:
                self.predictor.save_predictions()

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

        if not self.is_realtime_mode:# or config.is_predictor_experiment:
            if self.dir_path and self.video_out is None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                h, w = img.shape[:2]
                self.video_out = cv2.VideoWriter(self.video_path, fourcc, config.fps, (w, h), True)

            self.video_out.write(img)

        # img.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

    # def validate_acquire_stop(self):
    #     for key, value in self.acquire_stop.items():
    #         assert key in config.acquire_stop_options, f'unknown acquire_stop: {key}'
    #         if config.acquire_stop_options[key] == 'cache':
    #             assert self.cache is not None
    #         elif config.acquire_stop_options[key] == 'event':
    #             assert value is not None
    #             self.thread_event = value
    #         else:
    #             assert isinstance(value, int), f'acquire stop {key}: expected type int, received {type(value)}'

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
        return self.cache.get(CacheColumns.EXPERIMENT_BLOCK_ID)

    def check_thread_event(self, iteration):
        return self.thread_event.is_set()

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
        for k, v in zip(info_fields, self.info()):
            st += f'{k}: {v}\n'
        self.logger.debug(st)

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

    def info(self) -> list:
        """Get All camera values of INFO_FIELDS and return as a list"""
        nan_string = 'x'
        values = []
        for field in info_fields:
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
                value = nan_string
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
        return
        # for name, device_id in config.camera_names.items():
        #     if self.device_id == device_id:
        #         return name

    def management_log(self, msg):
        self.mqtt_client.publish_event(config.ui_console_channel, f'>> Camera {self.name}: {msg}')

    @property
    def video_path(self):
        return f'{self.dir_path}/{self.name}_{datetime_string()}.avi'

    @property
    def timestamp_path(self):
        mkdir(f'{self.dir_path}/timestamps')
        return f'{self.dir_path}/timestamps/{self.device_id}.csv'

    @property
    def device_id(self):
        return get_device_id(self.cam)

    @property
    def is_realtime_mode(self):
        return self.is_use_predictions and self.name == config.realtime_camera





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

    df = pd.DataFrame(df, columns=info_fields, index=index)
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


def start_streaming(sc: SpinCamera):
    """Thread function for start acquiring frames from camera"""
    sc.acquire()
    del sc


def capture_image(camera: str, exposure=0) -> (np.ndarray, None):
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


def record(exposure=0, cameras=None, output=None, folder_prefix=None,
           cache=None, is_use_predictions=False, **acquire_stop) -> str:
    """
    Record videos from Arena's cameras
    :param exposure: The exposure time to be set to the cameras
    :param cameras: (str) Cameras to be used. You can specify last digits of p/n or name. (for more than 1 use ',')
    :param output: Output dir for videos and timestamps, if not exist save into a timestamp folder in default output dir.
    :param folder_prefix: Prefix to be added to folder name. Not used if output is given.
    :param cache: memory cache to be used by the cameras
    :param is_use_predictions: relevant for realtime camera only - using strike prediction
    """
    if config.is_debug_mode:
        return 'DEBUG MODE'
    assert all(k in config.acquire_stop_options for k in acquire_stop.keys())
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    log_stream = None #get_log_stream()

    if cameras:
        filter_cameras(cam_list, cameras)

    if not output:
        folder_name = datetime_string()
        if folder_prefix:
            folder_name = f'{folder_prefix}_{folder_name}'
        output = f"{config.recordings_output_dir}/{folder_name}"
    output = mkdir(output)

    filtered = [(cam, acquire_stop, output, exposure, cache, log_stream, is_use_predictions) for cam in cam_list]
    print(f'\nCameras detected: {len(filtered)}')
    print(f'Acquire Stop: {acquire_stop}')
    if filtered:
        with Pool(len(filtered)) as pool:
            results = pool.starmap(start_camera, filtered)
            pool.starmap(start_streaming, [(sc,) for sc in results])
        del filtered, results  # must delete this list in order to destroy all pointers to cameras.

    cam_list.Clear()
    # system.ReleaseInstance()

    return log_stream.getvalue()
