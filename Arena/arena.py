#!/usr/bin/env python

import time
import re
import cv2
import argparse

from datetime import datetime
import pandas as pd
from multiprocessing.dummy import Pool
import PySpin
from cache import CacheColumns
from utils import get_logger, calculate_fps, mkdir, get_log_stream

DEFAULT_NUM_FRAMES = 1000
DEFAULT_MAX_THROUGHPUT = 94578303
EXPOSURE_TIME = 8000
OUTPUT_DIR = 'output'
UNSORTED_DIR = mkdir('output/unsorted')
FPS = 60
SAVED_FRAME_RESOLUTION = (1440, 1088)
INFO_FIELDS = ['AcquisitionFrameRate', 'AcquisitionMode', 'TriggerSource', 'TriggerMode', 'TriggerSelector',
               'PayloadSize', 'EventSelector', 'LineStatus', 'ExposureTime',
               'DeviceLinkCurrentThroughput', 'DeviceLinkThroughputLimit', 'DeviceMaxThroughput', 'DeviceLinkSpeed']
CAMERA_NAMES = {
    'realtime': '19506468',
    'right': '19506475',
    'left': '19506455',
    'back': '19506481'
}
ACQUIRE_STOP_OPTIONS = {
    'num_frames': int,
    'record_time': int,
    'manual_stop': 'cache',
    'experiment_alive': 'cache'
}


class SpinCamera:
    def __init__(self, cam: PySpin.Camera, acquire_stop=None, dir_path=None, cache=None, log_stream=None):
        self.cam = cam
        self.acquire_stop = acquire_stop or {'num_frames': DEFAULT_NUM_FRAMES}
        self.dir_path = dir_path
        self.cache = cache
        self.validate_acquire_stop()

        self.is_ready = False  # ready for acquisition
        self.video_out = None
        self.start_acquire_time = None

        self.cam.Init()
        self.logger = get_logger(self.device_id, dir_path, log_stream=log_stream)

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
        self.cam.DeInit()

    def get_max_throughput(self):
        try:
            max_throughput = int(self.cam.DeviceMaxThroughput.GetValue())
        except Exception as exc:
            self.logger.warning(exc)
            max_throughput = DEFAULT_MAX_THROUGHPUT

        self.logger.info(f'max throughput: {max_throughput}')
        return max_throughput

    def configure_camera(self, exposure):
        """Configure camera for trigger mode before acquisition"""
        try:
            self.cam.AcquisitionFrameRateEnable.SetValue(False)
            # self.cam.AcquisitionFrameRate.SetValue(60)
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
            i = 0
            while self.is_acquire_allowed(i):
                try:
                    image_result = self.cam.GetNextImage()  # Retrieve next received image
                    if i == 0:
                        self.start_acquire_time = time.time()
                        self.logger.info('Acquisition Started')

                    if image_result.IsIncomplete():  # Ensure image completion
                        sts = image_result.GetImageStatus()
                        self.logger.warning(f'Image incomplete with image status {sts}')
                        if sts == 9:
                            break
                    else:
                        frame_times.append(time.time())
                        self.image_handler(image_result)

                    image_result.Release()  # Release image

                except PySpin.SpinnakerException as exc:
                    self.logger.error(f'(acquire); {exc}')
                    continue

                finally:
                    i += 1

            self.logger.info(f'Number of frames taken: {i}')
            mean_fps, std_fps = calculate_fps(frame_times)
            self.logger.info(f'Calculated FPS: {mean_fps} ± {std_fps}')
            self.save_frames_timestamps(frame_times)

        self.cam.EndAcquisition()  # End acquisition
        if self.video_out:
            self.logger.info(f'Video path: {self.video_path}')
            self.video_out.release()
        self.is_ready = False

    def image_handler(self, image_result: PySpin.ImagePtr):
        img = image_result.GetNDArray()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.dir_path and self.video_out is None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            h, w = img.shape[:2]
            self.video_out = cv2.VideoWriter(self.video_path, fourcc, FPS, (w, h), True)

        self.video_out.write(img)
        # img.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

    def validate_acquire_stop(self):
        for key, value in self.acquire_stop.items():
            assert key in ACQUIRE_STOP_OPTIONS, f'unknown acquire_stop: {key}'
            if ACQUIRE_STOP_OPTIONS[key] == 'cache':
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

    def check_experiment_alive(self, iteration):
        return self.cache.get(CacheColumns.EXPERIMENT_NAME)

    def log_info(self):
        """Print into logger the info of the camera"""
        st = ''
        for k, v in zip(INFO_FIELDS, self.info()):
            st += f'{k}: {v}\n'
        self.logger.info(st)

    def save_frames_timestamps(self, frame_times):
        """Save frames timestamps to csv file"""
        pd.to_datetime(pd.Series(frame_times), unit='s').to_csv(self.timestamp_path)

    def info(self) -> list:
        """Get All camera values of INFO_FIELDS and return as a list"""
        nan = 'x'
        values = []
        for field in INFO_FIELDS:
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

    def is_firefly(self):
        """Check whether cam is a Firefly camere"""
        nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
        device_name = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceModelName')).GetValue()
        if 'firefly' in device_name.lower():
            return True

    @property
    def video_path(self):
        return f'{self.dir_path}/{self.device_id}.avi'

    @property
    def timestamp_path(self):
        mkdir(f'{self.dir_path}/timestamps')
        return f'{self.dir_path}/timestamps/{self.device_id}.csv'

    @property
    def device_id(self):
        return get_device_id(self.cam)

############################################################################################################


def get_device_id(cam) -> str:
    """Get the camera device ID of the cam instance"""
    nodemap_tldevice = cam.GetTLDeviceNodeMap()
    return PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceID')).GetValue()


def filter_cameras(cam_list: PySpin.CameraList, cameras_string: str) -> None:
    """Filter cameras according to camera_label, which can be a name or last digits of device ID"""
    current_devices = [get_device_id(c) for c in cam_list]
    chosen_devices = []
    for cam_id in cameras_string.split(','):
        if re.match(r'[a-zA-z]+', cam_id):
            device = CAMERA_NAMES.get(cam_id)
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

    df = pd.DataFrame(df, columns=INFO_FIELDS, index=index)
    del cam, sc
    output = f'\nCameras Info:\n\n{df.to_string()}\n'
    cam_list.Clear()
    # system.ReleaseInstance()
    return output


def start_camera(cam, dir_path, num_frames, exposure, cache, log_stream):
    """Thread function for configuring and starting spin cameras"""
    sc = SpinCamera(cam, dir_path, num_frames, cache=cache, log_stream=log_stream)
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


def record(exposure=EXPOSURE_TIME, cameras=None, output=OUTPUT_DIR, is_auto_start=False, cache=None, **acquire_stop) -> str:
    """
    Record videos from Arena's cameras
    :param exposure: The exposure time to be set to the cameras
    :param cameras: (str) Cameras to be used. You can specify last digits of p/n or name. (for more than 1 use ',')
    :param output: The output folder for saving the records and log
    :param is_auto_start: Start record automatically or wait for user input
    :param cache: memory cache to be used by the cameras
    """
    assert all(k in ACQUIRE_STOP_OPTIONS for k in acquire_stop.keys())
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    log_stream = get_log_stream()

    if cameras:
        filter_cameras(cam_list, cameras)

    label = datetime.now().strftime('%Y%m%d-%H%M%S')
    dir_path = mkdir(f"{output}/{label}")

    filtered = [(cam, acquire_stop, dir_path, exposure, cache, log_stream) for cam in cam_list]
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
    ap.add_argument("-o", "--output", type=str, default=OUTPUT_DIR,
                    help=f"Specify output directory path. Default={OUTPUT_DIR}")
    ap.add_argument("-e", "--exposure", type=int, default=EXPOSURE_TIME,
                    help=f"Specify cameras exposure time. Default={EXPOSURE_TIME}")
    ap.add_argument("-c", "--camera", type=str, required=False,
                    help=f"filter cameras by last digits or according to CAMERA_NAMES (for more than one use ',').")
    ap.add_argument("-i", "--info", action="store_true", default=False,
                    help=f"Show cameras information")

    args = vars(ap.parse_args())

    if args.get('info'):
        print(display_info())
    else:
        acquire_stop = {}
        for key in ACQUIRE_STOP_OPTIONS:
            if key in args:
                acquire_stop[key] = args[key]
        record(args.get('exposure'), args.get('camera'), args.get('output'), **acquire_stop)


if __name__ == '__main__':
    main()
