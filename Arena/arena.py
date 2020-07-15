#!/usr/bin/env python

import time
import re
import cv2
import argparse
import serial
from datetime import datetime
import pandas as pd
from multiprocessing.dummy import Pool
import PySpin
from utils import get_logger, calculate_fps, mkdir

DEFAULT_NUM_FRAMES = 1000
DEFAULT_MAX_THROUGHPUT = 94578303
EXPOSURE_TIME = 6000
OUTPUT_DIR = 'output'
FPS = 60
SAVED_FRAME_RESOLUTION = (1440, 1088)
SERIAL_PORT = '/dev/ttyACM0'
SERIAL_BAUD = 9600
INFO_FIELDS = ['AcquisitionFrameRate', 'AcquisitionMode', 'TriggerSource', 'TriggerMode', 'TriggerSelector',
               'PayloadSize', 'EventSelector', 'LineStatus', 'ExposureTime',
               'DeviceLinkCurrentThroughput', 'DeviceLinkThroughputLimit', 'DeviceMaxThroughput', 'DeviceLinkSpeed']
CAMERA_NAMES = {
    'realtime': '19506468',
    'right': '19506475'
}


class SpinCamera:
    def __init__(self, cam, dir_path=None, num_frames=None, is_stream=False):
        self.cam = cam
        self.num_frames = num_frames
        self.is_ready = False  # ready for acquisition
        self.dir_path = dir_path
        self.video_out = None
        self.is_stream = is_stream

        self.cam.Init()
        self.logger = get_logger(self.device_id, dir_path)

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

    def get_max_throuput(self):
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
            self.cam.AcquisitionFrameRateEnable.SetValue(True)
            self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line1)
            self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
            self.cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
            self.cam.LineSelector.SetValue(PySpin.LineSelector_Line1)
            self.cam.LineMode.SetValue(PySpin.LineMode_Input)
            self.cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
            self.cam.DeviceLinkThroughputLimit.SetValue(self.get_max_throuput())
            self.cam.ExposureTime.SetValue(exposure)
            self.logger.info(f'Finished Configuration')
            self.log_info()

        except PySpin.SpinnakerException as exc:
            self.logger.error(f'(configure_images); {exc}')

    def acquire(self):
        """Acquire images and measure FPS"""
        if self.is_ready:
            frame_times = list()

            for i in range(self.num_frames):
                try:
                    image_result = self.cam.GetNextImage()  # Retrieve next received image
                    if i == 0:
                        self.logger.info('Acquisition Started')

                    if image_result.IsIncomplete():  # Ensure image completion
                        self.logger.warning(f'Image incomplete with image status {image_result.GetImageStatus()}')
                    else:
                        frame_times.append(time.time())
                        self.image_handler(image_result)

                    image_result.Release()  # Release image

                except PySpin.SpinnakerException as exc:
                    self.logger.error(f'(acquire); {exc}')
                    continue

            self.logger.info(f'Calculated FPS: {calculate_fps(frame_times)}')

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

    def log_info(self):
        """Print into logger the info of the camera"""
        st = ''
        for k, v in zip(INFO_FIELDS, self.info()):
            st += f'{k}: {v}\n'
        self.logger.info(st)

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
    def device_id(self):
        return get_device_id(self.cam)

############################################################################################################

class Serializer:
    def __init__(self):
        self.ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(0.5)

    def start_acquisition(self):
        self.ser.write(b'H')

    def stop_acquisition(self):
        self.ser.write(b'L')

############################################################################################################


def get_device_id(cam) -> str:
    """Get the camera device ID of the cam instance"""
    nodemap_tldevice = cam.GetTLDeviceNodeMap()
    return PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceID')).GetValue()


def filter_cameras(cam_list: PySpin.CameraList, camera_label: str) -> None:
    """Filter cameras according to camera_label, which can be a name or last digits of device ID"""
    devices2remove = []
    device_ids = [get_device_id(c) for c in cam_list]
    if re.match(r'[a-zA-z]+', camera_label):
        device = CAMERA_NAMES.get(camera_label)
        if device and device in device_ids:
            devices2remove = [d for d in device_ids if d != device]
    elif re.match(r'[0-9]+', camera_label):
        devices = [d for d in device_ids if d[-len(camera_label):] == camera_label]
        if devices:
            devices2remove = [d for d in device_ids if d not in devices]
    for d in devices2remove:
        cam_list.RemoveBySerial(d)


def display_info(cam_list):
    """Function for displaying info of all FireFly cameras detected"""
    df = []
    index = []
    for cam in cam_list:
        sc = SpinCamera(cam)
        if not sc.is_firefly():
            continue
        df.append(sc.info())
        index.append(sc.device_id)

    df = pd.DataFrame(df, columns=INFO_FIELDS, index=index)
    print(f'\nCameras Info:\n\n{df}\n')


def start_camera(cam, dir_path, num_frames, exposure):
    """Thread function for configuring and starting spin cameras"""
    sc = SpinCamera(cam, dir_path, num_frames)
    sc.begin_acquisition(exposure)
    return sc


def wait_for_streaming(results: list):
    """Wait for user approval for start streaming and send serial for Arduino to start TTL.
    If keyboard interrupt turn all is_ready to false, so acquisition will not start"""
    serializer = None
    try:
        key = input(f'\nThere are {len([sc for sc in results if sc.is_ready])} cameras ready for streaming.\n'
                    f'Press any key for sending TTL serial to start streaming.\n'
                    f"If you like to start TTL manually press 'm'\n>> ")
        if not key == 'm':
            serializer = Serializer()
            serializer.start_acquisition()

    except Exception as exc:
        print(f'Error: {exc}')
        for sc in results:
            sc.is_ready = False

    return results, serializer


def start_streaming(sc: SpinCamera):
    """Thread function for start acquiring frames from camera"""
    sc.acquire()
    del sc


def main(is_web_stream=False):
    """Main function for Arena capture"""
    ap = argparse.ArgumentParser(description="Tool for capturing multiple cameras streams in the arena.")
    ap.add_argument("-n", "--num_frames", type=int, default=DEFAULT_NUM_FRAMES,
                    help=f"Specify Number of Frames. Default={DEFAULT_NUM_FRAMES}")
    ap.add_argument("-o", "--output", type=str, default=OUTPUT_DIR,
                    help=f"Specify output directory path. Default={OUTPUT_DIR}")
    ap.add_argument("-e", "--exposure", type=int, default=EXPOSURE_TIME,
                    help=f"Specify cameras exposure time. Default={EXPOSURE_TIME}")
    ap.add_argument("-c", "--camera", type=str, required=False,
                    help=f"filter cameras by last digits or according to CAMERA_NAMES.")
    ap.add_argument("-i", "--info", action="store_true", default=False,
                    help=f"Show cameras information")

    args = vars(ap.parse_args())

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num_frames = args.get('num_frames')
    exposure = args.get('exposure')

    if args.get('info'):
        display_info([c for c in cam_list])
    else:
        if args.get('camera'):
            filter_cameras(cam_list, args['camera'])

        label = datetime.now().strftime('%Y%m%d-%H%M%S')
        dir_path = mkdir(f"{args.get('output')}/{label}")

        filtered = [(cam, dir_path, num_frames, exposure) for cam in cam_list]
        print(f'\nCameras detected: {len(filtered)}\nNumber of Frames to take: {num_frames}\n')
        if filtered:
            with Pool(len(filtered)) as pool:
                results = pool.starmap(start_camera, filtered)
                results, serializer = wait_for_streaming(results)
                results = [(sc,) for sc in results]
                pool.starmap(start_streaming, results)
                if serializer:
                    serializer.stop_acquisition()
        del filtered, results  # must delete this list in order to destroy all pointers to cameras.

    cam_list.Clear()
    system.ReleaseInstance()


if __name__ == '__main__':
    main()
