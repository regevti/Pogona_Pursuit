#!/usr/bin/env python

import time
import cv2
import argparse
import serial
from datetime import datetime
import pandas as pd
from multiprocessing.dummy import Pool
import PySpin
from utils import get_logger, calculate_fps, mkdir

DEFAULT_NUM_FRAMES = 1000
OUTPUT_DIR = 'output'
FPS = 60
SAVED_FRAME_RESOLUTION = (640, 480)
SERIAL_PORT = '/dev/usb'
SERIAL_BAUD = 9600
INFO_FIELDS = ['AcquisitionFrameRate', 'AcquisitionMode', 'TriggerSource', 'TriggerMode', 'TriggerSelector',
               'PayloadSize', 'EventSelector', 'LineStatus',
               'DeviceLinkCurrentThroughput', 'DeviceLinkThroughputLimit', 'DeviceMaxThroughput', 'DeviceLinkSpeed']


class SpinCamera:
    def __init__(self, cam, dir_path=None, num_frames=None):
        self.cam = cam
        self.num_frames = num_frames
        self.is_ready = False  # ready for acquisition

        self.cam.Init()
        self.logger = get_logger(self.device_id, dir_path)
        self.video_out = self.configure_video_out(dir_path)

    def begin_acquisition(self):
        """Main function for running camera acquisition in trigger mode"""
        try:
            self.configure_camera()
            self.cam.BeginAcquisition()
            self.is_ready = True
            self.logger.info('Entering to trigger mode')
        except Exception as exc:
            self.logger.error(f'(run); {exc}')

    def __del__(self):
        self.cam.DeInit()
        if self.video_out:
            self.video_out.release()

    def configure_camera(self):
        """Configure camera for trigger mode before acquisition"""
        try:
            self.cam.AcquisitionFrameRateEnable.SetValue(True)
            self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line1)
            self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
            self.cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
            self.cam.LineSelector.SetValue(PySpin.LineSelector_Line1)
            self.cam.LineMode.SetValue(PySpin.LineMode_Input)
            self.cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
            self.cam.DeviceLinkThroughputLimit.SetValue(94578303)
            self.cam.AcquisitionFrameRate.SetValue(60)
            self.cam.ExposureTime.SetValue(15000)
            self.logger.info(f'Finished Configuration')

        except PySpin.SpinnakerException as exc:
            self.logger.error(f'(configure_images); {exc}')

    def configure_video_out(self, dir_path):
        if dir_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            return cv2.VideoWriter(f'{dir_path}/{self.device_id}.avi', fourcc, FPS, SAVED_FRAME_RESOLUTION)

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
        self.is_ready = False

    def image_handler(self, image_result: PySpin.ImagePtr):
        img = image_result.GetNDArray()
        self.video_out.write(img)
        # img.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

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
    def device_id(self):
        nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
        return PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceID')).GetValue()


# def save_list_to_avi(images, fps, device_id, dir_path):
#     try:
#         avi_recorder = PySpin.SpinVideo()
#         avi_filename = f'{dir_path}/{device_id}'
#         option = PySpin.MJPGOption()
#         option.frameRate = fps
#         option.quality = 75
#
#         avi_recorder.Open(avi_filename, option)
#         for img in images:
#             avi_recorder.Append(img)
#         avi_recorder.Close()
#
#         print(f'<CAM {device_id}>: Video saved at {avi_filename}-0000.avi , FPS: {fps:.2f}')
#         return True
#
#     except PySpin.SpinnakerException as exc:
#         print(f'<CAM {device_id}>: ERROR (save_list_to_avi); {exc}')
#         return


class Serializer:
    def __init__(self):
        self.ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)

    def start_acquisition(self):
        self.ser.write(b'H')

    def stop_acquisition(self):
        self.ser.write(b'L')


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


def start_camera(cam, dir_path, num_frames):
    """Thread function for configuring and starting spin cameras"""
    sc = SpinCamera(cam, dir_path, num_frames)
    sc.begin_acquisition()
    return sc


def wait_for_streaming(results: list):
    """Wait for user approval for start streaming and send serial for Arduino to start TTL.
    If keyboard interrupt turn all is_ready to false, so acquisition will not start"""
    serializer = None
    try:
        key = input(f'\nThere are {len([sc for sc in results if sc.is_ready])} ready for streaming.\n'
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


def main():
    """Main function for Arena capture"""
    ap = argparse.ArgumentParser(description="Tool for capturing multiple cameras streams in the arena.")
    ap.add_argument("-n", "--num_frames", type=int, default=DEFAULT_NUM_FRAMES,
                    help=f"Specify Number of Frames. Default={DEFAULT_NUM_FRAMES}")
    ap.add_argument("-o", "--output", type=str, default=OUTPUT_DIR,
                    help=f"Specify output directory path. Default={OUTPUT_DIR}")
    ap.add_argument("-i", "--info", action="store_true", default=False,
                    help=f"Show cameras information")
    args = vars(ap.parse_args())

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num_frames = args.get('num_frames')

    if args.get('info'):
        display_info([c for c in cam_list])

    else:
        label = datetime.now().strftime('%Y%m%d-%H%M%S')
        dir_path = mkdir(f"{args.get('output')}/{label}")

        filtered = [(cam, dir_path, num_frames) for cam in cam_list]
        print(f'\nCameras detected: {len(filtered)}\nNumber of Frames to take: {num_frames}\n')
        if filtered:
            with Pool(len(filtered)) as pool:
                results = pool.starmap(start_camera, filtered)
                results, serializer = wait_for_streaming(results)
                pool.starmap(start_streaming, results)
                if serializer:
                    serializer.stop_acquisition()
        del filtered  # must delete this list in order to destroy all pointers to cameras.

    cam_list.Clear()
    system.ReleaseInstance()


if __name__ == '__main__':
    main()
