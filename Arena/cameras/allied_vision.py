import vimba
import numpy as np
import pandas as pd
import time
import config
from arrayqueues.shared_arrays import Full
from arena import Camera
from image_handlers.video_writer import VideoWriter
from cache import RedisCache, CacheColumns as cc


cache = RedisCache()


class AlliedVisionCamera(Camera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = vimba.Vimba.get_instance()
        self.camera_time_delta = 0.0

    def configure(self, cam):
        try:
            cam.ExposureAuto.set('Off')
            cam.ExposureMode.set('Timed')
            cam.ExposureTime.set(self.cam_config['exposure'])
            cam.DeviceLinkThroughputLimit.set(450000000)
            # cam.set_pixel_format(vimba.PixelFormat.Mono8)
            if self.cam_config.get('fps') is None:
                cam.AcquisitionFrameRateEnable.set(False)
                cam.TriggerMode.set('Off')
                cam.TriggerSelector.set('FrameStart')
                cam.LineSelector.set('Line3')
                cam.LineMode.set('Input')
                cam.TriggerSource.set('Line3')

                cam.TriggerMode.set('On')
                cam.TriggerActivation.set('FallingEdge')
            else:
                cam.TriggerMode.set('Off')
                cam.AcquisitionFrameRateEnable.set(True)
                cam.AcquisitionFrameRate.set(self.cam_config['fps'])
                self.logger.debug(f"configured fps to: {self.cam_config['fps']}")

            cam.AcquisitionMode.set('Continuous')
            self.logger.info('Finish configuration')
        except Exception:
            self.logger.exception(f"Exception while configuring camera: ")

    def run(self):
        with self.system as v:
            cam_id = self.cam_config['id']
            cam = v.get_camera_by_id(cam_id)
            with cam:
                try:
                    self.configure(cam)
                    self.update_time_delta(cam)
                    self.logger.info('start streaming')
                    cache.append_to_list(cc.RECORDING_CAMERAS, self.cam_name)
                    cam.start_streaming(self._frame_handler, buffer_count=10)
                    self.stop_signal.wait()
                    cache.remove_from_list(cc.RECORDING_CAMERAS, self.cam_name)
                    if self.stop_signal.is_set():
                        self.logger.warning('received stop event')
                except KeyboardInterrupt:
                    pass
                finally:
                    if cam.is_streaming():
                        cam.stop_streaming()

    def _frame_handler(self, cam: vimba.Camera, frame: vimba.Frame):
        t0 = time.time()
        waiting_time = 0.1
        try:
            while True:
                try:
                    img = frame.as_numpy_ndarray()
                    timestamp = frame.get_timestamp() / 1e9 + self.camera_time_delta
                    self.frames_queue.put(img, timestamp)
                    break
                except Full:
                    if (time.time() - t0) > waiting_time:
                        self.logger.warning(f'Queue is still full after waiting {waiting_time}')
                        break
        except Exception:
            self.logger.exception(f"Exception while getting image from alliedVision camera: ")
        finally:
            cam.queue_frame(frame)

    def update_time_delta(self, cam):
        cam.TimestampLatch.run()
        cam_time = cam.TimestampLatchValue.get()  # in nanosecs
        server_time = time.time_ns()
        self.camera_time_delta = (server_time - cam_time) / 1e9


def init():
    info_df = scan_cameras()
    return {cam_name: [AlliedVisionCamera, VideoWriter] for cam_name in info_df.index if cam_name != 'unknown'}


def scan_cameras(is_print=True) -> pd.DataFrame:
    system = vimba.Vimba.get_instance()
    cam_names = []
    with system as v:
        cams = v.get_all_cameras()
        print('Cameras found: {}'.format(len(cams)))
        info = []
        for cam in cams:
            info.append(get_cam_info(cam))
            cam_name = 'unknown'
            for n, cam_config in config.cameras.items():
                if cam_config['id'] == cam.get_id():
                    cam_name = n
                    break
            cam_names.append(cam_name)
        info = pd.DataFrame(info, index=cam_names)

    return info


def get_cam_info(cam: vimba.Camera):
    info = dict()
    info['Camera Name'] = cam.get_name()
    info['Model Name'] = cam.get_model()
    info['Camera ID'] = cam.get_id()
    info['Serial Number'] = cam.get_serial()
    # info['Interface ID'] = cam.get_interface_id()
    return info
