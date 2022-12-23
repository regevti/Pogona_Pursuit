import cv2
import pandas as pd
import time
import config
from arrayqueues.shared_arrays import Full
from arena import Camera
from cache import RedisCache, CacheColumns as cc
import vimba

cache = RedisCache()


class AlliedVisionCamera(Camera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_time_delta = 0.0
        self.last_queue_warning_time = None

    def configure(self, cam):
        try:
            cam.ExposureAuto.set('Off')
            cam.ExposureMode.set('Timed')
            cam.ExposureTime.set(self.cam_config['exposure'])
            cam.DeviceLinkThroughputLimit.set(4e8)
            self.logger.debug(f'Throughput: {cam.DeviceLinkThroughputLimit.get():.0e}')
            if self.cam_config.get('pixel_format'):
                cam.set_pixel_format(getattr(vimba.PixelFormat, self.cam_config['pixel_format']))
            if self.cam_config.get('fps') is None:
                cam.AcquisitionFrameRateEnable.set(False)
                cam.TriggerMode.set('Off')
                cam.TriggerSelector.set('FrameStart')
                cam.LineSelector.set('Line3')
                cam.LineMode.set('Input')
                cam.TriggerSource.set('Line3')

                cam.TriggerMode.set('On')
                cam.TriggerActivation.set('RisingEdge')
                self.logger.debug('configured trigger source')
            else:
                cam.TriggerMode.set('Off')
                cam.AcquisitionFrameRateEnable.set(True)
                cam.AcquisitionFrameRate.set(self.cam_config['fps'])
                self.logger.debug(f"configured fps to: {self.cam_config['fps']}")

            cam.AcquisitionMode.set('Continuous')
            self.logger.debug('Finish configuration')
        except Exception:
            self.logger.exception(f"Exception while configuring camera: ")

    def _run(self):
        system = vimba.Vimba.get_instance()
        with system as v:
            cam_id = self.cam_config['id']
            cam = v.get_camera_by_id(cam_id)
            with cam:
                try:
                    self.configure(cam)
                    self.update_time_delta(cam)
                    self.logger.debug('start streaming')
                    cache.append_to_list(cc.RECORDING_CAMERAS, self.cam_name)
                    cam.start_streaming(self._frame_handler, buffer_count=10)
                    self.stop_signal.wait()
                    cache.remove_from_list(cc.RECORDING_CAMERAS, self.cam_name)
                    if self.stop_signal.is_set():
                        self.logger.debug('received stop event')
                except KeyboardInterrupt:
                    pass
                finally:
                    self.mp_metadata['cam_fps'].value = 0.0
                    if cam.is_streaming():
                        cam.stop_streaming()

    def _frame_handler(self, cam, frame):
        t0 = time.time()
        waiting_time = 0.1
        if self.stop_signal.is_set():
            return
        try:
            while True:
                try:
                    img = frame.as_numpy_ndarray()
                    timestamp = frame.get_timestamp() / 1e9 + self.camera_time_delta
                    self.frames_queue.put(img, timestamp)
                    self.calc_fps(timestamp)
                    break
                except Full:
                    if (time.time() - t0) > waiting_time:
                        if not self.last_queue_warning_time or (time.time() - self.last_queue_warning_time > 60):
                            self.logger.warning(f'Queue is still full after waiting {waiting_time}')
                            self.last_queue_warning_time = time.time()
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
    return {cam_name: [AlliedVisionCamera] for cam_name in info_df.index if cam_name != 'unknown'}


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


def get_cam_info(cam):
    info = dict()
    info['Camera Name'] = cam.get_name()
    info['Model Name'] = cam.get_model()
    info['Camera ID'] = cam.get_id()
    info['Serial Number'] = cam.get_serial()
    # info['Interface ID'] = cam.get_interface_id()
    return info
