import vimba
import numpy as np
import pandas as pd
import time
import config
from arena import Camera
from image_handlers.video_writer import VideoWriter


class AlliedVisionCamera(Camera):
    def __init__(self, *args, **kwargs):
        super(AlliedVisionCamera, self).__init__(*args, **kwargs)
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
                self.log.debug(f"configured fps to: {self.cam_config['fps']}")

            cam.AcquisitionMode.set('Continuous')
            self.log.info('Finish configuration')
        except Exception:
            self.log.exception(f"Exception while configuring camera: ")

    def run(self):
        with self.system as v:
            cam_id = self.cam_config['id']
            cam = v.get_camera_by_id(cam_id)
            with cam:
                try:
                    self.configure(cam)
                    self.update_time_delta(cam)
                    if self.start_event is not None:
                        self.start_event.wait()
                    self.log.info('start streaming')
                    cam.start_streaming(self._frame_handler)
                    self.stop_event.wait()
                    if self.stop_event.is_set():
                        self.log.info('received stop event')
                except KeyboardInterrupt:
                    pass
                finally:
                    if cam.is_streaming():
                        cam.stop_streaming()

    def _frame_handler(self, cam: vimba.Camera, frame: vimba.Frame):
        try:
            self.image_unloaded.wait(timeout=0.2)
            img = frame.as_numpy_ndarray()
            with self.lock:
                self.frame_timestamp.value = frame.get_timestamp() / 1e9 + self.camera_time_delta
                buf_np = np.frombuffer(self.shm.buf, dtype=config.shm_buffer_dtype).reshape(self.cam_config['image_size'])
                np.copyto(buf_np, img)
                self.image_unloaded.clear()
                if not self.cam_ready.is_set():
                    self.cam_ready.set()
        except Exception:
            self.log.exception(f"Exception while getting image from alliedVision camera: ")
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
        if is_print:
            with pd.option_context('display.max_colwidth', None,
                                   'display.max_columns', None,
                                   'display.max_rows', None):
                print(info)
                print()
    return info


def get_cam_info(cam: vimba.Camera):
    info = dict()
    info['Camera Name'] = cam.get_name()
    info['Model Name'] = cam.get_model()
    info['Camera ID'] = cam.get_id()
    info['Serial Number'] = cam.get_serial()
    # info['Interface ID'] = cam.get_interface_id()
    return info
