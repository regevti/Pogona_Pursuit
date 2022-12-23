import cv2
import threading
import imageio.v2 as iio
from utils import datetime_string


class Writer:
    video_suffix = '.avi'

    def __init__(self, fps: float, output_dir: str, cam_name: str, is_color=False):
        self.fps = fps
        self.output_dir = output_dir
        self.cam_name = cam_name
        self.is_color = bool(is_color)
        self.writer = None

    def init_writer(self, frame):
        pass

    def write(self, frame):
        pass

    def close(self):
        pass

    @property
    def video_path(self):
        return f'{self.output_dir}/{self.cam_name}_{datetime_string()}{self.video_suffix}'


class OpenCVWriter(Writer):
    def __init__(self, frame, *args):
        super().__init__(*args)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        h, w = frame.shape[:2]
        self.writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (w, h), self.is_color)

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()


class ImageIOWriter(Writer):
    video_suffix = '.mp4'

    def __init__(self, frame, *args):
        super().__init__(*args)
        self.writer = iio.get_writer(self.video_path, format="FFMPEG", mode="I",
                                     fps=self.fps, codec="libx264", quality=5,
                                     macro_block_size=8,  # to work with 1440x1080 image size
                                     ffmpeg_log_level="warning")

    def write(self, frame):
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()