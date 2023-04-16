import cv2
from utils import datetime_string
import config


class Writer:
    video_suffix = '.avi'

    def __init__(self, fps: float, output_dir: str = None, cam_name: str = None, is_color=False, full_path=None):
        self.fps = fps
        self.output_dir = output_dir
        self.cam_name = cam_name
        self.full_path = full_path
        self.is_color = bool(is_color)
        self.writer = None

    def write(self, frame):
        pass

    def close(self):
        pass

    @property
    def video_path(self):
        return self.full_path or f'{self.output_dir}/{self.cam_name}_{datetime_string()}{self.video_suffix}'


class OpenCVWriter(Writer):
    def __init__(self, frame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_WRITER_FORMAT)
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
        import imageio.v2 as iio
        self.writer = iio.get_writer(self.video_path, format="FFMPEG", mode="I",
                                     fps=self.fps, codec="libx264", quality=5,
                                     macro_block_size=8,  # to work with 1440x1080 image size
                                     ffmpeg_log_level="warning")

    def write(self, frame):
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()