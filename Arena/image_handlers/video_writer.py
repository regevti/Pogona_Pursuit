from arena import ImageHandler
import numpy as np
import time
import cv2
import config
from utils import datetime_string


class VideoWriter(ImageHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_frames = 0
        self.timestamps = []
        self.video_out = None

    def __str__(self):
        return f'Video Writer: {self.name}'

    def _on_start(self):
        pass

    def _on_end(self):
        self.log.info(f'Video with {self.n_frames} frames saved into {self.video_path}')
        if len(self.timestamps) > 0:
            self.log.debug(f'fs: {1/np.mean(np.diff(self.timestamps)):.2f}')

    def handle(self, frame, timestamp):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.output_dir and self.video_out is None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            h, w = frame.shape[:2]
            self.video_out = cv2.VideoWriter(self.video_path, fourcc, self.cam_config['fps'], (w, h), True)

        self.video_out.write(frame)
        self.n_frames += 1

    @property
    def video_path(self):
        return f'{self.output_dir}/{self.name}_{datetime_string()}.avi'
