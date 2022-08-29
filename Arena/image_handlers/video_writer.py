import threading
import multiprocessing as mp
from arena import ImageSink
import numpy as np
import time
import cv2
import config
from utils import datetime_string


class VideoWriter(ImageSink):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_frames = 0
        self.timestamps = []
        self.queue = mp.Queue(-1)
        self.video_out = None

    def __str__(self):
        return f'Video Writer: {self.cam_name}'

    def _on_start(self):
        pass

    def _on_end(self):
        self.logger.info(f'Video with {self.n_frames} frames saved into {self.video_path}')
        if len(self.timestamps) > 0:
            self.logger.debug(f'fs: {1 / np.mean(np.diff(self.timestamps)):.2f}')

    def handle(self, frame_, timestamp):
        if self.cam_config['output_dir'] and self.video_out is None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            h, w = frame_.shape[:2]
            self.video_out = cv2.VideoWriter(self.video_path, fourcc, self.cam_config['fps'], (w, h), True)

        def _handle(frame, queue):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            queue.put_nowait(frame)

        t = threading.Thread(target=_handle, args=(frame_, self.queue))
        t.start()
        self.n_frames += 1

    @property
    def video_path(self):
        return f'{self.cam_config["output_dir"]}/{self.cam_name}_{datetime_string()}.avi'
