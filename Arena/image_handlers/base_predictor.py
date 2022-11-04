import datetime
import time
import numpy as np
from arena import ImageHandler
from db_models import ORM
from cache import RedisCache
from utils import run_in_thread
import config
from calibration import PoseEstimator


class Predictor(ImageHandler):
    def __init__(self, *args, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.orm = ORM()
        self.cache = RedisCache()
        self.predictions = None
        self.current_db_video_id = None
        self.predictions_start_time = None
        self.last_timestamp = None
        self.caliber = PoseEstimator(self.cam_name,
                                     resize_dim=self.pred_image_size[:2][::-1] if self.pred_image_size else None)

    def __str__(self):
        return f'Predictor-{self.cam_name}'

    def loop(self):
        try:
            while not self.stop_signal.is_set():
                db_video_id = self.get_db_video_id()
                # cases to restart predictions log:
                # 1. new experiment-block started
                # 2. experiment-block ended
                # 3. max elements in self.predictions reached
                if (db_video_id and db_video_id != self.current_db_video_id) or \
                    (not db_video_id and self.current_db_video_id) or \
                        (self.predictions and len(self.predictions) > config.max_predictor_rows):
                    self.end_predictions_log()

                if self.predictions is None:
                    self.start_predictions_log(db_video_id)

                timestamp = self.wait_for_next_frame()
                img = np.frombuffer(self.shm.buf, dtype=config.shm_buffer_dtype).reshape(self.cam_config['image_size'])
                img = self.before_predict(img)
                if not self.caliber.is_initiated:
                    self.caliber.init(img)
                det, img = self.predict_frame(img, timestamp)
                # copy the image+predictions to pred_shm
                if self.pred_image_size is not None:
                    buf_np = np.frombuffer(self.pred_shm.buf, dtype=config.shm_buffer_dtype).reshape(
                        self.pred_image_size)
                    np.copyto(buf_np, img)

                if self.predictions is not None:
                    self.log_prediction(det, timestamp)

                t_end = time.time()
                self.calc_fps(t_end)
                self.calc_pred_delay(timestamp, t_end)
                self.last_timestamp = timestamp
        finally:
            self.mp_metadata[self.calc_fps_name].value = 0.0
            self.mp_metadata['pred_delay'].value = 0.0

    def before_predict(self, img):
        return img

    def predict_frame(self, img, timestamp):
        """Return the prediction vector and the image itself in case it was changed"""
        return None, img

    def start_predictions_log(self, db_video_id):
        self.predictions = []
        self.predictions_start_time = datetime.datetime.now()
        self.current_db_video_id = db_video_id

    def log_prediction(self, det, timestamp):
        raise NotImplemented()

    def end_predictions_log(self):
        if self.predictions:
            self.commit_to_db()
        self.predictions = None
        self.current_db_video_id = None
        self.predictions_start_time = None

    @run_in_thread
    def commit_to_db(self):
        self.orm.commit_video_predictions(predictor_name=str(self), data=self.predictions,
                                          video_id=self.current_db_video_id, start_time=self.predictions_start_time)

    def get_db_video_id(self):
        return self.mp_metadata['db_video_id'].value or None

    def wait_for_next_frame(self, timeout=2):
        current_timestamp = self.mp_metadata['shm_frame_timestamp'].value
        t0 = time.time()
        while self.last_timestamp and current_timestamp == self.last_timestamp:
            if (time.time() - t0 > timeout):
                raise Exception(f'waited for {timeout} seconds and no new frames came; abort')
            current_timestamp = self.mp_metadata['shm_frame_timestamp'].value
            time.sleep(0.001)
        return current_timestamp