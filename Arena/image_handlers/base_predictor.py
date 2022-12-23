import datetime
import time
import numpy as np
from arena import ImageHandler, QueueException
from db_models import ORM
from cache import RedisCache
from utils import run_in_thread
import config
from calibration import PoseEstimator


class Predictor(ImageHandler):
    is_use_caliber = True

    def __init__(self, *args, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.orm = ORM()
        self.cache = RedisCache()
        self.predictions = None
        self.current_db_video_id = None
        self.predictions_start_time = None
        self.last_timestamp = None
        self.last_commit = (time.time(), 0, 0)  # (timestamp, x, y)
        self.commit_interval = 2  # seconds
        caliber_image_size = self.pred_image_size[:2] if self.pred_image_size else None
        if self.is_use_caliber:
            self.caliber = PoseEstimator(self.cam_name, resize_dim=caliber_image_size, logger=self.logger)

    def __str__(self):
        return f'Predictor-{self.cam_name}'

    def loop(self):
        try:
            while not self.stop_signal.is_set():
                db_video_id = self.get_db_video_id()
                # # cases to restart predictions log:
                # # 1. new experiment-block started
                # # 2. experiment-block ended
                # # 3. max elements in self.predictions reached
                # if (db_video_id and db_video_id != self.current_db_video_id) or \
                #     (not db_video_id and self.current_db_video_id) or \
                #         (self.predictions and len(self.predictions) > config.max_predictor_rows):
                if (time.time() - self.last_commit[0]) >= self.commit_interval:
                    self.end_predictions_log()

                if self.predictions is None:
                    self.start_predictions_log(db_video_id)

                timestamp = self.wait_for_next_frame()
                img = np.frombuffer(self.shm.buf, dtype=config.shm_buffer_dtype).reshape(self.cam_config['image_size'])
                img = self.before_predict(img)

                det, img = self.predict_frame(img, timestamp)
                if self.is_use_caliber and self.caliber.state == 0:
                    self.caliber.init(img)

                if self.predictions is not None:
                    self.log_prediction(det, timestamp)

                # copy the image+predictions to pred_shm
                if self.pred_image_size is not None and self.is_streaming:
                    img = self.draw_pred_on_image(det, img)
                    buf_np = np.frombuffer(self.pred_shm.buf, dtype=config.shm_buffer_dtype).reshape(
                        self.pred_image_size)
                    np.copyto(buf_np, img)

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
        return

    def end_predictions_log(self):
        if self.predictions:
            self._analyze_predictions(self.predictions.copy(), self.current_db_video_id, self.predictions_start_time)
        self.predictions = None
        self.current_db_video_id = None
        self.predictions_start_time = None

    @run_in_thread
    def _analyze_predictions(self, predictions: list, current_db_video_id, predictions_start_time):
        self.analyze_predictions(predictions, current_db_video_id, predictions_start_time)

    def analyze_predictions(self, predictions: list, current_db_video_id, predictions_start_time):
        pass

    def commit_to_db(self):
        self.orm.commit_video_predictions(predictor_name=str(self), data=self.predictions,
                                          video_id=self.current_db_video_id, start_time=self.predictions_start_time)

    def get_db_video_id(self):
        return self.mp_metadata['db_video_id'].value or None

    def draw_pred_on_image(self, det, img):
        return img

    def wait_for_next_frame(self, timeout=2):
        current_timestamp = self.mp_metadata['shm_frame_timestamp'].value
        t0 = time.time()
        while self.last_timestamp and current_timestamp == self.last_timestamp:
            if (time.time() - t0 > timeout):
                raise QueueException(f'{str(self)} waited for {timeout} seconds and no new frames came; abort')
            current_timestamp = self.mp_metadata['shm_frame_timestamp'].value
            time.sleep(0.001)
        return current_timestamp
