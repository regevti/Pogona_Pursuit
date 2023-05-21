import cv2
import time
import numpy as np
import pandas as pd

import config
from pathlib import Path
from arena import ImageHandler, QueueException
from cache import RedisCache, CacheColumns as cc
from utils import run_in_thread
from analysis.predictors.tongue_out import TongueOutAnalyzer, TONGUE_CLASS
from analysis.pose import ArenaPose


class PredictHandler(ImageHandler):
    def __init__(self, *args, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.cache = RedisCache()
        self.last_timestamp = None
        self.is_initiated = False
        self.prediction_summary = ''

    def __str__(self):
        return f'Predictor-{self.cam_name}'

    def _init(self, img):
        try:
            self.init(img)
        except Exception as exc:
            raise Exception(f'Could not initiate predictor; {exc}')

    def init(self, img):
        pass

    def loop(self):
        self.logger.info('start predictor loop')
        try:
            while not self.stop_signal.is_set() and not self.mp_metadata['predictors_stop'].is_set():
                timestamp = self.wait_for_next_frame()
                img = np.frombuffer(self.shm.buf, dtype=config.shm_buffer_dtype).reshape(self.cam_config['image_size'])
                img = self.before_predict(img)

                pred, img = self.predict_frame(img, timestamp)
                if not self.is_initiated:
                    self._init(img)
                self.analyze_prediction(timestamp, pred)

                # copy the image+predictions to pred_shm
                if self.pred_image_size is not None and self.is_streaming:
                    img = self.draw_pred_on_image(pred, img)
                    buf_np = np.frombuffer(self.pred_shm.buf, dtype=config.shm_buffer_dtype).reshape(
                        self.pred_image_size)
                    np.copyto(buf_np, img)

                t_end = time.time()
                self.calc_fps(t_end)
                self.calc_pred_delay(timestamp, t_end)
                self.last_timestamp = timestamp
        finally:
            self.on_stop()
            self.mp_metadata[self.calc_fps_name].value = 0.0
            self.mp_metadata['pred_delay'].value = 0.0
            self.logger.info('predict loop is closed')

    def on_stop(self):
        pass

    def before_predict(self, img):
        return img

    def predict_frame(self, img, timestamp):
        """Return the prediction vector and the image itself in case it was changed"""
        return None, img

    def analyze_prediction(self, timestamp, pred):
        pass

    def get_db_video_id(self):
        return self.mp_metadata['db_video_id'].value or None

    def draw_pred_on_image(self, det, img):
        return img

    def wait_for_next_frame(self, timeout=2):
        current_timestamp = self.mp_metadata['shm_frame_timestamp'].value
        t0 = time.time()
        while self.last_timestamp and current_timestamp == self.last_timestamp:
            if time.time() - t0 > timeout:
                raise QueueException(f'{str(self)} waited for {timeout} seconds and no new frames came; abort')
            current_timestamp = self.mp_metadata['shm_frame_timestamp'].value
            time.sleep(0.001)
        return current_timestamp


class TongueOutHandler(PredictHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = TongueOutAnalyzer(action_callback=self.publish_tongue_out)
        self.last_detected_ts = None
        self.frames_probas = []

    def __str__(self):
        return f'tongue-out-{self.cam_name}'

    def publish_tongue_out(self):
        self.cache.publish_command('strike_predicted')
        self.logger.info('Tongue detected!')

    def predict_frame(self, img, timestamp):
        is_tongue, resized_img, prob = self.analyzer.predict(img, timestamp)
        self.frames_probas.append((timestamp, prob))
        if is_tongue:
            self.publish_tongue_out()
            self.last_detected_ts = timestamp
        return is_tongue, resized_img

    def log_prediction(self, is_tongue, timestamp):
        pass

    def on_stop(self):
        try:
            block_path = self.cache.get(cc.EXPERIMENT_BLOCK_PATH)
            if not block_path:
                self.logger.warning('unable to save tounge predictions; block path is None')
                return

            df = pd.DataFrame(self.frames_probas, columns=['timestamp', 'prob'])
            df.to_csv(Path(block_path) / 'tongue_probas.csv')
        except Exception as exc:
            self.logger.error(f'unable to save tongue_probas; {exc}')

    def draw_pred_on_image(self, prob, img):
        if not prob:
            return img

        h, w = img.shape[:2]
        font, color = cv2.FONT_HERSHEY_SIMPLEX, (255, 0, 255)
        img = cv2.putText(img, f'P={prob:.2f}', (20, h - 30), font, 1, color, 2, cv2.LINE_AA)
        # img = put_text(f'P={prob:.2f}', img, img.shape[1] - 120, 30, color=(255, 0, 0))
        return img


class PogonaHeadHandler(PredictHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arena_pose = ArenaPose(self.cam_name, 'pogona_head')
        self.last_det = None

    def __str__(self):
        return f'pogona-head-{self.cam_name}'

    def loop(self):
        detector = self.arena_pose.predictor.detector
        self.logger.info(
            f"YOLOv5 detector loaded successfully ({detector.model_width}x{detector.model_height} "
            f"weights: {detector.weights_path})."
        )
        super().loop()

    def _init(self, img):
        self.arena_pose.init(img)
        self.is_initiated = True

    def predict_frame(self, img, timestamp):
        """Get detection of pogona head on frame; {det := [x1, y1, x2, y2, confidence]}"""
        det, img = self.arena_pose.predictor.predict(img, return_centroid=False)
        return det, img

    def analyze_prediction(self, timestamp, det):
        if det[0] is None:
            return

        db_video_id = self.get_db_video_id()
        cam_x, cam_y = self.arena_pose.predictor.to_centroid(det)
        self.prediction_summary = self.arena_pose.analyze_frame(timestamp, cam_x, cam_y, db_video_id)

    def draw_pred_on_image(self, det, img, font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0)):
        img = self.arena_pose.predictor.draw_predictions(det, img)
        h, w = img.shape[:2]
        if self.prediction_summary:
            img = cv2.putText(img, str(self.prediction_summary), (20, h-30), font, 1, color, 2, cv2.LINE_AA)
        self.last_det = det
        return img