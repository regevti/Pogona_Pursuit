import cv2
import time
import numpy as np
from image_handlers.base_predictor import Predictor
from analysis.tongue_out import TongueDetector, TongueOutAnalyzer, TONGUE_CLASS, TONGUE_PREDICTED_DIR
from utils import run_in_thread


class TongueOutImageHandler(Predictor):
    is_use_caliber = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = TongueOutAnalyzer(action_callback=self.publish_tongue_out)
        self.last_detected_ts = None

    def __str__(self):
        return f'tongue-out-{self.cam_name}'

    def loop(self):
        self.logger.info(f'Tongue-out model was loaded')
        super().loop()

    def publish_tongue_out(self):
        self.cache.publish_command('strike_predicted')
        self.logger.info('Tongue detected!')

    def predict_frame(self, img, timestamp):
        is_tongue, resized_img, _ = self.analyzer.predict(img, timestamp)
        if is_tongue:
            self.publish_tongue_out()
            self.last_detected_ts = timestamp
            self.save_predicted(img, timestamp)
        return is_tongue, resized_img

    @run_in_thread
    def save_predicted(self, img, timestamp):
        cv2.imwrite(f'{TONGUE_PREDICTED_DIR}/{timestamp}.jpg', img)

    def log_prediction(self, is_tongue, timestamp):
        pass

    def analyze_predictions(self, predictions: list, current_db_video_id, predictions_start_time):
        pass

    @run_in_thread
    def commit_to_db(self):
        pass

    def draw_pred_on_image(self, is_tongue, img):
        if not is_tongue:
            return img

        h, w = img.shape[:2]
        font, color = cv2.FONT_HERSHEY_SIMPLEX, (255, 0, 255)
        img = cv2.putText(img, f'Tongue Detected!', (20, h - 30), font, 1, color, 2, cv2.LINE_AA)
        return img
