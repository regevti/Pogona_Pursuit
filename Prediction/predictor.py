# Module responsible for real-time analysis and prediction of the pogona

from Prediction.detector import Detector_v4, nearest_detection, xywh_to_centroid
import numpy as np


class HitPredictor:
    def __init__(
        self,
        trajectory_predictor,
        detector=None,
        history_size=512,
        prediction_y_threshold=930,
    ):

        if detector is None:
            detector = Detector_v4()

        self.trajectory_predictor = trajectory_predictor
        self.detector = detector
        self.prediction_y_threshold = prediction_y_threshold

        self.reset(history_size=history_size)

    def handle_frame(self, frame):
        """
        Process a single frame, update prediction, and send prediction as
        an MQTT message (TBD).
        """
        if self.frame_num == 0:
            height, width, _ = frame.shape
            self.detector.set_input_size(width, height)

        detection = self.detect_pogona_head(frame)
        self.update_history(detection)

        forecast, hit_point, hit_steps = None, None, None

        if not self.did_find_detections:
            # First detection. Initialize trajectory predictor.
            if detection is not None:
                self.did_find_detections = True
                self.trajectory_predictor.init_trajectory(detection)
        else:
            # Update trajectory predictor.
            forecast = self.trajectory_predictor.update_and_predict(
                self.history[: self.frame_num + 1]
            )
            if forecast is not None:
                hit_point, hit_steps = self.predict_hit(forecast)

        self.frame_num += 1
        self.forecasts.append(forecast)

        return forecast, hit_point, hit_steps

    def detect_pogona_head(self, frame):
        """
        Use the detector to find the pogona's head.
        Return a single best candidate detection or None if there was no detection.
        """
        detections = self.detector.detect_image(frame)
        if detections is not None:
            if len(self.history) > 0:
                prev_centroid = xywh_to_centroid(self.history[-1])
                detection = nearest_detection(detections, prev_centroid)
            else:
                detection = detections[0]
        else:
            detection = None

        return detection

    def predict_hit(self, forecast):
        """
        Predict when and where the pogona will hit the screen.
        Return the predicted hit point and the number of time steps until
        the predicted hit.
        """
        hit_idx = np.argmax(forecast[:, 1] >= self.prediction_y_threshold)
        if hit_idx == 0:
            if forecast[0, 1] >= self.prediction_y_threshold:
                return forecast[0], 0
            else:
                return None, None
        else:
            return forecast[hit_idx], hit_idx

    def update_history(self, detection):
        """
        Add detection to history, and double the history size if necessary.
        """
        history_size = self.history.shape[0]
        if self.frame_num >= history_size:
            new_hist = np.empty((history_size, 4), np.float)
            new_hist[:] = np.nan

            self.history = np.vstack((self.history, new_hist))

        if detection is not None:
            self.history[self.frame_num] = detection[:4]

    def reset(self, history_size=512):
        """
        Revert HitPredictor to its initialized state.
        Clear history etc.
        """
        self.frame_num = 0
        self.did_find_detections = False
        self.history = np.empty((history_size, 4), np.float)
        self.history[:] = np.nan
        self.forecasts = []


class TrajectoryPredictor:
    def __init__(self, forecast_horizon):
        self.forecast_horizon = forecast_horizon

    def init_trajectory(self, detection):
        pass

    def update_and_predict(self, history):
        """
        Receive an updated bbox history and generate and return a forecast
        trajectory.
        """
        pass
