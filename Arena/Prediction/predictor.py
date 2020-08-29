# Module responsible for real-time analysis and prediction of the pogona

from Prediction.detector import Detector_v4, nearest_detection, xyxy_to_centroid
from Prediction.calibration import undistort_point, transform_point
import numpy as np


class HitPredictor:
    def __init__(
        self,
        trajectory_predictor,
        undist_mapping=None,
        aff_transform=None,
        detector=None,
        history_size=512,
        prediction_y_threshold=930,
        y_thresh_above=True,
    ):

        if detector is None:
            detector = Detector_v4()

        self.trajectory_predictor = trajectory_predictor
        self.detector = detector
        self.prediction_y_threshold = prediction_y_threshold
        self.undist_mapping = undist_mapping
        self.aff_transform = aff_transform
        self.y_thresh_above = y_thresh_above

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

        return self.handle_detection(detection)

    def handle_detection(self, detection):
        """
        Return forecast, hit point and hit steps based on detection (x, y, x, y, conf)
        :param detection: xyxy
        :return:
        """
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

    def correct_detection(self,detection):
        """
        :param detection: xyxy bbox
        :return: corrected bbox after undistortion and transformation for screen coordinates
        """
        x1y1 = undistort_point(detection[:2], self.undist_mapping)
        x2y2 = undistort_point(detection[2:], self.undist_mapping)
        x1y1 = transform_point(x1y1, aff)
        x2y2 = transform_point(x2y2, aff)
        return np.concatenate([x1y1, x2y2])

    def detect_pogona_head(self, frame):
        """
        Use the detector to find the pogona's head.
        Return a single best candidate detection or None if there was no detection.
        """
        detections = self.detector.detect_image(frame)
        if detections is not None:
            if len(self.history) > 0:
                prev_centroid = xyxy_to_centroid(self.history[-1])
                detection = nearest_detection(detections, prev_centroid)
            else:
                detection = detections[0]
        else:
            detection = None

        if self.undist_mapping is not None and self.aff_transform is not None:
            detection = self.correct_detection(detection)
        return detection

    def predict_hit(self, forecast):
        """
        Predict when and where the pogona will hit the screen.
        Return the predicted hit point and the number of time steps until
        the predicted hit.
        :param forecast: an (forecast horizon length, 4) array, each row wth x1 y1 x2 y2
        :return: x value of hit (middle of edge) ,index of first touch in screen in forcast array
        """

        # if data is corrected, screen is above, else it's below
        if self.y_thresh_above:
            hit_idxs = np.argwhere(forecast[:, 3] >= self.prediction_y_threshold)
        else:
            hit_idxs = np.argwhere(forecast[:, 3] <= self.prediction_y_threshold)

        if hit_idxs.shape[0] == 0:
            return None, None

        hit_idx = int(hit_idxs[0])
        x_val = (forecast[hit_idx][0] + forecast[hit_idx][2])/2
        return x_val, hit_idx

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
