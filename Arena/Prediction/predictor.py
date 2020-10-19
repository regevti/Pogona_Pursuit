"""
This module is responsible for real-time analysis and prediction of pogona behavior in the arena.

HitPredictor - The entry point of the real-time prediction module. A class that processes camera frames
               and produces predictions of screen touch events (hits).

TrajectoryPredictor - An abstract trajectory predictor class. This class receives a history of bounding box
                      detections and produces a forecast sequence of future bounding box locations.
"""

from Prediction.detector import nearest_detection, xyxy_to_centroid
from Prediction import calibration as calib
import numpy as np
from datetime import datetime
import json
import os


class HitPredictor:
    """
    The main object of the real-time prediction module, exposed to the other modules of the arena system.
    An object responsible for processing video frames into predictions of touch events.

    First, the video frame is passed to a Detector object which returns an array of bounding box detections
    of the Pogona head in the frame. The bounding box coordinates are then transformed to a coordinate system
    relative to the screen using the Calibration module. The transformed detections are added to an array of
    past detections (history attribute) and this data is passed to a TrajectoryPredictor that generates a trajectory
    forecast. From this trajectory forecast a hit prediction consisting of an approximate x coordinate of the hit
    (relative to the monitor resolution), and the number of frames until the predicted hit is generated.

    A hit is predicted when the bottom edge of the bounding box passes a certain y-coordinate threshold
    (prediction_y_threshold attribute).

    All trajectory forecasts are stored in the forecasts attribute (a list of forecast arrays) for later
    analysis if necessary.

    The history array grows dynamically as new frames are processed. The reset() method clears the history
    and should be used to prevent excess memory usage when running for a long time.
    """

    def __init__(
        self,
        trajectory_predictor,
        detector=None,
        history_size=512,
        prediction_y_threshold=0,
        y_thresh_above=False,
    ):
        """
        Initialize HitPredictor.
        Load data from the last camera calibration, which can be overriden using the calibrate method.

        :param trajectory_predictor: an initialized TrajectoryPredictor
        :param detector: an initialized Pogona head object Detector
        :param history_size: initial size of the detections history array.
        :param prediction_y_threshold: y-coordinate threshold for a screen hit relative to the screen coordinate system.
        :param y_thresh_above: boolean indicating whether the threshold is crossed from below or above.
        """

        # look for last homography in some folder and load it. maybe also save dims
        self.homography, cam_width, cam_height = calib.get_last_homography()
        if self.homography is not None:
            # in this case one must call the calibrate() method
            _, _, self.camera_matrix = calib.get_undistort_mapping(
                cam_width, cam_height
            )

        self.trajectory_predictor = trajectory_predictor
        self.detector = detector
        self.prediction_y_threshold = prediction_y_threshold
        self.y_thresh_above = y_thresh_above

        self.reset(history_size=history_size)

    def calibrate(self, cal_img):
        """
        Finds the homography matrix from markers in the image. If found, saves the homography to file for
        later use, stores the new homography matrix in the homography attribute, and returns the matrix with
        a marked image to the caller.

        :param cal_img: numpy opencv image to extract homography from.
        :return: (homography, marked_image, error) where
                 homography - The homography matrix or None in case of error
                 marked_image - cal_img with several markings that can help with debugging
                 error - An error string or None when successful.
        """
        cam_width, cam_height = cal_img.shape[1], cal_img.shape[0]
        mapping, _, self.camera_matrix = calib.get_undistort_mapping(
            cam_width, cam_height
        )
        cal_img = calib.undistort_image(cal_img, mapping)
        h, h_im, error = calib.find_arena_homography(cal_img)

        if error is None:
            self.homography = h
            date = datetime.now().strftime("%Y%m%d-%H%M%S")
            json_name = os.path.join(
                calib.HOMOGRAPHIES_FOLDER, "homog_" + date + ".json"
            )
            d = {"homography": h.tolist(), "width": cam_width, "height": cam_height}

            with open(json_name, "w") as fp:
                json.dump(d, fp)

        return h, h_im, error

    def handle_frame(self, frame):
        """
        Process a single video frame, update trajectory forecast and predict screen touches (hits).
        See handle_detection for returned values.
        """
        if self.frame_num == 0:
            height, width, _ = frame.shape
            if self.detector is not None:
                self.detector.set_input_size(width, height)

        detection = self.detect_pogona_head(frame)

        return self.handle_detection(detection)

    def handle_detection(self, detection):
        """
        Update detection history, send history to trajectory_predictor to generate a trajectory forecast, and
        see whether a hit event is predicted.

        :param detection: xyxy single detection bounding box
        :return: (forecast, hit_point, hit_steps), where
                 forecast - trajectory forecast for the next time steps.
                 hit_point - the predicted x-coordinate of the hit event relative to the screen coordinate system.
                 hit_steps - number of time steps until the predicted hit.
                 return forecast, None, None when there is no hit prediction for this forecast.
        """

        forecast, hit_point, hit_steps = None, None, None

        # Add new detection to history
        self.update_history(detection)

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

    def correct_detection(self, detection):
        """
        Undistort and transforms the detection bounding box coordinates to the screen coordinate system
        using the currently set homography matrix.

        :param detection: xyxy detection bounding box
        :return: corrected bounding box after undistortion and transformation to screen coordinates.
        """
        if self.homography is None:
            raise calib.CalibrationException(
                "HitPredictor has no homography configured"
            )

        x1y1 = calib.undistort_point(detection[:2], self.camera_matrix)
        x2y2 = calib.undistort_point(detection[2:4], self.camera_matrix)
        x1y1 = calib.transform_point(x1y1, self.homography)
        x2y2 = calib.transform_point(x2y2, self.homography)
        return np.concatenate([x1y1, x2y2])

    def detect_pogona_head(self, frame):
        """
        Use the detector to find the pogona's head.
        Return a single best candidate detection or None if there was no detection.
        """
        if self.detector is not None:
            detections = self.detector.detect_image(frame)
        else:
            detections = None
        if detections is not None:
            if len(self.history) > 0:
                prev_centroid = xyxy_to_centroid(self.history[-1])
                detection = nearest_detection(detections, prev_centroid)
            else:
                detection = detections[0]
            if self.camera_matrix is not None and self.homography is not None:
                detection = self.correct_detection(detection)
        else:
            detection = None

        return detection

    def predict_hit(self, forecast):
        """
        Predict when and where the pogona will hit the screen.
        Return the predicted hit point and the number of time steps until the predicted hit.
        :param forecast: a trajectory forecast array.
        :return: (x, hit_idx), where
                 x - x-coordinate of hit (middle of bottom bbox edge)
                 hit_idx - index of first screen touch in the forecast array.
                 Return (None, None) when no hit is predicted.
        """

        # Find hit indices in the forecast using a simple y-coordinate threshold.
        # if data is corrected the screen is above, otherwise it's below.
        if self.y_thresh_above:
            hit_idxs = np.argwhere(forecast[:, 3] >= self.prediction_y_threshold)
        else:
            hit_idxs = np.argwhere(forecast[:, 3] <= self.prediction_y_threshold)

        if hit_idxs.shape[0] == 0:
            return None, None

        hit_idx = int(hit_idxs[0])
        x_val = (forecast[hit_idx][0] + forecast[hit_idx][2]) / 2
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
        Revert HitPredictor to its initialized state. Clears history and forecasts arrays.
        """
        self.frame_num = 0
        self.did_find_detections = False
        self.history = np.empty((history_size, 4), np.float)
        self.history[:] = np.nan
        self.forecasts = []


class TrajectoryPredictor:
    """
    Abstract class for a trajectory predictor. Any object of this type recieves a history of detections and
    predicts future coordinates by implementing the private _update_and_predict method.

    This class contains some machinery for dealing with frames with no detections. See update_and_predict
    for more information.
    """

    def __init__(self, input_len, forecast_horizon):
        """
        Initialize a TrajectoryPredictor.

        :param input_len: length of input sequence. The number of detections that are actually looked back when predicting.
        :param forecast_horizon: length of the generated trajectory forecast.
        """
        self.forecast_horizon = forecast_horizon
        self.input_len = input_len
        self.last_forecast = None
        self.last_forecast_age = None
        self.past_input = np.empty((input_len, 4))
        self.past_input[:] = np.nan

    def init_trajectory(self, detection):
        pass

    def _update_and_predict(self, past_input):
        """
        Abstract private method. Subclasses should override this with the actual prediction logic.
        This is called from the public update_and_predict method.

        :param past_input: numpy array of shape (self.input_len, 4) containing the previous measurements.

        Should return a forecast numpy array of shape (self.forecast_horizon, 4).
        """
        pass

    def update_and_predict(self, history):
        """
        Receive an updated bbox history, call asbtract method _update_and_predict to generate and return a forecast
        trajectory.

        Tries to fill in nan values with values from the previous forecast.
        When the last forecast is too old (older then the entire past_input length), the predictor input
        will contain nan values (and depending on the subclass behavior will likely return nans).
        """

        if self.last_forecast_age is not None:
            self.last_forecast_age += 1

        # current input is the past input, shifted back by 1 index, with the new detection (if exists) placed last index
        self.past_input = np.roll(self.past_input, -1, axis=0)

        if np.isnan(history[-1, 0]):
            # if there is no detection in the current frame, place the relevant prediction from last forecast instead,
            # or NaN when the last forecast that was based on an actual detection history was more than input_len time steps ago.
            if (
                self.last_forecast_age is not None
                and self.last_forecast_age < self.input_len
            ):
                self.past_input[-1] = self.last_forecast[self.last_forecast_age - 1]
            else:
                self.past_input[-1] = np.nan
        else:
            # if detection exists, place it in the input array
            self.past_input[-1] = history[-1]

        # get forecast and update latest forecast if exists
        forecast = self._update_and_predict(self.past_input)
        if not np.isnan(history[-1, 0]):
            self.last_forecast = forecast
            self.last_forecast_age = 0

        return forecast
