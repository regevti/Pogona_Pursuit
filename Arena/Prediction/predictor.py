# Module responsible for real-time analysis and prediction of the pogona

from Prediction.detector import nearest_detection, xyxy_to_centroid
from Prediction import calibration as calib
import numpy as np
from datetime import datetime
import json
import os


class HitPredictor:
    def __init__(
        self,
        trajectory_predictor,
        detector=None,
        history_size=512,
        prediction_y_threshold=0,
        y_thresh_above=True,
    ):

        # look for last homography in some folder and load it. maybe also save dims
        self.homography, cam_width, cam_height = calib.get_last_homography()
        if self.homography is not None:
            _, _, self.camera_matrix = calib.get_undistort_mapping(cam_width, cam_height)

        # TODO: need to calibrate if no homography is found

        self.trajectory_predictor = trajectory_predictor
        self.detector = detector
        self.prediction_y_threshold = prediction_y_threshold
        self.y_thresh_above = y_thresh_above

        self.reset(history_size=history_size)

    def calibrate(self, cal_img):
        """
        Finds homography from markers in the image. If homography is found,
        saves it to file and returns the homography and the marked image to caller,
        and also updates the HitPredictor homography.
        If homography is not found, doesn't save anything to file, returns None,
        marked image and the error.
        :param cal_img: numpy opencv image to extract homography from
        :return: Homography, marked image with markers, error
        """
        cam_width, cam_height = cal_img.shape[1], cal_img.shape[0]
        (mapx, mapy), _, self.camera_matrix = calib.get_undistort_mapping(cam_width, cam_height)
        cal_img = calib.undistort_image(cal_img, (mapx, mapy))
        h, h_im, error = calib.find_arena_homography(cal_img)

        if error is None:
            self.homography = h
            date = datetime.now().strftime('%Y%m%d-%H%M%S')
            json_name = os.path.join(calib.HOMOGRAPHIES_FOLDER, 'homog_'+date+'.json')
            d = {'homography': h.tolist(),
                 'width': cam_width,
                 'height': cam_height}

            with open(json_name, 'w') as fp:
                json.dump(d, fp)

        return h, h_im, error

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

    def correct_detection(self, detection):
        """
        :param detection: xyxy bbox
        :return: corrected bbox after undistortion and transformation for screen coordinates
        """
        if self.homography is None:
            raise calib.CalibrationException("HitPredictor has no homography configured")
        x1y1 = calib.undistort_point(detection[:2], self.camera_matrix)
        x2y2 = calib.undistort_point(detection[2:], self.camera_matrix)
        x1y1 = calib.transform_point(x1y1, self.homography)
        x2y2 = calib.transform_point(x2y2, self.homography)
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
            if self.camera_matrix is not None and self.homography is not None:
                detection = self.correct_detection(detection)
        else:
            detection = None

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
