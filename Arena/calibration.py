import json

import cv2
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import config
import utils
from loggers import get_logger

ARUCO_DICT = cv2.aruco.DICT_4X4_1000
ARUCO_IDS = np.arange(240).tolist()
CENTER_ID = 0  # the marker that stands for (0, 0) in the new coordinate system
IDS2PLOT = [0, 7, 160, 175, 232, 239]
TEST_IDS = []  # this marker should have both positive coordinates, if not happen will induce fix
SHIFTED_CAMS = ['top']  # frames that their x,y need to be swapped
CHARUCO_COLS = 8


class CalibrationError(Exception):
    """"""


class Calibrator:
    def __init__(self, cam_name, resize_dim=None):
        self.cam_name = cam_name
        self.logger = get_logger('calibrator')
        self.calib_params = {
            'mtx': None,
            'dist': None,
            'rvecs': None,
            'tvecs': None,
            'w': None,
            'h': None
        }
        self.resize_dim = resize_dim
        self.undistort_mappers = None
        self.load_calibration()

    def calibrate_camera(self, is_plot=True):
        self.logger.info(f'start camera {self.cam_name} calibration')
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((np.prod(config.CHESSBOARD_DIM), 3), np.float32)
        objp[:, :2] = np.mgrid[:config.CHESSBOARD_DIM[0], :config.CHESSBOARD_DIM[1]].T.reshape(-1, 2)
        img_files = self.get_calib_images()
        if is_plot:
            cols = 3
            rows = int(np.ceil(len(img_files) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(30, 5 * rows))
            axes = axes.flatten()

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        for i, p in enumerate(img_files):
            gray = cv2.imread(p.as_posix(), 0)
            if self.resize_dim:
                gray = cv2.resize(gray, self.resize_dim)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, config.CHESSBOARD_DIM, None)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners_ = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners_)
                if is_plot:
                    cv2.drawChessboardCorners(gray, config.CHESSBOARD_DIM, corners_, ret)
                    axes[i].imshow(gray)

        h, w = gray.shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
        if not ret:
            raise CalibrationError('calibrateCamera failed')
        # save calibration params to class
        for k in self.calib_params.copy().keys():
            self.calib_params[k] = locals().get(k)
        self.save_calibration()
        if is_plot:
            fig.tight_layout()
            fig.savefig(self.calib_image_detections_path)
        self.calc_projection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
        self.logger.info('calibration finished successfully')

    def undistort_image(self, img) -> np.ndarray:
        return cv2.remap(img, *self.undistort_mappers, cv2.INTER_LINEAR)

    def calc_undistort_mappers(self):
        mtx, dist = self.calib_params['mtx'], self.calib_params['dist']
        w, h = self.calib_params['w'], self.calib_params['h']
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # undistort
        self.undistort_mappers = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

    def calc_projection_error(self, objpoints, imgpoints, rvecs, tvecs, mtx, dist):
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        self.logger.info("total projection error: {}".format(mean_error / len(objpoints)))

    def load_calibration(self):
        if not self.calib_params_path.exists():
            self.calibrate_camera()

        with self.calib_params_path.open('rb') as f:
            self.calib_params = pickle.load(f)
        self.calc_undistort_mappers()

    def save_calibration(self):
        with self.calib_params_path.open('wb') as f:
            pickle.dump(self.calib_params, f)

    def get_calib_images(self):
        calib_dir = Path(config.calibration_dir) / self.cam_name
        if not calib_dir.exists():
            raise CalibrationError(f'{calib_dir} not exist')
        img_files = list(calib_dir.glob('*.png'))
        if len(img_files) < config.min_calib_images:
            raise CalibrationError(f'found only {len(img_files)} images for calibration, '
                                   f'expected {config.min_calib_images}')
        return img_files

    @property
    def calib_params_path(self):
        return Path(config.calibration_dir) / f'calib_params_{self.cam_name}_{json.dumps(self.resize_dim)}.pkl'

    @property
    def calib_image_detections_path(self):
        return Path(config.calibration_dir) / f'calib_detections_{self.cam_name}.png'


SIGN_MASKS = {
    'front': [-1, -1],
    'top': [1, -1]
}


class CharucoEstimator:
    def __init__(self, cam_name, resize_dim=None, logger=None, is_debug=True):
        self.cam_name = cam_name
        self.resize_dim = resize_dim
        self.resize_scale = None
        self.is_debug = is_debug
        self.id_key = 'id'
        self.logger = get_logger('calibration-pose-estimator') if logger is None else logger
        self.calibrator = Calibrator(cam_name, resize_dim=resize_dim)
        self.arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.is_swap_xy = cam_name in SHIFTED_CAMS
        self.sign_mask = SIGN_MASKS[cam_name]
        self.state = 0  # 0 - not initiated, 1 - failure, 2 - initiated
        self.markers = {}

    def __str__(self):
        return self.markers.get(self.id_key)

    def init(self, img, img_shape=None, is_plot=False):
        try:
            if img_shape is None:
                img_shape = img.shape[:2]
            if self.resize_dim:
                self.resize_scale = (img.shape[0] // self.resize_dim[0], img.shape[1] // self.resize_dim[1])
                if json.dumps(img_shape) != json.dumps(self.resize_dim):
                    raise Exception(f'Image size does not fit. expected: {tuple(self.resize_dim)}, received: {tuple(img_shape)}')

            self.load_markers()
            # self.align_axes()
            self.state = 2
            if self.is_debug:
                self.logger.info(f'started pose-estimator caliber for frames of shape: {img_shape}')
            if is_plot:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = self.plot_calibrated_line(img)
                plt.imshow(img)
                plt.show()
        except Exception as exc:
            if self.is_debug:
                self.logger.error(f'Error in pose-estimator init; {exc}')
            self.state = 1
        return self

    def get_location(self, frame_x, frame_y, check_init=True):
        """Convert camera x, y to real-world coordinates"""
        if check_init and not self.is_initiated:
            return

        if self.resize_scale:
            frame_x, frame_y = self.resize_scale[0] * frame_x, self.resize_scale[1] * frame_y
        dists = {marker_id: distance.euclidean((frame_x, frame_y), d['top_left'])
                 for marker_id, d in self.markers.items() if marker_id != self.id_key}
        if not dists:
            return
        closest_marker_id = min(dists, key=dists.get)
        d = self.markers[closest_marker_id]
        top = distance.euclidean(d['top_right'], d['top_left'])
        side = distance.euclidean(d['top_right'], d['bottom_right'])
        pixel2cm_x = config.ARUCO_MARKER_SIZE / (side if self.is_swap_xy else top)
        pixel2cm_y = config.ARUCO_MARKER_SIZE / (top if self.is_swap_xy else side)
        dx = self.sign_mask[0] * (frame_x - d['top_left'][0])
        dy = self.sign_mask[1] * (frame_y - d['top_left'][1])
        xr = d['x'] + pixel2cm_x * (dx if not self.is_swap_xy else dy)
        yr = d['y'] + pixel2cm_y * (dy if not self.is_swap_xy else dx)
        return xr, yr

    def find_aruco_markers(self, frame, is_plot=True):
        if frame.shape[2] > 1:
            gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            gray_frame = frame.copy()
            frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2RGB)
        # if self.resize_dim and frame.shape[:2] != self.resize_dim:
        #     frame = cv2.resize(frame, self.resize_dim)
        #     gray_frame = cv2.resize(gray_frame, self.resize_dim)

        self.logger.info(f'start Aruco marker detection for image size: {frame.shape}')
        marker_corners, ids, rejected = cv2.aruco.detectMarkers(gray_frame, self.arucoDict, parameters=self.arucoParams)
        mtx, dist = self.calibrator.calib_params['mtx'], self.calibrator.calib_params['dist']
        rVec, tVec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners, config.ARUCO_MARKER_SIZE, mtx, dist)

        missing_ids = []
        for marker_id in ARUCO_IDS:
            if marker_id not in ids:
                missing_ids.append(marker_id)
                continue

            i = ids.astype(int).ravel().tolist().index(marker_id)
            corners = marker_corners[i]
            top_right, top_left, bottom_right, bottom_left = self.flatten_corners(corners)
            row, col = marker_id // CHARUCO_COLS, marker_id % CHARUCO_COLS
            self.markers[marker_id] = {
                'x': round((2*col + 1 if row % 2 else 2*col) * config.ARUCO_MARKER_SIZE),
                'y': round(row * config.ARUCO_MARKER_SIZE),
                # 'x': round(tVec[i][0][0], 1),
                # 'y': round(tVec[i][0][1], 1),
                'cam_distance': np.sqrt(tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2),
                'corners': corners,
                'top_right': top_right,
                'bottom_right': bottom_right,
                'bottom_left': bottom_left,
                'top_left': top_left,
                'center': ((top_right[0] + top_left[0]) // 2, (top_right[1] + bottom_right[1]) // 2),
                'rVec': rVec[i],
                'tVec': tVec[i]
            }
        self.markers[self.id_key] = utils.datetime_string()
        self.logger.warning(f'The following markers were not detected: {missing_ids}')
        # self.align_axes()
        # self.align_coords_to_center_marker()
        self.save_markers()
        if is_plot:
            frame = self.plot_aruco_detections(frame)
            plt.imshow(frame)
            plt.show()
        return frame, True

    def align_axes(self):
        """align the real-world coordinates"""
        assert any(x in self.markers for x in TEST_IDS)
        for tid in TEST_IDS:
            if tid in self.markers:
                break
        x_real, y_real = self.get_location(*self.markers[tid]['center'], check_init=False)
        if x_real < 0:
            self.sign_mask[0] = -1
        if y_real < 0:
            self.sign_mask[1] = -1

    def align_coords_to_center_marker(self):
        if CENTER_ID not in self.markers:
            return

        x_center, y_center = self.markers[CENTER_ID]['x'], self.markers[CENTER_ID]['y']
        for marker_id, d in self.markers.copy().items():
            if marker_id == self.id_key:
                continue
            self.markers[marker_id]['x'] = d['x'] - x_center
            self.markers[marker_id]['y'] = d['y'] - y_center

    def plot_aruco_detections(self, frame):
        font, line_type, font_size = cv2.FONT_HERSHEY_PLAIN, cv2.LINE_AA, 1.8
        for marker_id, d in self.markers.items():
            if marker_id == self.id_key:
                continue
            cv2.polylines(frame, [d['corners'].astype(np.int32)], True, (0, 255, 255), 4, line_type)
            # Draw the pose of the marker
            # mtx, dist = self.calibrator.calib_params['mtx'], self.calibrator.calib_params['dist']
            # cv2.drawFrameAxes(frame, mtx, dist, d['rVec'], d['tVec'], 4, 4)
            if d['y'] == 0 or d['x'] == 0:
                x_, y_ = d['top_left']
                cv2.circle(frame, (x_, y_), 2, (255, 0, 255), 3)
                if d['y'] == 0:  # screen axis
                    pos = (x_ - self.sign_mask[0] * 50, y_) if self.is_swap_xy else (x_, y_ - self.sign_mask[1] * 50)
                    cv2.putText(frame, f"{d['x']}", pos, font, font_size, (255, 0, 255), 2, line_type)
                if d['x'] == 0:
                    pos = (x_, y_ - self.sign_mask[1] * 50) if self.is_swap_xy else (x_ - self.sign_mask[0] * 50, y_)
                    cv2.putText(frame, f"{d['y']}", pos, font, font_size, (255, 0, 255), 2, line_type)
        frame = self.plot_calibrated_line(frame)
        return frame

    def plot_calibrated_line(self, frame, color=(218, 165, 32)):
        font, line_type, font_size = cv2.FONT_HERSHEY_PLAIN, cv2.LINE_AA, 1.8
        # plot center image coord for checking the get_location algorithm
        h, w = frame.shape[:2]
        for frame_pos in [(w // 2, h // 2), (w // 2, h // 3), (w // 2, round(h / 1.3))]:
            xc, yc = self.get_location(*frame_pos, check_init=False)
            cv2.circle(frame, frame_pos, 2, color, 3)
            cv2.putText(frame, f"({xc:.1f}, {yc:.1f})", frame_pos, font, font_size, color, 3, line_type)
        return frame

    def load_markers(self):
        if not self.cached_markers_path.exists():
            raise Exception(f'Aruco cache file {self.cached_markers_path} does not exist')

        with self.cached_markers_path.open('rb') as f:
            markers = pickle.load(f)
            missing_markers = []
            for marker_id in ARUCO_IDS + [self.id_key]:
                marker_dict = markers.get(marker_id, {})
                if not marker_dict and marker_id != self.id_key:
                    missing_markers.append(marker_id)
                else:
                    self.markers[marker_id] = marker_dict
            if self.is_debug:
                self.logger.info(f'Loaded {len(self.markers)} markers from {self.cached_markers_path}')
            # self.logger.warning(f'The following markers are missing: {missing_markers}')

    def save_markers(self):
        with self.cached_markers_path.open('wb') as f:
            pickle.dump(self.markers, f)
            self.logger.info(f'Markers saved to {self.cached_markers_path}')

    def flatten_corners(self, corners):
        corners = corners.reshape(4, 2).astype(int)
        top_left = corners[0].ravel()
        top_right = corners[1].ravel()
        bottom_right = corners[2].ravel()
        bottom_left = corners[3].ravel()
        return top_right, top_left, bottom_right, bottom_left

    @property
    def is_initiated(self):
        return self.state > 0

    @property
    def is_on(self):
        return self.state == 2

    @property
    def cached_markers_path(self):
        return Path(config.calibration_dir) / f'markers_{self.cam_name}_{json.dumps(self.resize_dim)}.pkl'


def main(img, cam='top'):
    pe = CharucoEstimator(cam)
    img, ret = pe.find_aruco_markers(img)


if __name__ == "__main__":
    # cam_, image_ = 'top', cv2.imread('/data/Pogona_Pursuit/output/captures/20230221T131003_top.png')
    cam_, image_ = 'front', cv2.imread('/data/Pogona_Pursuit/output/captures/20230221T131133_front.png')
    main(image_, cam=cam_)