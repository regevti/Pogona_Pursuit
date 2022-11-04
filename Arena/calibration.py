import cv2
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import config
from loggers import get_logger

CHESSBOARD_DIM = (9, 6)
ARUCO_MARKER_SIZE = 3.5  # centimeters
ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_IDS = [0, 1, 2, 3, 4, 5]
CENTER_ID = 0  # the marker that stands for (0, 0) in the new coordinate system
TEST_ID = 5  # this marker should have both positive coordinates, if not happen will induce fix
SHIFTED_CAMS = ['top']  # frames that their x,y need to be swapped
logger = get_logger('calibration')


class CalibrationError(Exception):
    """"""


class Calibrator:
    def __init__(self, cam_name, resize_dim=None):
        self.cam_name = cam_name
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
        logger.info(f'start camera {self.cam_name} calibration')
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((np.prod(CHESSBOARD_DIM), 3), np.float32)
        objp[:, :2] = np.mgrid[:CHESSBOARD_DIM[0], :CHESSBOARD_DIM[1]].T.reshape(-1, 2)
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
            img = cv2.imread(p.as_posix())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.resize_dim:
                img = cv2.resize(img, self.resize_dim)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_DIM, None)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners_ = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners_)
                if is_plot:
                    cv2.drawChessboardCorners(img, CHESSBOARD_DIM, corners_, ret)
                    axes[i].imshow(img)

        h, w = img.shape[:2]
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
        logger.info('calibration finished successfully')

    def undistort_image(self, img) -> np.ndarray:
        return cv2.remap(img, *self.undistort_mappers, cv2.INTER_LINEAR)

    def calc_undistort_mappers(self):
        mtx, dist = self.calib_params['mtx'], self.calib_params['dist']
        w, h = self.calib_params['w'], self.calib_params['h']
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # undistort
        self.undistort_mappers = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

    @staticmethod
    def calc_projection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        logger.info("total projection error: {}".format(mean_error / len(objpoints)))

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
        return Path(config.calibration_dir) / f'calib_params_{self.cam_name}_{self.resize_dim}.pkl'

    @property
    def calib_image_detections_path(self):
        return Path(config.calibration_dir) / f'calib_detections_{self.cam_name}.png'


class PoseEstimator:
    def __init__(self, cam_name, resize_dim=None):
        self.cam_name = cam_name
        self.calibrator = Calibrator(cam_name, resize_dim=resize_dim)
        self.arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.is_swap_xy = cam_name in SHIFTED_CAMS
        self.sign_mask = [1, 1]
        self.state = 0  # 0 - not initiated, 1 - failure, 2 - initiated
        self.markers = {}

    def init(self, frame):
        try:
            _, is_initiated = self.find_aruco_markers(frame, is_plot=False)
        except Exception as exc:
            logger.exception(exc)
            self.state = 1
        else:
            self.state = 2 if is_initiated else 1

    def get_location(self, frame_x, frame_y, check_init=True):
        """Convert camera x, y to real-world coordinates"""
        if check_init and not self.is_initiated:
            return

        dists = {marker_id: distance.euclidean((frame_x, frame_y), d['center'])
                 for marker_id, d in self.markers.items()}
        if not dists:
            return
        closest_marker_id = min(dists, key=dists.get)
        d = self.markers[closest_marker_id]
        pixel2cm = ARUCO_MARKER_SIZE / distance.euclidean(d['top_right'], d['bottom_right'])
        xr = d['x'] + pixel2cm * (frame_x - d['center'][0]) - self.markers[CENTER_ID]['x']
        yr = d['y'] + pixel2cm * (frame_y - d['center'][1]) - self.markers[CENTER_ID]['y']
        if self.is_swap_xy:
            xr, yr = yr, xr

        return xr * self.sign_mask[0], yr * self.sign_mask[1]

    def align_axes(self):
        """align the real-world coordinates"""
        x_real, y_real = self.get_location(*self.markers[TEST_ID]['center'], check_init=False)
        if x_real < 0:
            self.sign_mask[0] = -1
        if y_real < 0:
            self.sign_mask[1] = -1

    def find_aruco_markers(self, frame, is_plot=True):
        if frame.shape[2] > 1:
            gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            gray_frame = frame.copy()
            frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2RGB)
        marker_corners, ids, rejected = cv2.aruco.detectMarkers(gray_frame, self.arucoDict, parameters=self.arucoParams)

        if not marker_corners:
            logger.error('No aruco markers were found')
            return frame, False

        if CENTER_ID not in ids:
            logger.error(f'Center marker ID:{CENTER_ID} was not found')
            return frame, False

        mtx, dist = self.calibrator.calib_params['mtx'], self.calibrator.calib_params['dist']
        rVec, tVec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners, ARUCO_MARKER_SIZE, mtx, dist)

        for marker_id in ARUCO_IDS:
            if marker_id not in ids:
                logger.warning(f'aruco marker ID:{marker_id} was not found')
                continue

            i = ids.astype(int).ravel().tolist().index(marker_id)
            corners = marker_corners[i]
            top_right, top_left, bottom_right, _ = self.flatten_corners(corners)
            self.markers[marker_id] = {
                'x': round(tVec[i][0][0], 1),
                'y': round(tVec[i][0][1], 1),
                'cam_distance': np.sqrt(tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2),
                'corners': corners,
                'top_right': top_right,
                'bottom_right': bottom_right,
                'center': ((top_right[0] + top_left[0]) // 2, (top_right[1] + bottom_right[1]) // 2),
                'rVec': rVec[i],
                'tVec': tVec[i]
            }
        self.align_axes()
        if is_plot:
            frame = self.plot_aruco_detections(frame)
        return frame, True

    def align_coords_to_center_marker(self):
        if CENTER_ID not in self.markers:
            return

        x_center, y_center = self.markers[CENTER_ID]['x'], self.markers[CENTER_ID]['y']
        for marker_id, d in self.markers.copy().items():
            self.markers[marker_id]['x'] = d['x'] - x_center
            self.markers[marker_id]['y'] = d['y'] - y_center

    def plot_aruco_detections(self, frame):
        font, line_type, font_size = cv2.FONT_HERSHEY_PLAIN, cv2.LINE_AA, 1.8
        for marker_id, d in self.markers.items():
            cv2.polylines(frame, [d['corners'].astype(np.int32)], True, (0, 255, 255), 4, line_type)
            # Draw the pose of the marker
            mtx, dist = self.calibrator.calib_params['mtx'], self.calibrator.calib_params['dist']
            cv2.drawFrameAxes(frame, mtx, dist, d['rVec'], d['tVec'], 4, 4)
            cv2.putText(frame, f"ID: {marker_id} Dist: {round(d['cam_distance'], 2)}", d['top_right'] - 10,
                        font, font_size, (0, 0, 255), 2, line_type)
            real_x, real_y = self.get_location(*d['center'], check_init=False)
            cv2.putText(frame, f"{(round(real_x), round(real_y))}", d['bottom_right'] + 2,
                        font, font_size, (0, 0, 255), 2, line_type)

        # plot center image coord for checking the get_location algorithm
        h, w = frame.shape[:2]
        frame_pos = (w // 2, h // 2)
        xc, yc = self.get_location(*frame_pos, check_init=False)
        cv2.putText(frame, f"({round(xc)}, {round(yc)})", frame_pos, font, font_size, (0, 0, 255), 2, line_type)
        return frame

    @staticmethod
    def flatten_corners(corners):
        corners = corners.reshape(4, 2).astype(int)
        top_right = corners[0].ravel()
        top_left = corners[1].ravel()
        bottom_right = corners[2].ravel()
        bottom_left = corners[3].ravel()
        return top_right, top_left, bottom_right, bottom_left

    @property
    def is_initiated(self):
        return self.state > 0

    @property
    def is_on(self):
        return self.state == 2
