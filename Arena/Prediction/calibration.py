"""
This module is responsible for geometric transformations for correcting the distortion of the lens and transforming
the data into a unified coordinate system, using existing functions from Open-CV.
The functions from this are called by the HitPredictor object or offline by functions from the Dataset module.

The first part of the module deals with correcting the lens distortion. As the camera is located close to the target,
and captures a relatively wide angle frame, the barrel distortion is significant. The function get_distortion_matrix
analyzes images that contain a checkerboard, and produces the required camera matrix. This parameter is static and
does not depend on the camera location, but might change something changes in the lens.
The function get_undistort_mapping computes from these parameters the required transformations, to be used upon frames,
single points of data, or data arrays.

The second part deals with transforming the data into a single coordinate system, in which the the screen lies
approximately on the X axis. This part is more complex, as it depends on the precise location and angle of the camera
with respect the arena. For that purpose, there are 4 Aruco markers located in the arena in known distance from each
other. A function finds the Aruco markers in the arena, and computes the required homography transformation. Other
functions use this transformation upon frames or coordinates data.

"""

import numpy as np
import cv2 as cv
import json
from datetime import datetime
from pathlib import Path

# Undistortion code from:
#   https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

# Static un-distortion matrices for the Flir camera.
MTX = np.array(
    [
        [1.14515564e03, 0.00000000e00, 7.09060713e02],
        [0.00000000e00, 1.14481967e03, 5.28220061e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

DIST = np.array(
    [
        [
            -4.25580120e-01,
            3.02361751e-01,
            -1.56952670e-03,
            -4.04385846e-04,
            -2.27525587e-01,
        ]
    ]
)

# Homography and arena constants
ARENA_H_CM = 53
ARENA_W_CM = 34.5

ARUCO_BOTTOM_RIGHT = 0
ARUCO_BOTTOM_LEFT = 1
ARUCO_TOP_LEFT = 2
ARUCO_TOP_RIGHT = 3

ARUCO_CORNER = 0

ARUCO_DICT = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)

HOMOGRAPHIES_DIR = Path("calibration")


class CalibrationException(Exception):
    pass


def get_distortion_matrix(chkr_im_path: Path, rows=6, cols=9):
    """
    Finds the undistortion matrix of the lens based on multiple images
    with checkerboard. It's possible to implement this function using Aruco markers as well.

    :param: chkr_im_path - path to folder with images with checkerboards
    :param: rows - number of rows in checkerboard
    :param: cols - number of cols in checkerboard
    :return: numpy array: camera matrix, numpy array: distortion coefficients
    """

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    image_paths = list(chkr_im_path.iterdir())

    # drawings = []
    # imgs = []
    for fname in image_paths:
        img = cv.imread(str(fname))
        shape = img.shape
        # imgs.append(img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (cols, rows), None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv.drawChessboardCorners(img, (cols,rows), corners2,ret)
            # drawings.append(img)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, shape[::-1][1:], None, None
    )

    if not ret:
        raise CalibrationException("Error finding distortion matrix")

    return mtx, dist


def get_undistort_mapping(width, height, mtx=MTX, dist=DIST, alpha=0):
    """
    Computes the undistortion mapping for the given mtx and dist matrices.

    :param width: int, width of the image
    :param height: int, height of the image
    :param mtx: numpy array, camera matrix
    :param dist: numpy array, distortion coefficients
    :param alpha: float in the range (0,1)
    :return:
        mapx, mapy - numpy arrays, x,y coordinates for each image coordinate for undistorting an image.
        roi - tuple, (x, y, w, h) region of interest tuple
        newcameramtx - numpy array,  camera matrix for undistorting points in the specific (width, height) frame.
    """
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, dist, (width, height), alpha, (width, height)
    )
    return (
        cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width, height), 5),
        roi,
        newcameramtx,
    )


def undistort_image(img, mapping, roi=None):
    """
    When roi is not None the image is cropped to the ROI.
    :param img: numpy array: image to undistort
    :param mapping: a tuple (mapx, mapy)
    :return: numpy array: undistorted version of img according to the mapping
    """
    mapx, mapy = mapping

    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    if roi:
        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]

    return dst


def undistort_point(p, newcameramtx, dist=DIST):
    """
    :param p: iterable, coordinate to undistort
    :param newcameramtx: numpy array, camera matrix for the specific (width, height) frame.
    :param dist: numpy array, distortion coefficients
    :return: numpy array, undistorted points
    """

    p = np.array(p)
    if np.any(np.isnan(p)):
        return np.nan, np.nan

    return cv.undistortPoints(np.expand_dims(p, axis=0), newcameramtx, dist).squeeze()


def undistort_data(
    data, width, height, cols=(("x1", "y1"), ("x2", "y2")), mtx=MTX, dist=DIST, alpha=0,
):
    """
    Undistorts a bulk of data. assumes location data in (cent_x, cent_y, x1, y1, x2, y2) format
    :param data: Pandas DataFrame, data to undistort
    :param width: int
    :param height: int
    :param cols: tuple of pairs of strings which are column names in the df
    :param mtx: numpy array, camera matrix
    :param dist: numpy array,
    :param alpha: float in (0,1)
    :return: pandas df, the undistorted data (deep copy of the original data)
    """

    # deep copy of the original data
    ret_df = data.copy()

    # get transformation for the specific frame
    _, _, newcameramtx = get_undistort_mapping(
        width, height, mtx=mtx, dist=dist, alpha=alpha
    )

    # for each pair of columns which constitute a coordinate, undistort
    for xy in cols:
        x = xy[0]
        y = xy[1]
        points = data[[x, y]].values
        undistorted = cv.undistortPoints(
            np.expand_dims(points, axis=0), newcameramtx, dist, P=newcameramtx
        )
        ret_df[[x, y]] = undistorted.squeeze(1)

    return ret_df


# Calibration: Use a calibration image to find the touch screen transition transform.

# Aruco calibration tutorial and docs
# https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/aruco_basics.html
# https://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html


def get_point_aruco(ids_array, corners_list, aruco_dict_index):
    """
    :param ids_array: numpy 2d (unsqueezed) array of indices for Aruco markers
    :param corners_list: list of 3d (unsqueezed) numpy 4X2 arrays, corners of each marker
    :param aruco_dict_index: real world corner to retrieve
    :return: a (2,) numpy array, the required point from the marker
    """
    ids_array = ids_array.squeeze()
    corner_ind = np.where(ids_array == aruco_dict_index)[0][0]
    corner = corners_list[corner_ind].squeeze()
    return corner[ARUCO_CORNER]


def get_homography_from_points(points, screen_x_res=1920):
    """
    Calculates the arena homography matrix from a list of 4 points.

    :param points: A list of 4 points - [bottom-right, bottom-left, top-right, top-left]
    :param screen_x_res: Width of the screen in pixels.

    :return: Arena homography matrix.
    """
    arena_h_pixels = screen_x_res * (ARENA_H_CM / ARENA_W_CM)
    dst_p = np.array(
        [[0, 0], [screen_x_res, 0], [0, arena_h_pixels], [screen_x_res, arena_h_pixels]]
    )
    src_p = np.vstack(points).astype(np.float64)

    homography, _ = cv.findHomography(src_p, dst_p)
    return homography


def find_arena_homography(cal_img, screen_x_res=1920, aruco_dict=ARUCO_DICT):
    """
    Calculate the homography matrix to map from camera coordinates to a coordinate system relative to the touch screen.
    Assumes cal_img contains 4 visually clear Aruco patterns, which are the first 4 indices in the pattern list.
    Each pattern has it's original top left corner in a specified place in the arena to create a rectangular shape.

    :param cal_img: Numpy image to extract the homography from. Assumed to be lens corrected
    :param screen_x_res: int, horizontal resolution of the screen
    :param aruco_dict: Aruco dictionary which contains the patterns. Default: aruco_dict_4x4_50, using ascending order
    :return: numpy array homography H, labelled image, screen length in image pixels
    """

    gray = cv.cvtColor(cal_img, cv.COLOR_BGR2GRAY)
    parameters = cv.aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = cv.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )
    frame_markers = cv.aruco.drawDetectedMarkers(cal_img.copy(), corners, ids)

    # if less (or strictly more) than 4 Aruco markers found, return error
    if len(corners) != 4:
        return (
            None,
            None,
            frame_markers,
            "Could not find 4 Aruco patterns in the image.",
        )

    p_bottom_r = get_point_aruco(ids, corners, ARUCO_BOTTOM_RIGHT)
    p_bottom_l = get_point_aruco(ids, corners, ARUCO_BOTTOM_LEFT)
    p_top_r = get_point_aruco(ids, corners, ARUCO_TOP_RIGHT)
    p_top_l = get_point_aruco(ids, corners, ARUCO_TOP_LEFT)

    # Draw the rectangle formed by the corners
    thickness = 5
    color = (0, 255, 0)  # green
    cv.line(
        frame_markers,
        pt1=tuple(p_bottom_r),
        pt2=tuple(p_bottom_l),
        color=color,
        thickness=thickness,
    )
    cv.line(
        frame_markers,
        pt1=tuple(p_bottom_r),
        pt2=tuple(p_top_r),
        color=color,
        thickness=thickness,
    )
    cv.line(
        frame_markers,
        pt1=tuple(p_bottom_l),
        pt2=tuple(p_top_l),
        color=color,
        thickness=thickness,
    )
    cv.line(
        frame_markers,
        pt1=tuple(p_top_r),
        pt2=tuple(p_top_l),
        color=color,
        thickness=thickness,
    )

    points = [p_bottom_r, p_bottom_l, p_top_r, p_top_l]

    homography = get_homography_from_points(points, screen_x_res)

    return homography, points, frame_markers, None


def transform_point(p, h):
    """
    :param p: point, numpy array with shape (2,)
    :param h: homography matrix, numpy array
    :return: transformed point, numpy array
    """
    return cv.perspectiveTransform(p.reshape(-1, 1, 2), h).squeeze()


def transform_image(img, h, screen_x_res=1920):
    """
    Applies homography transformation and crop to image

    :param img: image to transform
    :param h: homograhy matrix, 3X3 numpy array
    :param screen_x_res: int, resolution of the screen in the arena
    :param screen_width: int, size of the screen in the image, as returned from calibrate function
    :return: numbpy array image in the new coordinate space
    """
    arena_h_pixels = screen_x_res * (ARENA_H_CM / ARENA_W_CM)
    img_shape = (screen_x_res, int(arena_h_pixels))
    return cv.warpPerspective(img, h, img_shape)


def transform_data(data, h, cols=(("x1", "y1"), ("x2", "y2"))):
    """
    Computes the transformed coordinates of undistorted 2D data by matrix multiplication

    :param data: pandas DataFrame, columns assume to contain ('cent_x', 'cent_y', 'x', 'y') and optionally 'w' and 'h'
    :param h: homography matrix, 3X3 numpy array
    :param cols: pairs of columns in the dataframe to transform
    :return: pandas dataframe with the transformed data in the
    """
    ret_df = data.copy()

    for xy in cols:
        x = xy[0]
        y = xy[1]

        ret_df[[x, y]] = cv.perspectiveTransform(
            data[[x, y]].values.reshape(-1, 1, 2), h
        ).squeeze()

        # copy NaN values to the new dataframe
        nan_rows = np.isnan(data[x]) | np.isnan(data[y])
        ret_df.loc[nan_rows, [x, y]] = [np.nan, np.nan]

    return ret_df


def get_last_homography(homographies_dir: Path = HOMOGRAPHIES_DIR):
    """
    Get the latest homography from the homographies folder according to date. Called by the HitPredictor

    :param homographies_folder: path to the homographies files
    :return: numpy 3x3 homography matrix, source width and height (ints)
    """
    homographies_files = list(homographies_dir.glob("homog_*.json"))

    if len(homographies_files) == 0:
        #  No homographies found
        return None, None, None

    # TODO: maybe extract date objects and sort by date (maybe easier with pandas)
    sorted_homographies_files = sorted(homographies_files)
    last_homography_file = sorted_homographies_files[-1]

    with open(last_homography_file, "r") as fp:
        homog_dict = json.load(fp)

    homography = np.array(homog_dict["homography"])
    width = homog_dict["width"]
    height = homog_dict["height"]

    return homography, width, height


def save_homography(
    h, points, cal_img, date, cam_width, cam_height, homographies_dir=HOMOGRAPHIES_DIR
):
    date = date.strftime("%Y%m%d-%H%M%S")
    json_path = homographies_dir / ("homog_" + date + ".json")
    img_path = homographies_dir / ("homog_" + date + ".jpg")

    d = {
        "homography": h.tolist(),
        "width": cam_width,
        "height": cam_height,
        "points": list(map(lambda p: p.tolist(), points)),
    }

    with open(json_path, "w") as fp:
        json.dump(d, fp)

    cv.imwrite(str(img_path), cal_img)


def calibrate(cal_img):
    cam_width, cam_height = cal_img.shape[1], cal_img.shape[0]
    mapping, _, _ = get_undistort_mapping(cam_width, cam_height)
    cal_img_ud = undistort_image(cal_img, mapping)
    h, ps, h_im, error = find_arena_homography(cal_img_ud)

    if error is None:
        save_homography(h, ps, cal_img, datetime.now(), cam_width, cam_height)

    return h, ps, h_im, error
