import numpy as np
import cv2 as cv
import glob
import json
import os

# Undistortion code from:
#   https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

# Undistort matrices for the Flir camera.
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

# Homography constants
ARENA_H_CM = 53
ARENA_W_CM = 34.5

ARUCO_BOTTOM_RIGHT = 0
ARUCO_BOTTOM_LEFT = 1
ARUCO_TOP_LEFT = 2
ARUCO_TOP_RIGHT = 3

ARUCO_CORNER = 0

ARUCO_DICT = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)

HOMOGRAPHIES_FOLDER = 'calibration'

class CalibrationException(Exception):
    pass


# TODO: possible to use aruco.detectMarkers + aruco.estimatePoseSingleMarkers
# but also needs multiple viewpoints.
def get_distortion_matrix(chkr_im_path, rows=6, cols=9):
    """
    Finds the undistortion matrix of the lense based on multiple images
    with checkerboard.
    Returns 2 matrices: mtx, dist

    chkr_im_path - path to folder with images with checkerboards
    rows - number of rows in checkerboard
    cols - number of cols in checkerboard
    """

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(chkr_im_path)

    # drawings = []
    # imgs = []
    for fname in images:
        img = cv.imread(fname)
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
    Undistort mapping for the given mtx and dist matrices.
    Returns (mapx, mapy), roi, newcameramtx
    mapx, mapy - x,y coordinates for each image coordinate for undistorting an image.
    roi - (x, y, w, h) region of interest tuple
    newcameramtx - new camera matrix for undistorting points.
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
    Return an undistorted version of img according to the mapping.
    When roi is not None the image is cropped to the ROI.
    :param img: image to undistort
    :param mapping: a tuple (mapx, mapy)
    """
    mapx, mapy = mapping

    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    if roi:
        x, y, w, h = roi
        dst = dst[y: y + h, x: x + w]

    return dst


def undistort_point(p, newcameramtx, dist=DIST):
    """
    Undistort point p.
    newcameramtx - the matrix returned by get_undistort_mapping
    """
    x, y = p.astype(int)
    if np.isnan(x):
        return np.nan, np.nan

    return cv.undisortPoints([[[x, y]]], newcameramtx, dist)


def undistort_data(
        data,
        width,
        height,
        cols=(("cent_x", "cent_y"), ("x1", "y1"), ("x2", "y2")),
        mtx=MTX,
        dist=DIST,
        alpha=0
):
    """
    Undistorts a bulk of data. assumes location data in (cent_x, cent_y, x1, y1, x2, y2) format
    TODO possible to make more generic
    :param data: pandas DataFrame
    :param mapping: (mapx, mapy) tuple returned by gen_undistort_mapping.
    :param cols: an iterable of iterables, each nested iterable is a pair of column names to undistort.
    for example [(cent_x, cent_y), (x, y)]
    :return: dataframe with the same columns (returns copy, doesn't change inplace)

    for each pair of (x,y) points to undistort, create new dataframe with corrected data,
    and assign to the returned dataframe
    """

    ret_df = data.copy()

    _, _, newcameramtx = get_undistort_mapping(
        width, height, mtx=mtx, dist=dist, alpha=alpha
    )

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
# code taken from:
#    https://pysource.com/2018/09/25/simple-shape-detection-opencv-with-python-3/


# TODO: old black squares function
def get_points(polygons, max_y=True):
    """
    Return the rightmost and leftmost upper points (closest to the screen) of two squares.
    """

    if any((polygons[0] - polygons[1])[:, 0] < 0):
        right, left = polygons[1], polygons[0]
    else:
        right, left = polygons[0], polygons[1]

    min_xs = np.argsort(right[:, 0])
    if max_y:
        p_right_ind = np.argmax(right[min_xs][:2, 1])
    else:
        p_right_ind = np.argmin(right[min_xs][:2, 1])

    max_xs = np.argsort(left[:, 0])[::-1]
    if max_y:
        p_left_ind = np.argmax(left[max_xs][:2, 1])
    else:
        p_left_ind = np.argmin(left[max_xs][:2, 1])

    return right[min_xs][p_right_ind], left[max_xs][p_left_ind]

# TODO: old black squares function
def thresh_dist(poly, min_thresh, max_thresh):
    """
    Return True if the distance between each pair of points is larger than
    min_thresh and smaller than max_thresh.
    """
    for i, p1 in enumerate(poly):
        for j, p2 in enumerate(poly[i + 1:]):
            norm = np.linalg.norm(p1 - p2)
            if norm < min_thresh or norm > max_thresh:
                return False
    return True


# TODO: old black squares function
def polygons_min_distance(polygons, min_dist=300):
    """
    Checks if the polygons centers are mutually far away from another, in case some reflections
    are detected by mistake.
    :param polygons: a list of polygons, numpy arrays each with 4 edges
    :param min_dist: minimal L2 distance between polygons
    :return: True if polygons are too close, else False
    """

    centroids = np.empty((len(polygons), 2))
    for i, poly in enumerate(polygons):
        centroids[i] = poly.mean(axis=0)

    for i, cent in enumerate(centroids):
        for j, cent2 in enumerate(centroids[i + 1:]):
            dist = np.linalg.norm(cent - cent2)
            if dist < min_dist:
                return True
    return False

# TODO: Old function, operates on black squares
def find_arena_homography_black_squares(
        cal_img,
        screen_x_res=1920,
        contrast=2.4,
        brightness=0,
        min_near_edge_size=30,
        min_far_edge_size=10,
        max_near_edge_size=100,
        max_far_edge_size=100,
        near_far_y_split=700,
        min_dist=300,
):
    """
    Calculate the homography matrix to map from camera coordinates to
    a coordinate system relative to the touch screen.

    Assumes cal_img contains 4 visually clear black squares marking the screen edges
    and the rear end of the arena.
    Assumes the image is corrected for lense distortion.
    Finds the innermost right and left points that are closest to the screen and
    returns the transformation and an image with the features highlighted.
    :return: homography H, labelled image, screen length in image pixels
    """

    img = cv.cvtColor(cal_img.copy(), cv.COLOR_BGR2GRAY)
    img = cv.convertScaleAbs(img, alpha=contrast, beta=brightness)

    _, threshold = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    near_polygons = []
    far_polygons = []

    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
        if len(approx) != 4:
            continue

        approx = approx.squeeze()

        if all(approx[:, 1] > near_far_y_split):
            if thresh_dist(approx, min_near_edge_size, max_near_edge_size):
                near_polygons.append(approx)
                cv.drawContours(img, [approx], 0, (255, 255, 0), 5)
            else:
                cv.drawContours(img, [approx], 0, (255, 0, 0), 5)
        else:
            if thresh_dist(approx, min_far_edge_size, max_far_edge_size):
                far_polygons.append(approx)
                cv.drawContours(img, [approx], 0, (0, 0, 255), 5)
            else:
                cv.drawContours(img, [approx], 0, (255, 0, 0), 5)

    if len(near_polygons) != 2 or len(far_polygons) != 2:
        return None, img, "Could not find 2 far and 2 near square marks in the image."

    if polygons_min_distance(near_polygons, min_dist):
        return None, img, "Some of the near polygons are too close to each other."

    if polygons_min_distance(far_polygons, min_dist):
        return None, img, "Some of the far polygons are too close to each other."

    p_bottom_r, p_bottom_l = get_points(near_polygons)
    p_top_r, p_top_l = get_points(far_polygons, max_y=False)

    # Draw the screen line.
    cv.line(img, pt1=tuple(p_bottom_r), pt2=tuple(p_bottom_l), color=(0, 255, 0), thickness=10)
    cv.line(img, pt1=tuple(p_bottom_r), pt2=tuple(p_top_r), color=(0, 255, 0), thickness=10)
    cv.line(img, pt1=tuple(p_bottom_l), pt2=tuple(p_top_l), color=(0, 255, 0), thickness=10)
    cv.line(img, pt1=tuple(p_top_r), pt2=tuple(p_top_l), color=(0, 255, 0), thickness=10)

    arena_h_pixels = screen_x_res * (ARENA_H_CM / ARENC_W_CM)
    dst_p = np.array([[0, 0],
                      [screen_x_res, 0],
                      [0, arena_h_pixels],
                      [screen_x_res, arena_h_pixels]])
    src_p = np.vstack([p_bottom_r, p_bottom_l, p_top_r, p_top_l]).astype(np.float64)

    homography, _ = cv.findHomography(src_p, dst_p)

    return homography, img, None


# Aruco calibration
# https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/aruco_basics.html
# https://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html
def get_point_aruco(ids_array, corners_list, aruco_dict_index):
    """
    :param ids_array: 2d (unsqueezed) array of indices for Aruco markers
    :param corners_list: list of 3d (unsqueezed) numpy 4X2 arrays, corners of each marker
    :param aruco_dict_index: real world corner to retrieve
    :return: a (2,) numpy array, the required point from the marker
    """
    ids_array = ids_array.squeeze()
    corner_ind = np.where(ids_array == aruco_dict_index)[0][0]
    corner = corners_list[corner_ind].squeeze()
    return corner[ARUCO_CORNER]


def find_arena_homography(
        cal_img,
        screen_x_res=1920,
        aruco_dict=ARUCO_DICT):
    """
    Calculate the homography matrix to map from camera coordinates to
    a coordinate system relative to the touch screen.

    Assumes cal_img contains 4 visually clear Aruco patterns, which are the first 4 indices in the pattern list
    from the predefined Aruco_dict_4X4_50. Each pattern has it's original top left corner in a specified place
    in the arena to create a rectangular shape.
    :param cal_img: Image to extract the homography from. Numpy image, assumed to be lense corrected
    :param screen_x_res:
    :param aruco_dict: Aruco dictionary which contains the patterns. Default: aruco_dict_4x4_50, using ascending order
    :return: homography H, labelled image, screen length in image pixels
    """

    gray = cv.cvtColor(cal_img, cv.COLOR_BGR2GRAY)
    parameters = cv.aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = cv.aruco.detectMarkers(gray,
                                                               aruco_dict,
                                                               parameters=parameters)
    frame_markers = cv.aruco.drawDetectedMarkers(cal_img.copy(), corners, ids)

    # TODO what if 4 found but one is false positive?
    if len(corners) != 4:
        return None, frame_markers, "Could not find 4 Aruco patterns in the image."

    p_bottom_r = get_point_aruco(ids, corners, ARUCO_BOTTOM_RIGHT)
    p_bottom_l = get_point_aruco(ids, corners, ARUCO_BOTTOM_LEFT)
    p_top_r = get_point_aruco(ids, corners, ARUCO_TOP_RIGHT)
    p_top_l = get_point_aruco(ids, corners, ARUCO_TOP_LEFT)

    # Draw the rectangle formed by the corners
    thickness = 5
    color = (0, 255, 0) # green
    cv.line(frame_markers, pt1=tuple(p_bottom_r), pt2=tuple(p_bottom_l), color=color, thickness=thickness)
    cv.line(frame_markers, pt1=tuple(p_bottom_r), pt2=tuple(p_top_r), color=color, thickness=thickness)
    cv.line(frame_markers, pt1=tuple(p_bottom_l), pt2=tuple(p_top_l), color=color, thickness=thickness)
    cv.line(frame_markers, pt1=tuple(p_top_r), pt2=tuple(p_top_l), color=color, thickness=thickness)

    arena_h_pixels = screen_x_res * (ARENA_H_CM / ARENC_W_CM)
    dst_p = np.array([[0, 0],
                      [screen_x_res, 0],
                      [0, arena_h_pixels],
                      [screen_x_res, arena_h_pixels]])
    src_p = np.vstack([p_bottom_r, p_bottom_l, p_top_r, p_top_l]).astype(np.float64)

    homography, _ = cv.findHomography(src_p, dst_p)

    return homography, frame_markers, None


def transform_point(p, h):
    """
    :param p: point, numpy array with shape (2,)
    :param h: homography matrix
    :return: transformed point
    """
    return cv.perspectiveTransform(p.reshape(-1, 1, 2), h).squeeze()


def transform_image(img, h, screen_x_res=1920):
    """
    Applies homography transformation and crop to image
    :param img: image to transform
    :param h: homograhy matrix, 3X3 numpy array
    :param screen_x_res: resolution of the screen in the arena
    :param screen_width: size of the screen in the image, as returned from calibrate function
    :return:image in the new coordinate space
    """
    arena_h_pixels = screen_x_res * (ARENA_H_CM / ARENC_W_CM)
    img_shape = (screen_x_res, int(arena_h_pixels))
    return cv.warpPerspective(img, h, img_shape)


def transform_data(data, h, cols=(("cent_x", "cent_y"), ("x1", "y1"), ("x2", "y2"))):
    """
    Computes the transformed coordinates of undistorted 2D data by matrix multiplication
    :param data: pandas DataFrame, columns assume to contain ('cent_x', 'cent_y', 'x', 'y') and optionally 'w' and 'h'
    :param h: homography matrix, 3X3 numpy array
    :param cols: pairs of columns in the dataframe to transform
    :return: dataframe with the transformed data in the
    """
    ret_df = data.copy()

    for xy in cols:
        x = xy[0]
        y = xy[1]

        ret_df[[x, y]] = cv.perspectiveTransform(data[[x, y]].values.reshape(-1, 1, 2), h).squeeze()

    return ret_df


def get_last_homography(homographies_folder=HOMOGRAPHIES_FOLDER):
    """
    Get the latest homography from the homographies folder according to date
    :param homographies_folder: path to the homographies files
    :return: numpy 3x3 homography matrix, source width and height (ints)
    """
    homographies_files = glob.glob(os.path.join(homographies_folder, 'homog_*'))

    if len(homographies_files) == 0:
        #  No homographies found
        return None, None, None

    # TODO: maybe extract date objects and sort by date (maybe easier with pandas)
    sorted_homographies_files = sorted(homographies_files)
    last_homogprahy_file = sorted_homographies_files[-1]

    with open(last_homogprahy_file, 'r') as fp:
        homog_dict = json.load(fp)

    homography = np.array(homog_dict['homography'])
    width = homog_dict['width']
    height = homog_dict['height']

    return homography, width, height
