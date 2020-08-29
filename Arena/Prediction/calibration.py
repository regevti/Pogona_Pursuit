import numpy as np
import pandas as pd
import cv2 as cv
import glob

"""
TODO note:
- https://www.learnopencv.com/homography-examples-using-opencv-python-c/
maybe fix for angle of camera, so the data will be from an overhead (90) angle.
if we move the camera (or at all), may be redundent.
"""

# Undistortion code from:
#   https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

# Undistort matrices for the Flir camera.

MTX = np.array([[1.14515564e+03, 0.00000000e+00, 7.09060713e+02],
                [0.00000000e+00, 1.14481967e+03, 5.28220061e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

DIST = np.array([[-4.25580120e-01, 3.02361751e-01, -1.56952670e-03,
                  -4.04385846e-04, -2.27525587e-01]])


class CalibrationException(Exception):
    pass


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

            corners2 = cv.cornerSubPix(gray, corners, (11, 11),
                                       (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv.drawChessboardCorners(img, (cols,rows), corners2,ret)
            # drawings.append(img)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, shape[::-1][1:], None, None)

    if not ret:
        raise CalibrationException('Error finding distortion matrix')

    return mtx, dist


def get_undistort_mapping(width, height, mtx=MTX, dist=DIST, alpha=0):
    """
    Undistort mapping for the given mtx and dist matrices.
    Returns (mapx, mapy), roi
    mapx - x coordinate for each image coordinate
    mapy - y coordinate for each image coordinate
    roi - (x, y, w, h) region of interest tuple
    """
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist,
                                                     (width, height),
                                                     alpha, (width, height))
    return cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx,
                                      (width, height), 5), roi


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
        dst = dst[y:y + h, x:x + w]

    return dst

"""
TODO  - correct the formula for mapx mapy
"""
def undistort_point(p, mapping):
    """
    Undistort point p.
    Notice: mapx, mapy are 2D arrays with opencv shape, i.e, (dim_y, dim_x)
    mapping - a (mapx, mapy) tuple returned by gen_undistort_mapping.
    """
    mapx, mapy = mapping
    x, y = p
    if np.isnan(x):
        return np.nan, np.nan

    x = int(x)
    y = int(y)  # in case x and y are floats, round down to an integer for indexing
    return mapx[y, x], mapy[y, x]


def undistort_data(data, mapping,
                   cols=(('cent_x', 'cent_y'), ('x1', 'y1'),('x2','y2'))):
    """
    Undistorts a bulk of data. assumes location data in (cent_x, cent_y, x1, y1, x2, y2) format
    TODO possible to make more generic
    :param data: pandas DataFrame
    :param mapping: (mapx, mapy) tuple returned by gen_undistort_mapping.
    :param cols: an iterable of iterables, each nested iterable is a pair of column names to undistort.
    for example [(cent_x, cent_y), (x, y)]
    :return: dataframe with the same columns (returns copy, doesn't change inplace)
    """
    mapx, mapy = mapping
    ret_df = data.copy()

    """
    for each pair of (x,y) points to undistort, create new dataframe with corrected data, 
    and assign to the returned dataframe
    """
    for xy in cols:
        x = xy[0]
        y = xy[1]
        undist = pd.DataFrame(ret_df.apply(lambda r: undistort_point((r[x], r[y]),
                                                                (mapx, mapy)), axis=1).to_list())
        undist.columns = [x, y]
        ret_df[[x, y]] = undist

    return ret_df


# Calibration: Use a calibration image to find the touch screen transition transform.
# code taken from:
#    https://pysource.com/2018/09/25/simple-shape-detection-opencv-with-python-3/


def get_normal(v):
    x, y = v
    n = np.array([-y / x, 1])
    return n / np.linalg.norm(n)


def get_points(polygons):
    """
    Return the rightmost and leftmost upper points (closest to the screen) of two squares.
    """

    if any((polygons[0] - polygons[1])[:, 0] < 0):
        right, left = polygons[1], polygons[0]
    else:
        right, left = polygons[0], polygons[1]

    max_xs = np.argsort(right[:, 0])[::-1]
    p_right_ind = np.argmax(right[max_xs][:2, 1])

    min_xs = np.argsort(left[:, 0])
    p_left_ind = np.argmax(left[min_xs][:2, 1])

    return right[max_xs][p_right_ind], left[min_xs][p_left_ind]


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


def calibrate(cal_img, screen_x_res=1920,
              contrast=2.2, brightness=0,
              min_edge_size=20, max_edge_size=100):
    """
    Calculate the affine transform matrix to go from camera coordinates to
    a coordinate system relative to the touch screen.

    Assumes cal_img contains two visually clear black squares marking the screen edges.
    Finds the rightmost and leftmost points that are closest to the screen and returns
    the transformation and an image with the features highlighted.
    :return: affine trns, labelled image, screen length in image pixels
    """

    img = cv.cvtColor(cal_img.copy(), cv.COLOR_BGR2GRAY)
    img = cv.convertScaleAbs(img, alpha=contrast, beta=brightness)

    _, threshold = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    count = 0
    polygons = []

    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
        if len(approx) != 4 or not thresh_dist(approx, min_edge_size, max_edge_size):
            continue
        count += 1

        # draw the sqaure polygons on the calibration image.
        cv.drawContours(img, [approx], 0, (255, 0, 0), 5)

        polygons.append(approx)

    if count != 2:
        raise CalibrationException('Could not find two square marks in the image.')

    polygons = [poly.squeeze() for poly in polygons]
    p_right, p_left = get_points(polygons)

    # Draw the screen line.
    cv.line(img, pt1=tuple(p_right), pt2=tuple(p_left), color=(0, 255, 0),
            thickness=10)

    v = p_left - p_right
    p_norm = np.linalg.norm(p_left - p_right)
    v = v / np.linalg.norm(v)

    normal = get_normal(v)

    ps_src = np.stack([p_right, p_left, p_right - normal]).astype(np.float32)
    ps_dst = np.float32([[0, 0], [screen_x_res, 0], [0, screen_x_res / p_norm]])
    aff = cv.getAffineTransform(ps_src, ps_dst)

    return aff, img, p_norm


def transform_point(p, aff):
    return np.dot(aff[:, :2], p) + aff[:, 2]


def transform_image(img, aff, screen_x_res, screen_width):
    """

    :param img:
    :param aff:
    :param screen_x_res: resolution of the screen in the arena
    :param screen_width: size of the screen in the image, as returned from calibrate function
    :return:
    """
    y_val = img.shape[0]
    img_shape = (screen_x_res, int(y_val * (screen_x_res / screen_width)))
    return cv.warpAffine(img, aff, img_shape)


def transform_data(data, aff, cols=(('cent_x', 'cent_y'), ('x1','y1'), ('x2','y2'))):
    """
    Comutes the transformed coordinates of undistorted 2D data by matrix multiplication
    :param data: pandas DataFrame, columns assume to contain ('cent_x', 'cent_y', 'x', 'y') and optionally 'w' and 'h'
    :param aff: opencvaffine transformation
    :return: dataframe with the transformed data in the
    """
    ret_df = data.copy()

    for xy in cols:
        x = xy[0]
        y = xy[1]
        data_matrix = np.concatenate([data[[x, y]].values, np.ones((data.shape[0], 1))], axis=1)
        transformed_data = np.dot(data_matrix, aff.T)
        ret_df[[x, y]] = transformed_data

    return ret_df

