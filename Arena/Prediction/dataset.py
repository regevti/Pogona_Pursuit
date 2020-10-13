import glob
import pandas as pd
import numpy as np
import re
import os
import random
import cv2 as cv
import pickle
from tqdm.auto import tqdm
import json
import pickle

from Prediction.detector import nearest_detection, xyxy_to_centroid
import Prediction.calibration as calib

REALTIME_ID = "19506468"

# depending on from where the program is run, add ../..
EXPERIMENTS_ROOT = "../../Pogona_Pursuit/Arena/experiments/"
OUTPUT_ROOT = "../../Pogona_Pursuit/Arena/output/"
CALIBRATIONS_ROOT = "../../Pogona_Pursuit/Arena/calibration"

RT_DATA_FOLDER = "rt_data"
HEAD_CROPS_FN = "head_crops.p"
DETECTIONS_DF_FN = "detections.csv"
DLC_FN = REALTIME_ID + "DLC_resnet50_pogona_pursuitJul19shuffle1_400000.csv"
TIMESTAMPS_FN = "timestamps/" + REALTIME_ID + ".csv"
VID_STATS_FN = "vid_stats.json"
TOUCHES_FN = "screen_touches.csv"
HOMOGRAPHY_IM_FN = "homography.jpg"

EXCLUDE_TERMS = [
    "initial_20200916",
    "delete",
    "fps",
    "vegetables",
    "test-no-streaming",
    "sleepy",
    "predictor",
    "forecasts",
    "feeding3_20200902-122338",  # false positive in homography
    "feeding3_20200902-122611",  # same
    "cockroach_circle_20200902T121652",  # same
    "cockroach_20200907T152020",  # carpet fucks up calibration
    "cockroach_20200907T145302",  # same
    "circle_20200907T145105",  # same
    "feeding3_20200902-142400",  # alpha 0.3 probably caused a corner black hole
    "202007",
    "20200902",
]

OUTPUT_TERMS = ["feeding"]

DF_COLUMNS = ["cent_x", "cent_y", "x1", "y1", "x2", "y2", "conf", "num_bbox"]

# DEFAULT_HOMOGRAPHY_JSON = "homographies.json"
DEFAULT_HOMOGRAPHY_JSON = "Prediction/homographies.json"


""" --------------- Data Analysis Functions --------------- """

"""
TODO - cropped heads are saved before any correction. coordinate transformation might also be relevant
"""


class DatasetException(Exception):
    pass


def get_cropped_head(img, detection, orig_dim):
    """
    Returns the flattened cropped head
    :param img: the resized image that was detector run on last (usually 416X416)
    :param detection: the 5-array (x1,y1,x2,y2,prob) that includes the detection
    :param orig_dim: the dimensions of the original image, before resizing to YOLO dimensions
    :param resize_head: the size of the edge of the square image of the head
    :return: the cropped resized flattened image array (copy)
    """
    src_x1, src_y1 = detection[0], detection[1]
    src_x2, src_y2 = detection[2], detection[3]

    img_x = img.shape[1]
    img_y = img.shape[0]
    orig_x = orig_dim[1]
    orig_y = orig_dim[0]  # open_cv shape order - (Y,X) not (X,Y)

    x_scale = img_x / orig_x
    y_scale = img_y / orig_y

    dst_x1, dst_y1 = round(src_x1 * x_scale), round(src_y1 * y_scale)
    dst_x2, dst_y2 = round(src_x2 * x_scale), round(src_y2 * y_scale)

    dst_x1, dst_y1 = int(dst_x1), int(dst_y1)
    dst_x2, dst_y2 = int(dst_x2), int(dst_y2)

    dst_x1, dst_x2 = max(0, dst_x1), max(0, dst_x2)
    dst_y1, dst_y2 = max(0, dst_y1), max(0, dst_y2)

    dst_x1, dst_x2 = min(img_x, dst_x1), min(img_x, dst_x2)
    dst_y1, dst_y2 = min(img_y, dst_y1), min(img_y, dst_y2)

    cropped_head = img[
        dst_y1:dst_y2, dst_x1:dst_x2
    ].copy()  # open_cv shape order - (Y,X) not (X,Y)

    cropped_head = cv.cvtColor(cropped_head, cv.COLOR_BGR2GRAY)
    cropped_head = 255 - cropped_head

    return cropped_head


# Old function, works with black squares and not Aruco markers.
def get_homography_from_video_black_squares(
    video_path, undist_alpha=0, max_test_frames=50, **homography_args
):
    """
    An offline function to get the homography transformation from a video. Grid searching values of brightness and
    contrast until the correct number of polygons is found
    :param video_path: path to video
    :param undist_alpha: alpha paramater to the lense undistortion function
    :param max_test_frames: maximal number of random frames to try
    :param homography_args: arguments for find_arena_homography function
    :return: homography, or None if no homography found
    """
    vcap = cv.VideoCapture(video_path)
    width = int(vcap.get(3))
    height = int(vcap.get(4))
    num_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT))

    (mapx, mapy), roi, _ = calib.get_undistort_mapping(
        width, height, calib.MTX, calib.DIST, alpha=undist_alpha
    )

    error = None

    # Try random video frames until one works.
    for _ in range(max_test_frames):
        ret, frame = vcap.read()
        if not ret:
            raise DatasetException("Error reading frame.")

        undistorted_img = calib.undistort_image(frame, (mapx, mapy))

        # Try a few contrast and brightness values until one works.
        for contrast in np.arange(0, 3.1, 0.1):
            for brightness in np.arange(-40, 20, 5):
                (
                    homography,
                    homography_im,
                    error,
                ) = calib.find_arena_homography_black_squares(
                    undistorted_img,
                    contrast=contrast,
                    brightness=brightness,
                    **homography_args,
                )
                if homography is not None:
                    print(
                        f"Found homography with brightness: {brightness}, contrast: {contrast}"
                    )
                    break

            if homography is not None:
                break

        if homography is not None:
            break
        else:
            random_frame = random.randint(0, num_frames)
            vcap.set(cv.CAP_PROP_POS_FRAMES, random_frame)

    if error is not None:
        print("Could not find homography:", error)

    vcap.release()

    return homography, homography_im


# Maybe delete, not necessary. If not, create similiar Aruco function
def save_homography_data(path, undist_alpha=0, homography_args={}):
    vid_path = os.path.join(path, REALTIME_ID + ".avi")
    rt_data_path = os.path.join(path, RT_DATA_FOLDER)
    json_fn = os.path.join(rt_data_path, VID_STATS_FN)
    homography_im_fn = os.path.join(rt_data_path, HOMOGRAPHY_IM_FN)

    homography, homography_im = get_homography_from_video_black_squares(
        vid_path, undist_alpha=undist_alpha, **homography_args
    )

    if not os.path.exists(rt_data_path):
        os.mkdir(rt_data_path)

    if homography is not None:
        homography = homography.tolist()

    if homography_im is not None:
        cv.imwrite(homography_im_fn, homography_im)

    vcap = cv.VideoCapture(vid_path)
    vid_width = int(vcap.get(3))
    vid_height = int(vcap.get(4))
    vcap.release()

    with open(json_fn, "w") as fp:
        vid_stats = {"width": vid_width, "height": vid_height, "homography": homography}
        if undist_alpha != 0:
            vid_stats["undist_alpha"] = undist_alpha

        json.dump(vid_stats, fp)


def analyze_single_video(
    video_path, detector, start_frame=0, num_frames=None,
):
    """
    Analyze a single video by running each frame through the detector, and returning a 2d array of detections.
    also saving the flattened resized cropped head images in 2d array
    :return: 2D array with detections and a list with cropped heads images which are numpy 2d arrays
    """

    vcap = cv.VideoCapture(video_path)

    if start_frame != 0:
        vcap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    if num_frames is None:
        num_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT)) - start_frame

    width = int(vcap.get(3))
    height = int(vcap.get(4))
    detector.set_input_size(width, height)

    # frames_data: centroid x,centroid y, x1, y1, x2, y2, confidence, num_boxes
    frames_data = np.empty((num_frames, 8))
    frames_data[:] = np.nan

    head_crops = []

    vcap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    for frameCounter in tqdm(range(num_frames)):
        ret, frame = vcap.read()

        if not ret:
            raise DatasetException("Error reading frame.")

        detections = detector.detect_image(frame)

        if detections is not None:
            if frameCounter > 0:
                prev = frames_data[frameCounter - 1][:2]
                detection = nearest_detection(detections, prev)
            else:
                detection = detections[0]

            centroid = xyxy_to_centroid(detection)
            # detections: Each row is x1, y1, x2, y2

            frames_data[frameCounter][0:2] = centroid
            frames_data[frameCounter][2:7] = detection
            frames_data[frameCounter][7] = detections.shape[0]

            cropped_head = get_cropped_head(
                detector.curr_img, detection, (height, width)
            )

            # append cropped head to list
            head_crops.append(cropped_head)

        else:
            # if no there's no detection, append None element so the order will be kept
            head_crops.append(None)
            frames_data[frameCounter][7] = 0

    vcap.release()

    return frames_data, head_crops, width, height


def get_date_from_path(path):
    date_rgx = r"\d{8}-\d{6}"
    to_search = os.path.split(path)[-1]
    date_str = re.findall(date_rgx, to_search)[-1]
    return pd.to_datetime(date_str)


def find_last_homography(path):
    """
    Find the last homography relative to date of video by finding the calibration date with smallest
    positive timedelta relative to video date
    :param path: path to folder containing a realtime video
    :return: homography numpy array, json file path
    """
    video_date = get_date_from_path(path)
    calibration_paths = np.array(glob.glob(os.path.join(CALIBRATIONS_ROOT, "*")))
    calibration_dates = pd.Series(
        [get_date_from_path(calib_path) for calib_path in calibration_paths]
    )
    tiled_video_date = pd.Series([video_date] * calibration_dates.shape[0])
    date_diffs = tiled_video_date - calibration_dates
    dates_before = date_diffs >= pd.Timedelta(0)
    last_date_idx = (date_diffs[dates_before]).argmin()
    last_calib_path = calibration_paths[dates_before.to_numpy()][last_date_idx]

    with open(last_calib_path, "r") as fp:
        homog_dict = json.load(fp)

    return np.array(homog_dict["homography"]), last_calib_path


def analyze_rt_data(path, detector):
    """
    Analyze a single video and save data to the realtime directory:
    parse detections and other metadata into a single trial dict

    path: path of the folder containing the video.
    detector: a bbox detector for the pogona head.

    Does not return a value, saves files to disk.
    """
    vid_path = os.path.join(path, REALTIME_ID + ".avi")

    rt_data_path = os.path.join(path, RT_DATA_FOLDER)
    head_crops_fn = os.path.join(rt_data_path, HEAD_CROPS_FN)
    detections_df_fn = os.path.join(rt_data_path, DETECTIONS_DF_FN)
    json_fn = os.path.join(rt_data_path, VID_STATS_FN)
    # homography_im_fn = os.path.join(rt_data_path, HOMOGRAPHY_IM_FN)

    try:
        # analyze video with detector
        (detections, head_crops, vid_width, vid_height) = analyze_single_video(
            video_path=vid_path, detector=detector
        )
    except DatasetException:
        print(f"Error reading video file: {vid_path}")
        return

    if not os.path.exists(rt_data_path):
        os.mkdir(rt_data_path)

    detections_df = pd.DataFrame(data=detections, columns=DF_COLUMNS)
    detections_df.to_csv(detections_df_fn, index=False)

    # save data to file
    with open(head_crops_fn, "wb") as fp:
        pickle.dump(head_crops, fp)

    # find last homography relative to video date
    homography, calib_path = find_last_homography(path)

    with open(json_fn, "w") as fp:
        vid_stats = {
            "width": vid_width,
            "height": vid_height,
            "homography": homography.tolist(),
            "homography_src_file": calib_path,
        }

        json.dump(vid_stats, fp)

    print(f"Saved {detections_df_fn}\n {head_crops_fn}\n {json_fn}")


def collect_analysis_paths(
    output_root=OUTPUT_ROOT,
    experiments_root=EXPERIMENTS_ROOT,
    output_terms=OUTPUT_TERMS,
):
    """
    Return all paths that contain a video from the realtime camera, a timestamp file, 
    that do not already contain an rt_data folder and that don't contain an excluding term in their name.

    output_root: path to the output directory (general video recordings)
    experiments_root: path to the experiments directory
    return: List of paths to analyse
    """
    output_paths = []
    for term in output_terms:
        output_paths += glob.glob(os.path.join(output_root, f"*{term}*"))
    trial_paths = glob.glob(os.path.join(experiments_root, "*/trial*/videos/*"))
    all_paths = output_paths + trial_paths

    def path_filter(path):
        if os.path.exists(os.path.join(path, RT_DATA_FOLDER)):
            return False
        if not os.path.exists(os.path.join(path, REALTIME_ID + ".avi")):
            return False
        if not os.path.exists(os.path.join(path, "timestamps", REALTIME_ID + ".csv")):
            return False
        if any([dont in path for dont in EXCLUDE_TERMS]):
            return False
        return True

    return list(filter(path_filter, all_paths))


def analyze_new_data(
    detector, output_root=OUTPUT_ROOT, experiments_root=EXPERIMENTS_ROOT
):
    """
    Get all new trials from output and experiment folders, and analyze them. A function to be called
    from console or from a notebook.
    """
    paths = collect_analysis_paths(output_root, experiments_root)

    for path in paths:
        print(f"Analyzing {path}:")
        analyze_rt_data(path, detector)


""" --------------- Data Selection Functions --------------- """


def parse_exper_log(exper):
    """
    Parse the log file of an experiment
    TODO - possible to use yaml parser
    """
    with open(os.path.join(exper, "experiment.log"), "r") as f:
        exp_log = f.read()

    name = re.search(r"experiment_name: (.*)", exp_log)
    animal_id = re.search(r"animal_id: (\d+)\n", exp_log)
    num_trials = re.search(r"num_trials: (\d+)\n", exp_log)
    bug_type = re.search(r"bug_type: (.*)", exp_log)
    bug_speed = re.search(r"bug_speed: (\d+)\n", exp_log)
    mov_type = re.search(r"movement_type: (.*)", exp_log)

    d = dict()

    if name is not None:
        d["name"] = name.group(1)
    if animal_id is not None:
        d["animal_id"] = int(animal_id.group(1))
    if num_trials is not None:
        d["num_trials"] = int(num_trials.group(1))
    if bug_type is not None:
        d["bug_type"] = bug_type.group(1)
    if bug_speed is not None:
        d["bug_speed"] = int(bug_speed.group(1))
    if mov_type is not None:
        d["mov_type"] = mov_type.group(1)

    return d


def get_data_paths(path, data_sources):
    """
    :param path: a path to an trial folder
    :param data_sources: files to find paths to
    :return: a dictionary of paths to files
    """
    data_paths = {key: None for key, val in data_sources.items()}

    p = os.path.join(path, RT_DATA_FOLDER, VID_STATS_FN)
    if os.path.exists(p):
        data_paths["vid_stats"] = p
    else:
        return None

    if data_sources["detections"]:
        p = os.path.join(path, RT_DATA_FOLDER, DETECTIONS_DF_FN)
        if os.path.exists(p):
            data_paths["detections"] = p
        else:
            return None

    if data_sources["timestamps"]:
        p = os.path.join(path, TIMESTAMPS_FN)
        if os.path.exists(p):
            data_paths["timestamps"] = p
        else:
            return None

    if data_sources["dlc"]:
        p = os.path.join(path, DLC_FN)
        if os.path.exists(p):
            data_paths["dlc"] = p
        else:
            return None

    if data_sources["touches"]:
        trial_path = os.path.split(os.path.split(path)[0])[0]
        p = os.path.join(trial_path, TOUCHES_FN)
        if os.path.exists(p):
            data_paths["touches"] = p

    return data_paths


def select_paths(
    output_root=OUTPUT_ROOT,
    experiments_root=EXPERIMENTS_ROOT,
    output_terms=OUTPUT_TERMS,
    data_sources={"detections": True, "timestamps": True, "dlc": True, "touches": True},
):
    """
    Find all of the paths from which to parse data, called by collect_data function
    :param output_root: path of folder containing videos only, no additional data
    :param experiments_root: path of folder containing the experiments data
    :param output_terms: parse folders in the output folder that contain these strings
    :param data_sources: files to parse: detections, timestamps, etc.
    :return: a dictionary of dictionaries of paths, dictionary for each trial
    """
    output_paths = []
    for term in output_terms:
        output_paths += glob.glob(os.path.join(output_root, f"*{term}*"))
    trial_paths = glob.glob(os.path.join(experiments_root, "*/trial*/videos/*"))
    out_dict = {}

    for path in output_paths:
        if any([exclude in path for exclude in EXCLUDE_TERMS]):
            continue
        key = (os.path.split(path)[1], None)
        data_paths = get_data_paths(path, data_sources)
        if data_paths:
            out_dict[key] = data_paths

    for path in trial_paths:
        if any([exclude in path for exclude in EXCLUDE_TERMS]):
            continue
        split_path = path.split(os.path.sep)
        trial = split_path[-3]
        experiment = split_path[-4]
        key = (experiment, trial)
        data_paths = get_data_paths(path, data_sources)
        if data_paths:
            out_dict[key] = data_paths

    return out_dict


def collect_data(
    output_root=OUTPUT_ROOT,
    experiments_root=EXPERIMENTS_ROOT,
    output_terms=OUTPUT_TERMS,
    data_sources={"detections": True, "timestamps": True, "dlc": True, "touches": True},
    dlc_joints=("nose", "left_ear", "right_ear"),
    video_dims=(1440, 1080),
):  # TODO more filtering, according to data in JSON?
    """
    Load experimental data to RAM as a Pandas DataFrame. Assumes that the number of columns is relatively small,
    so the DataFrame can be fitted in the RAM even with hundreds of trials.
    :param output_root: path of folder containing videos only, no additional data
    :param experiments_root: path of folder containing the experiments data
    :param output_terms: parse folders in the output folder that contain these strings
    :param data_sources: files to parse: detections, timestamps, etc.
    :param dlc_joints: columns from the DLC dataframe to parse (nose, ears, etc.)
    :param video_dims: dimensions of videos to parse
    :return: a Pandas DataFrame containing all of the data
    """

    data_paths = select_paths(output_root, experiments_root, output_terms, data_sources)

    dataframes_list = []

    for trial in data_paths.keys():
        trial_dict = {key: None for key, val in data_sources.items()}

        with open(data_paths[trial]["vid_stats"], "r") as f:
            vid_stats = json.load(f)

        # skip trial if video dimensions do not fit
        if (
            not vid_stats["width"] == video_dims[0]
            or not vid_stats["height"] == video_dims[1]
        ):
            continue

        # parse source to a Pandas Dataframe
        for source in data_sources.keys():
            if data_paths[trial][source]:
                trial_dict[source] = pd.read_csv(data_paths[trial][source])

        # initialize empty dataframe and join sources to it
        df = pd.DataFrame()
        homography_column_pairs = []  # dataframe columns to calibrate later

        if data_paths[trial]["detections"]:
            df[trial_dict["detections"].columns] = trial_dict["detections"]

            homography_column_pairs += [
                ("cent_x", "cent_y",),
                ("x1", "y1"),
                ("x2", "y2"),
            ]

        if data_paths[trial]["timestamps"]:
            df["frame_ts"] = pd.to_datetime(trial_dict["timestamps"]["0"])

        # TODO: maybe arrange in a function after all
        if data_paths[trial]["dlc"] and trial_dict["dlc"] is not None:
            temp_dlc = trial_dict["dlc"]
            temp_dlc.drop(columns=["scorer"], inplace=True)
            temp_dlc.columns = [
                tup[0] + "_" + tup[1] for tup in zip(temp_dlc.iloc[0], temp_dlc.iloc[1])
            ]
            temp_dlc.drop(labels=[0, 1], inplace=True).reset_index(
                drop=True, inplace=True
            )
            drop_dlc_cols = [
                col
                for col in temp_dlc.columns
                if not any([joint in col for joint in dlc_joints])
            ]
            temp_dlc.drop(columns=drop_dlc_cols)
            df[temp_dlc.columns] = temp_dlc

            homography_column_pairs += [
                (joint + "_x", joint + "_y") for joint in dlc_joints
            ]

        if data_paths[trial]["touches"]:
            assert data_paths[trial][
                "timestamps"
            ], "Attempted aligning touches without timestamps"

            # if file exists but no touches recorded, pass
            if not trial_dict["touches"].shape[0] == 0:
                align_touches(df, trial_dict["touches"])

        df = transform_df(df, homography_column_pairs, vid_stats)

        df.index = [str(trial[0]) + "_" + str(trial[1])] * df.shape[0]

        if not (df.num_bbox == 0).all():
            dataframes_list.append(df)

    if len(dataframes_list) == 0:
        print("No data found with specified data sources")
        return

    unified_df = pd.concat(dataframes_list)
    if "is_touch" in unified_df.columns:
        unified_df.is_touch = unified_df.is_touch.fillna(False)
    if "is_hit" in unified_df.columns:
        unified_df.is_hit = unified_df.is_hit.fillna(False)

    # place NaN's instead of 0's where there are no detections
    # TODO: where they become zeros anyway?
    unified_df.loc[unified_df.num_bbox == 0, DF_COLUMNS] = np.nan

    print(f"{len(dataframes_list)} trials loaded")
    return unified_df


def transform_df(df, cols, vid_stats):
    """
    Applies the  lense undistortion and homography transformation according to pairs of columns
    :param df: panads dataframe to correct
    :param cols: iterable of pairs of columns (x,y) to correct
    :param vid_stats: dictionary containing the homography
    :return: the corrected dataframe
    """
    if vid_stats["homography"] is not None:
        homography = np.array(vid_stats["homography"])
    else:
        with open(DEFAULT_HOMOGRAPHY_JSON, "r") as fp:
            homography_dict = json.load(fp)
        homography = np.array(homography_dict["new_h"])

    if "undist_alpha" in vid_stats:
        alpha = vid_stats["undist_alpha"]
    else:
        alpha = 0

    df = calib.undistort_data(
        df, vid_stats["width"], vid_stats["height"], cols, alpha=alpha
    )

    df = calib.transform_data(df, homography, cols)

    return df


def align_touches(df, temp_touches):
    """
    Align the the screen touching data to the detections data, by aligning the data accoring to closest timestamps
    changes dataframe inplace.
    
    Added a partial fix for the timezones issue, that operates directly on the numerical diffrences and shifts the 
    screen touch timestamps by a round number of hours back. This assumes that the trials are a few minutes long
    at most, that they discrepency between the camera and screen is caused by timezones issues, so the shift is by round hours.
    This is probably prone to errors and further bugs.
    
    A more permanent and robust fix is to make sure any timestamp written by the system is written in the same timezone,
    or that the timezone itself ("Asis/Tel_Aviv", "UTC" or other) are saved with the timestamp.
    :param df: detections dataframe
    :param temp_touches: screen touches dataframe
    """
    temp_touches.drop(columns=[temp_touches.columns[0]], inplace=True)
    if "is_hit" not in temp_touches.columns:
        temp_touches.insert(
            loc=len(temp_touches.columns) - 1,
            column="is_hit",
            value=[True] * temp_touches.shape[0],
        )
    
    temp_touches.columns = ["hit_x", "hit_y", "bug_x", "bug_y", "is_hit", "timestamp"]
    temp_touches["timestamp"] = pd.to_datetime(temp_touches["timestamp"])

    # initalize columns for screen touching data
    for col in ["hit_x", "hit_y", "bug_x", "bug_y", "is_hit", "touch_ts"]:
        df[col] = np.nan
    df["is_touch"] = False
    
    # timezones workaround
    max_touch_ts = temp_touches.timestamp.max()
    max_diff_hr = np.min((max_touch_ts - df.frame_ts.min()).total_seconds()/60/60)
    hr_diff = np.round(max_diff_hr)
    
    if np.abs(hr_diff) >= 1:
        hr_delta = pd.Timedelta(hr_diff, unit='h')            
        if hr_diff < 0:
            temp_touches.timestamp = temp_touches.timestamp + hr_delta
        else:
            temp_touches.timestamp = temp_touches.timestamp - hr_delta
    
    
    # for timestamp of each touch, get frame with closest timestamp
    for i, ts in enumerate(temp_touches.timestamp):

        frame_argmin = np.argmin((df["frame_ts"] - ts).dt.total_seconds().abs())

        col_inds = [
            df.columns.get_loc(col)
            for col in ["hit_x", "hit_y", "bug_x", "bug_y", "is_hit", "touch_ts"]
        ]
        to_set = [
            temp_touches.columns.get_loc(col)
            for col in ["hit_x", "hit_y", "bug_x", "bug_y", "is_hit", "timestamp"]
        ]

        # setting values for part of row
        df.iloc[frame_argmin, col_inds] = temp_touches.iloc[i, to_set].values
        df.iloc[frame_argmin, df.columns.get_loc("is_touch")] = True
    df["touch_ts"] = pd.to_datetime(df["touch_ts"])


def ret_date(st):
    tokens = st.split("_")
    date = tokens[-1]
    return pd.to_datetime(date)


def get_cropped_dict(vid_dims, first_date, all_path=EXPERIMENTS_ROOT):
    heads_dict = dict()
    for exper in glob.glob(all_path + "*"):

        if not os.path.isdir(exper):
            continue

        exper_date = ret_date(exper)
        if exper_date < first_date:
            continue

        # ignore words
        if any([dont in exper for dont in EXP_DONT]):
            print(f"skipped {exper}, ignored word")
            continue

        exper_log = parse_exper_log(exper)
        exper_name = os.path.split(exper)[-1]
        heads_dict[exper_name] = dict()

        for k in range(1, exper_log["num_trials"] + 1):
            try:
                rt_data_path = os.path.join(exper, f"trial{k}", RT_DATA_FOLDER)

                json_fn = os.path.join(rt_data_path, VID_STATS_FN)
                with open(json_fn, "r") as fp:
                    vid_stat = json.load(fp)

                if vid_stat["width"] != vid_dims[0]:
                    print(
                        f'ignored {exper} trial{k}, {vid_stat["width"]} != {vid_dims[0]}'
                    )
                    continue

                head_crops_fn = os.path.join(rt_data_path, HEAD_CROPS_FN)

                with open(head_crops_fn, "rb") as fp:
                    heads_dict[exper_name][k] = pickle.load(fp)
            except FileNotFoundError:
                continue

    for key in list(heads_dict.keys()):
        if len(heads_dict[key].keys()) < 1:
            heads_dict.pop(key)

    return heads_dict


def heads_list2mat(l, resize):
    flt_imgs = np.empty((len(l), resize ** 2))
    flt_imgs[:] = np.nan

    for i, img in enumerate(l):
        if img is not None:
            resized_img = cv.resize(img, (resize, resize))
            flt_imgs[i, :] = resized_img.flatten()
    return flt_imgs.astype(
        "uint8"
    )  # TODO returning as uint8 converts np.nan to zero (0)


"""
Generating the full matrix from 25~ trials is 1GB, should not use or maybe restrict sizes
or number of trials
"""


def get_unified_heads_mat(vid_dims, first_date, resize=32, all_path=EXPERIMENTS_ROOT):
    heads_dict = get_cropped_dict(vid_dims, first_date, all_path)
    mat_list = []

    for key in heads_dict.keys():
        for trial in heads_dict[key].keys():
            mat_list.append(heads_list2mat(heads_dict[key][trial], resize))
    return np.concatenate(mat_list).astype("uint8")


# Utility functions


def trial_name_to_tuple(trial_name):
    spl = trial_name.split("_")
    experiment = "_".join(spl[:-1])
    trial = spl[-1] if spl[-1] != "None" else None

    return experiment, trial


def trial_tuple_to_name(trial_tuple):
    return "_".join(trial_tuple)


def get_trial_video_path(trial):
    return os.path.join(get_trial_path(trial), REALTIME_ID + ".avi")


def get_trial_path(trial):
    if type(trial) is str:
        trial = trial_name_to_tuple(trial)

    if trial[1] is not None:
        return glob.glob(
            os.path.join(EXPERIMENTS_ROOT, trial[0], trial[1], "videos/*")
        )[0]
    else:
        return os.path.join(OUTPUT_ROOT, trial[0])


def homography_for_trial(trial):

    trial_path = get_trial_path(trial)
    json_path = os.path.join(trial_path, RT_DATA_FOLDER, VID_STATS_FN)

    with open(json_path, "r") as f:
        vid_stats = json.load(f)

    return np.array(vid_stats["homography"])
