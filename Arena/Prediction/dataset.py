import glob
import pandas as pd
import numpy as np
import re
import os
import sys
import cv2 as cv
import pickle
from tqdm import tqdm
import torch
import json
import pickle

from Prediction.detector import nearest_detection, xyxy_to_centroid
import Prediction.calibration as calib
#  maybe import less from the calibration module?

REALTIME_ID = '19506468'
DLC_FILENAME = REALTIME_ID + 'DLC_resnet50_pogona_pursuitJul19shuffle1_400000.csv'
BASE_PATH = '../../Pogona_Pursuit/Arena/experiments/'  # depending on from where the program is run, add ../..
RT_DATA_FOLDER = 'rt_data'
HEAD_CROPS_FN = 'head_crops.p'
DETECTIONS_DF_FN = 'detections_df.csv'
VID_STATS_FN = 'vid_stats.json'

EXP_DONT = ['initial', 'delete', 'fps', 'vegetables', 'test-no-streaming','sleepy']

DF_COLUMNS = ['cent_x', 'cent_y', 'x1', 'y1', 'x2', 'y2', 'conf', 'num_bbox']

"""
TODO these variables are useless
"""


# FIRST_EXPR_TIMESTAMP = 'line_20200803T081429'
# FIRST_TIMESTAMP = ret_date(FIRST_EXPR_TIMESTAMP)

"""
TODO - cropped heads are saved before any correction. coordinate transformation might also be relevant
"""
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

    cropped_head = img[dst_y1: dst_y2, dst_x1: dst_x2].copy()  # open_cv shape order - (Y,X) not (X,Y)

    if resize_size is not None:
        cropped_head = cv.resize(cropped_head, (resize_size, resize_size))

    cropped_head = cv.cvtColor(cropped_head, cv.COLOR_BGR2GRAY)
    cropped_head = 255 - cropped_head

    return cropped_head


def stats_save_frames_data(video_path,
                           detector,
                           resize_head=40,
                           start_frame=0,
                           num_frames=None):
    """
    Analyze a single video by running each frame through the detector, and returning a 2d array
    also saving the flattened resized cropped head images in 2d array
    return: 2 2d arrays, one with detections and one with cropped head
    """

    vcap = cv.VideoCapture(video_path)

    if start_frame != 0:
        vcap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    if num_frames is None:
        num_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT)) - start_frame

    width = int(vcap.get(3))
    height = int(vcap.get(4))
    detector.set_input_size(width, height)

    # frames_data: centroid x,centroid y, left x, top y, w, h,
    #              confidence, num_boxes
    frames_data = np.empty((num_frames, 8))
    frames_data[:] = np.nan

    # num frames rows by resize^2 columns to store crops of head
    # head_crops = np.empty((num_frames, resize_head ** 2))
    # head_crops[:] = np.nan

    head_crops = []
    aff, aff_im, screen_size = None, None, None

    print(f'num_frames {num_frames}')

    for frameCounter in tqdm(range(num_frames)):
        ret, frame = vcap.read()

        if not ret:
            print("error reading frame")
            break

        """
        calculate and save affine calibration matrix and p_norm on the first frame
        assumes that the black squares used for calibration are visible in the arena
        ************************** not tested yet **********************************
        """
        if frameCounter == 0:
            mapx, mapy, roi = calib.get_undistort_mapping(width, height, calib.MTX, calib.DIST)
            undistorted_img = calib.undistort_image(frame,(mapx, mapy))
            try:
                aff, aff_im, screen_size = calibrate(undistorted_img)
            except calib.CalibrationException:
                print(f'Could not calibrate video at {video_path}')


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

            cropped_head = get_cropped_head(detector.curr_img,
                                            detection, (height, width))

            """
            # save flattened cropped head image to matrix
            head_crops[frameCounter][:] = cropped_head
            """
            # append cropped head to list
            head_crops.append(cropped_head)

        else:
            # if no there's no detection, append None element so the order will be kept
            head_crops.append(None)
            frames_data[frameCounter][7] = 0

    vcap.release()

    return frames_data, head_crops, width, height, aff, aff_im, screen_size


def parse_exper_log(exper):
    """
    Parse the log file of an experiment
    """
    with open(os.path.join(exper, 'experiment.log'), 'r') as f:
        exp_log = f.read()

    name = re.search(r'experiment_name: (.*)', exp_log)
    animal_id = re.search(r'animal_id: (\d+)\n', exp_log)
    num_trials = re.search(r'num_trials: (\d+)\n', exp_log)
    bug_type = re.search(r'bug_type: (.*)', exp_log)
    bug_speed = re.search(r'bug_speed: (\d+)\n', exp_log)
    mov_type = re.search(r'movement_type: (.*)', exp_log)

    d = dict()

    if name is not None:
        d['name'] = name.group(1)
    if animal_id is not None:
        d['animal_id'] = int(animal_id.group(1))
    if num_trials is not None:
        d['num_trials'] = int(num_trials.group(1))
    if bug_type is not None:
        d['bug_type'] = bug_type.group(1)
    if bug_speed is not None:
        d['bug_speed'] = int(bug_speed.group(1))
    if mov_type is not None:
        d['mov_type'] = mov_type.group(1)

    return d


def get_trial_path(exper, trial_k):
    """
    :param exper: path of the root of a single experiment
    :param trial_k: number of trial
    :return: paths to the video and timestamps files in the trial folder
    """
    vid_time = os.listdir(os.path.join(exper, f'trial{trial_k}', 'videos'))[0]
    vid_path = os.path.join(exper, f'trial{trial_k}', 'videos', vid_time, REALTIME_ID + '.avi')
    ts_path = os.path.join(exper, f'trial{trial_k}', 'videos', vid_time, 'timestamps', REALTIME_ID + '.csv')  # not here
    return vid_path, ts_path


def trial_analyze_video(exper, k, detector):
    """
    Analyze a single video: parse detections and other metadata into single trial dict
    exper: path of experiment
    """
    try:
        vid_path, timestamp_path = get_vid_path(exper, k)
    except FileNotFoundError:
        return

    rt_data_path = os.path.join(exper, f"trial{k}", RT_DATA_FOLDER)
    head_crops_fn = os.path.join(rt_data_path, HEAD_CROPS_FN)
    detections_df_fn = os.path.join(rt_data_path, DETECTIONS_DF_FN)
    affine_trans_fn = os.path.join(rt_data_path, )

    # if already analyzed, continue

    if os.path.exists(head_crops_fn) and os.path.exists(detections_df_fn):
        print(f'skipped {exper}, already analyzed')
        return

    # ignore trial if there's no realtime recording or no timestamps
    if (not os.path.exists(vid_path)) or (not os.path.exists(timestamp_path)):
        print(f'{vid_path} or {timestamp_path} dont exist')
        return

    if not os.path.exists(rt_data_path):
        os.mkdir(rt_data_path)

    # analyze video with YOLO detector
    frames_data_test, head_crops, wid, hgt, \
    aff, aff_im, screen_size = stats_save_frames_data(video_path=vid_path,
                                                      detector=detector)
    detections_df = pd.DataFrame(data=frames_data_test, columns=DF_COLUMNS)

    # parse frames timestamps and add as a column
    timestamps = pd.read_csv(timestamp_path, parse_dates=['0'], usecols=['0'])
    timestamps.columns = ['frame_ts']
    timestamps = timestamps.values.squeeze()
    detections_df['frame_ts'] = timestamps

    # save data to file
    with open(head_crops_fn, 'wb') as fp:
        pickle.dump(head_crops, fp)
    detections_df.to_csv(detections_df_fn, index=False)
    json_fn = os.path.join(rt_data_path, VID_STATS_FN)

    # save affine transformation as a 2D list if not None, convert back to numpy upon loading
    if aff is not None:
        aff = aff.tolist()
    with open(json_fn, 'w') as fp:
        json.dump({'width': wid, 'height': hgt, 'affine_mat': aff, 'screen_size': screen_size}, fp)

    print(f'saved {detections_df_fn}\n {head_crops_fn}\n {json_fn}')


def align_screen_touches(exper, detections_df, k):
    """
    Gets the detections dataframe and adds the screen touches data inplace
    does not return a value
    TODO - maybe return a new dataframe instead
    """

    screen_path = os.path.join(exper, f'trial{k}', 'screen_touches.csv')
    if not os.path.exists(screen_path):
        return

    screen_df = pd.read_csv(screen_path,
                            usecols=['x', 'y', 'bug_x',
                                     'bug_y', 'timestamp'],
                            parse_dates=['timestamp'])

    # if file exists but no touches recorded, pass
    if screen_df.shape[0] == 0:
        return

    screen_df.columns = ['hit_x', 'hit_y', 'bug_x', 'bug_y', 'timestamp']

    # initalize columns for screen touching data
    for col in ['hit_x', 'hit_y', 'bug_x', 'bug_y', 'touch_ts']:
        detections_df[col] = np.nan
    detections_df['hit'] = False

    # for timestamp of each touch, get frame with closest timestamp
    for i, ts in enumerate(screen_df.timestamp):
        frame_argmin = np.argmin((detections_df['frame_ts'] - ts).dt.total_seconds().abs())

        col_inds = [detections_df.columns.get_loc(col) for col in
                    ['hit_x', 'hit_y', 'bug_x', 'bug_y', 'touch_ts']]
        to_set = [screen_df.columns.get_loc(col) for col in
                  ['hit_x', 'hit_y', 'bug_x', 'bug_y', 'timestamp']]

        # setting values for part of row
        detections_df.iloc[frame_argmin, col_inds] = \
            screen_df.iloc[i, to_set].values
        detections_df.iloc[frame_argmin,
                           detections_df.columns.get_loc('hit')] = True


def trial_parse_dlc(dlc_path):
    """
    Parse DLC csv file to multiindexed dataframe
    assumes that row 0 can be ignored, row 1 includes bodyparts, row 2 includes coordinates
    """
    return pd.read_csv(dlc_path, header=[1, 2]).drop(columns=['bodyparts'], level=0).astype('float64')


def trial_add_dlc(det_df, dlc_path, joints=('nose')):
    """
    add DLC data if not already in the dictionary
    changes the detections dataframe in place
    TODO maybe return a new dataframe instead
    """

    dlc = trial_parse_dlc(dlc_path)
    for joint in joints:
        if joint + '_x' not in det_df.columns:
            det_df[joint + '_x'] = dlc[joint]['x']
            det_df[joint + '_y'] = dlc[joint]['y']


def analyze_experiment(exper, detector):
    """
    Receive a path for an experiment and a detector,
    return dictionary with dataframes of analyzed trials
    """

    print(f'Analysing {exper}')

    # TODO parse experiment log data?
    exper_details = parse_exper_log(exper)

    # Parse and analyse a single trial
    for k in range(1, exper_details['num_trials'] + 1):
        trial_analyze_video(exper, k, detector)

    print(f'Finished {exper}')


def ret_date(st):
    tokens = st.split('_')
    date = tokens[-1]
    return pd.to_datetime(date)


def analyze_new_experiments(detector,
                            first_date,
                            all_path=BASE_PATH):
    """
    finds experiments who are not analyzed yet after first date,
    """
    for exper in glob.glob(all_path + '*'):

        if not os.path.isdir(exper):
            continue

        exper_date = ret_date(exper)
        if exper_date < first_date:
            continue

        # ignore words
        if any([dont in exper for dont in EXP_DONT]):
            print(f'skipped {exper}, ignored word')
            continue

        analyze_experiment(exper, detector)


def get_unified_dataframe(vid_dims,
                          first_date=None,
                          all_path=BASE_PATH,
                          align_touch_screen=False,
                          add_dlc=False,
                          multi_index=True,
                          to_correct=False,
                          ):

    df_list = []
    count = 0
    for exper in glob.glob(all_path + '*'):

        if not os.path.isdir(exper):
            continue

        exper_date = ret_date(exper)
        if first_date is not None and exper_date < first_date:
            continue

        # ignore words
        if any([dont in exper for dont in EXP_DONT]):
            print(f'skipped {exper}, ignored word')
            continue

        try:
            exper_log = parse_exper_log(exper)
        except FileNotFoundError:
            print(f'skipped {exper}, no experiment log file')
            continue

        for k in range(1, exper_log['num_trials'] + 1):

            rt_data_path = os.path.join(exper, f"trial{k}", RT_DATA_FOLDER)
            #head_crops_fn = os.path.join(rt_data_path, HEAD_CROPS_FN)
            detections_df_fn = os.path.join(rt_data_path, DETECTIONS_DF_FN)
            json_fn = os.path.join(rt_data_path, VID_STATS_FN)
            try:
                with open(json_fn, 'r') as fp:
                    vid_stat = json.load(fp)

                if vid_stat['width'] != vid_dims[0]:
                    print(f'ignored {exper} trial{k}, {vid_stat["width"]} != {vid_dims[0]}')
                    continue

                trial_df = pd.read_csv(detections_df_fn, parse_dates=['frame_ts'])
                trial_df['exper'] = os.path.split(exper)[-1]
                trial_df['trial'] = k

                if to_correct:
                    #print(f'{exper} trial{k} ')
                    _, roi, newcameramtx = calib.get_undistort_mapping(vid_stat['width'], vid_stat['height'],
                                                                  calib.MTX, calib.DIST)

                    trial_df = calib.undistort_data(trial_df, vid_stat['width'], vid_stat['height'])

                    if vid_stat['affine_mat'] is not None:
                        aff_mat = np.array(vid_stat['affine_mat'])
                        #screen_size = vid_stat['screen_size']

                        trial_df = calib.transform_data(trial_df, aff_mat)
                    else:
                        print(f'{exper} trial{k} is not calibrated')

                if align_touch_screen:
                    align_screen_touches(exper, trial_df, k)

                if add_dlc:

                    vid_time = os.listdir(os.path.join(exper, f'trial{k}', 'videos'))[0] # TODO fucntion to get vid_time
                    dlc_path = os.path.join(exper, f'trial{k}', 'videos', vid_time, DLC_FILENAME)
                    trial_add_dlc(trial_df, dlc_path)

            except FileNotFoundError:
                print(f'did not find files in {exper} trial{k}')
                continue

            df_list.append(trial_df)
            count += 1

    unified_df = pd.concat(df_list)

    if multi_index:
        unified_df.set_index(['exper', 'trial'], inplace=True)

    # not clear where this column was added to the dataframe
    if 'Unnamed: 0' in unified_df.columns:
        unified_df.drop(columns=['Unnamed: 0'], inplace=True)

    if 'hit' in unified_df:
        unified_df['hit'] = unified_df['hit'].fillna(False)

    unified_df.rename(columns={'left_x': 'x', 'right_x': 'w', 'top_y': 'y', 'bottom_y': 'h'}, inplace=True)

    print(f'Finished, loaded {count} trials')
    return unified_df


def get_cropped_dict(vid_dims,
                     first_date,
                     all_path=BASE_PATH):
    heads_dict = dict()
    for exper in glob.glob(all_path + '*'):

        if not os.path.isdir(exper):
            continue

        exper_date = ret_date(exper)
        if exper_date < first_date:
            continue

        # ignore words
        if any([dont in exper for dont in EXP_DONT]):
            print(f'skipped {exper}, ignored word')
            continue

        exper_log = parse_exper_log(exper)
        exper_name = os.path.split(exper)[-1]
        heads_dict[exper_name] = dict()

        for k in range(1, exper_log['num_trials'] + 1):
            try:
                rt_data_path = os.path.join(exper, f"trial{k}", RT_DATA_FOLDER)

                json_fn = os.path.join(rt_data_path, VID_STATS_FN)
                with open(json_fn, 'r') as fp:
                    vid_stat = json.load(fp)

                if vid_stat['width'] != vid_dims[0]:
                    print(f'ignored {exper} trial{k}, {vid_stat["width"]} != {vid_dims[0]}')
                    continue

                head_crops_fn = os.path.join(rt_data_path, HEAD_CROPS_FN)

                with open(head_crops_fn, 'rb') as fp:
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
            resized_img = cv.resize(img, (resize,resize))
            flt_imgs[i, :] = resized_img.flatten()
    return flt_imgs.astype('uint8')  # TODO returning as uint8 converts np.nan to zero (0)

"""
Generating the full matrix from 25~ trials is 1GB, should not use or maybe restrict sizes
or number of trials
"""
def get_unified_heads_mat(vid_dims,
                          first_date,
                          resize=32,
                          all_path=BASE_PATH):

    heads_dict = get_cropped_dict(vid_dims, first_date,all_path)
    mat_list = []

    for key in heads_dict.keys():
        for trial in heads_dict[key].keys():
            mat_list.append(heads_list2mat(heads_dict[key][trial], resize))
    return np.concatenate(mat_list).astype('uint8')

