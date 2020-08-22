import glob
import pandas as pd
import numpy as np
import re
import os
import sys
import cv2 as cv
import pickle
from tqdm import tqdm

from Prediction.detector import nearest_detection, xywh_to_centroid

"""
data dictionary format
Experiments dict
    - single experiment (key - dir name) (dict)
        - name
        - animal_id
        - bug_type
        - bug_speed
        - mov_type (strings)
        - num_trials (int)
        - trials (dict with ints keys)
            - single trial (dict)
                - frames (Pandas df)
                - no_frames (bool)
                - no_dlc (bool)
                - no_screen (bool)
                - screen (Pandas df)
                - dim (tuple with video dimensions (w,h))

The module's functions iteratively scan the experiments folder and build the dictionary

TODO: make more modular
TODO: add "save batch to disk" if dictionary becomes too big 
(~hundreds-thousands of trials may become too big for RAM) 
"""

REALTIME_ID = '19506468'
DLC_FILENAME = REALTIME_ID + 'DLC_resnet50_pogona_pursuitJul19shuffle1_400000.csv'


def stats_save_frames_data(video_path,
                           detector,
                           start_frame=0,
                           num_frames=None):
    """
    Analyze a single video by running each frame through the detector

    :return: Pandas DataFrame with the centroids, raw bounding boxes, number of bboxes and conf
    """

    vcap = cv.VideoCapture(video_path)

    if start_frame != 0:
        vcap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    if num_frames is None:
        num_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT)) - start_frame

    width = int(vcap.get(3))
    height = int(vcap.get(4))
    detector.set_input_size(width, height)

    # frames_data: centroid x,centroid y, left x, top y, right x, bottom y,
    #              confidence, num_boxes
    frames_data = np.empty((num_frames, 8))
    frames_data[:] = np.nan

    print(f'analysing {video_path}, num_frames {num_frames}')

    for frameCounter in tqdm(range(num_frames)):
        ret, frame = vcap.read()

        if not ret:
            print("error reading frame")
            break

        detections = detector.detect_image(frame)

        if detections is not None:
            if frameCounter > 0:
                prev = frames_data[frameCounter - 1][:2]
                detection = nearest_detection(detections, prev)
            else:
                detection = detections[0]

            centroid = xywh_to_centroid(detection)
            # detections: Each row is x, y, w, h (top-left corner)

            frames_data[frameCounter][0:2] = centroid
            frames_data[frameCounter][2:7] = detection
            frames_data[frameCounter][7] = detections.shape[0]
        else:
            frames_data[frameCounter][7] = 0

    vcap.release()

    return frames_data, width, height


def parse_exper_log(exper_dict, exper):
    """
    Parse the log file of an experiment
    """
    with open(os.path.join(exper, 'experiment.log'), 'r') as f:
        exp_log = f.read()

    exper_dict['name'] = re.search(r'experiment_name: (.*)', exp_log).group(1)
    exper_dict['animal_id'] = int(re.search(r'animal_id: (\d+)\n', exp_log).group(1))
    exper_dict['num_trials'] = int(re.search(r'num_trials: (\d+)\n', exp_log).group(1))
    exper_dict['bug_type'] = re.search(r'bug_type: (.*)', exp_log).group(1)
    exper_dict['bug_speed'] = int(re.search(r'bug_speed: (\d+)\n', exp_log).group(1))
    exper_dict['mov_type'] = re.search(r'movement_type: (.*)', exp_log).group(1)


def trial_analyze_video(exper_dict, exper, k, detector):
    """
    Analyze a single video: parse detections and oter metadata into single trial dict
    """
    exper_dict['trials'][k] = dict()
    vid_time = os.listdir(os.path.join(exper, f'trial{k}', 'videos'))[0]
    vid_path = os.path.join(exper, f'trial{k}', 'videos', vid_time, REALTIME_ID + '.avi')
    timestamp_path = os.path.join(exper, f'trial{k}', 'videos',
                                  vid_time, 'timestamps', REALTIME_ID + '.csv')

    # ignore trial if there's no realtime recording or no timestamps
    if (not os.path.exists(vid_path)) or (not os.path.exists(timestamp_path)):
        exper_dict['trials'][k]['no_realtime'] = True
        exper_dict['trials'][k]['no_screen'] = True
        exper_dict['trials'][k]['no_dlc'] = True
        return

    exper_dict['trials'][k]['no_realtime'] = False
    exper_dict['trials'][k]['vid_time'] = vid_time

    # analyze video with YOLO detector
    frames_data_test, wid, hgt = stats_save_frames_data(video_path=vid_path, detector=detector)
    exper_dict['trials'][k]['frames'] = pd.DataFrame(data=frames_data_test,
                                                     columns=('centroid_x', 'centroid_y',
                                                              'left_x', 'top_y',
                                                              'right_x', 'bottom_y',
                                                              'conf', 'num_bbox'))

    exper_dict['trials'][k]['dims'] = (wid, hgt)  # set video dimensions

    # add more identifying columns to the results dataframe - maybe remove, maybe add more columns
    exper_dict['trials'][k]['frames']['exper_name'] = exper_dict['name']
    exper_dict['trials'][k]['frames']['trial'] = k

    # parse frames timestamps and add as a column
    timestamps = pd.read_csv(timestamp_path, parse_dates=['0'], usecols=['0'])
    timestamps.columns = ['timestamp']
    timestamps = timestamps.values.squeeze()
    exper_dict['trials'][k]['frames']['timestamp'] = timestamps


def trial_analyze_screen(exper_dict, exper, k):
    """
    Parse screen touches
    """

    screen_path = os.path.join(exper, f'trial{k}', 'screen_touches.csv')
    if not os.path.exists(screen_path):
        exper_dict['trials'][k]['no_screen'] = True
        return

    exper_dict['trials'][k]['no_screen'] = False
    exper_dict['trials'][k]['screen'] = pd.read_csv(screen_path,
                                                    usecols=['x', 'y', 'bug_x',
                                                             'bug_y', 'timestamp'],
                                                    parse_dates=['timestamp'])

    # if file exists but no touches recorded, pass
    if exper_dict['trials'][k]['screen'].shape[0] == 0:
        exper_dict['trials'][k]['no_screen'] = True
        return

    exper_dict['trials'][k]['screen'].columns = ['hit_x', 'hit_y', 'bug_x', 'bug_y', 'timestamp']

    # initalize columns for screen touching data
    for col in ['hit_x', 'hit_y', 'bug_x', 'bug_y', 'touch_ts']:
        exper_dict['trials'][k]['frames'][col] = np.nan
    exper_dict['trials'][k]['frames']['hit'] = False

    # align screen touches to frames with hit
    if exper_dict['trials'][k]['screen'].shape[0] == 0:
        exper_dict['trials'][k]['no_screen'] = True
        return
    else:
        # for timestamp of each touch, get frame with closest timestamp
        for i, ts in enumerate(exper_dict['trials'][k]['screen'].timestamp):
            frame_argmin = np.argmin((exper_dict['trials'][k]['frames']['timestamp'] - ts).dt.total_seconds().abs())

            col_inds = [exper_dict['trials'][k]['frames'].columns.get_loc(col) for col in
                        ['hit_x', 'hit_y', 'bug_x', 'bug_y', 'touch_ts']]
            to_set = [exper_dict['trials'][k]['screen'].columns.get_loc(col) for col in
                      ['hit_x', 'hit_y', 'bug_x', 'bug_y', 'timestamp']]

            # setting values for part of row
            exper_dict['trials'][k]['frames'].iloc[frame_argmin, col_inds] = \
                exper_dict['trials'][k]['screen'].iloc[i, to_set].values
            exper_dict['trials'][k]['frames'].iloc[frame_argmin,
                                                   exper_dict['trials'][k]['frames'].columns.get_loc('hit')] = True

    exper_dict['trials'][k]['screen']['exper_name'] = exper_dict['name']
    exper_dict['trials'][k]['screen']['trial'] = k


def trial_parse_dlc(dlc_path):
    """
    Parse DLC csv file to multiindexed dataframe
    assumes that row 0 can be ignored, row 1 includes bodyparts, row 2 includes coordinates
    """
    return pd.read_csv(dlc_path, header=[1, 2]).drop(columns=['bodyparts'], level=0).astype('float64')


def trial_add_dlc(exper_dict, k, dlc_path, joints=['nose']):
    """
    add DLC data if not already in the dictionary
    """

    dlc = trial_parse_dlc(dlc_path)
    for joint in joints:
        if joint + '_x' not in exper_dict['trials'][k]['frames'].columns:
            exper_dict['trials'][k]['frames'][joint + '_x'] = dlc[joint]['x']
            exper_dict['trials'][k]['frames'][joint + '_y'] = dlc[joint]['y']


def trial_update_dlc(exper_dict, exper, k, joints=['nose']):
    vid_time = os.listdir(os.path.join(exper, f'trial{k}', 'videos'))[0]
    dlc_path = os.path.join(exper, f'trial{k}', 'videos',
                            vid_time, DLC_FILENAME)
    if not os.path.exists(dlc_path):
        exper_dict['trials'][k]['no_dlc'] = True
        exper_dict['no_dlc'] = True
    else:
        # read dlc file and reindex columns for easier access
        trial_add_dlc(exper_dict, k, dlc_path, joints)
        exper_dict['trials'][k]['no_dlc'] = False


def exper_update_dlc(exper_dict, exper):
    for k in range(1, exper_dict['num_trials'] + 1):
        trial_update_dlc(exper_dict, exper, k)

    for k in range(1, exper_dict['num_trials'] + 1):
        if exper_dict['trials'][k]['no_dlc']:
            return
    exper_dict['no_dlc'] = False


def analyze_experiment(exper, detector):
    """
    Receive a path for an experiment and a detector,
    return dictionary with dataframes of analyzed trials
    """

    print(f'analysing {exper}')
    exper_dict = dict()

    parse_exper_log(exper_dict, exper)

    exper_dict['trials'] = dict()

    # Parse and analyse a single trial
    for k in range(1, exper_dict['num_trials'] + 1):
        # analyze both video and screen touches
        trial_analyze_video(exper_dict, exper, k, detector)
        trial_analyze_screen(exper_dict, exper, k)

        # add DLC joint coordinates
        trial_update_dlc(exper_dict, exper, k)

    return exper_dict


def ret_date(st):
    return pd.to_datetime(st.split('_')[-1])


BASE_PATH = '../Pogona_Pursuit/Arena/experiments/'
EXP_DONT = ['initial', 'delete', 'fps', 'vegetables', 'test-no-streaming']

FIRST_EXPR_TIMESTAMP = 'line_20200803T081429'
FIRST_TIMESTAMP = ret_date(FIRST_EXPR_TIMESTAMP)


def analyze_new_experiments(detector,
                            all_path=BASE_PATH,
                            dict_path='Dataset/all_exper.p'):
    """
    finds experiments who are not analyzed yet after first date,
    """

    if not os.path.exists(dict_path):
        all_exper = dict()
    else:
        with open('Dataset/all_exper.p', 'rb') as fp:
            all_exper = pickle.load(fp)

    for exper in glob.glob(all_path + '*'):
        # broken path
        if not os.path.isdir(exper):
            print(f'{exper} not a dir')
            continue

        # ignore words
        if any([dont in exper for dont in EXP_DONT]):
            print(f'skipped {exper}, ignored word')
            continue

        # parse date
        exper_name = os.path.split(exper)[-1]
        exper_date = ret_date(exper_name)

        # ignore experiments before first date
        if exper_date < FIRST_TIMESTAMP:
            print(f'skipped {exper}, no timestamps')
            continue

        # if already analyzed, check if there's dlc to update
        if exper_name in all_exper.keys():
            if all_exper[exper_name]['no_dlc']:
                exper_update_dlc(all_exper[exper_name], exper)
            continue

        print(f'analysing {exper}')

        exper_dict = analyze_experiment(exper, detector)

        exper_name = os.path.split(exper)[-1]
        all_exper[exper_name] = exper_dict

        with open(dict_path, 'wb') as fp:
            pickle.dump(all_exper, fp)

    with open(dict_path, 'wb') as fp:
        pickle.dump(all_exper, fp)

    print(f'Analysis complete')

    return all_exper


def get_unified_df(all_exper, mult=False):
    """
    Parse experiments to a unified dataframe
    return with multiindex if mult==True
    :return: Pandas df with all experiments and trials
    """

    first = True
    all_df = pd.DataFrame()
    for exper in all_exper.keys():
        for trial in all_exper[exper]['trials'].keys():
            if first:
                if all_exper[exper]['trials'][trial]['no_realtime']:
                    continue
                all_df = all_exper[exper]['trials'][trial]['frames']
                all_df['frame_ind'] = np.arange(all_df.shape[0])
                first = False
            else:
                if all_exper[exper]['trials'][trial]['no_realtime']:
                    continue
                new = all_exper[exper]['trials'][trial]['frames']
                new['frame_ind'] = np.arange(new.shape[0])
                all_df = pd.concat([all_df, new])
    if mult:
        mult_tuples = list(zip(all_df.exper_name, all_df.trial))
        all_df = all_df.set_index(keys=pd.MultiIndex.from_tuples(mult_tuples))
        all_df.drop(columns=['exper_name', 'trial'], inplace=True)

    # additional corrections and additions for the dataframe
    # TODO make more modular
    all_df.hit.replace(np.nan, False, inplace=True)
    all_df['touch_ts'] = pd.to_datetime(all_df['touch_ts'])

    return all_df
