import time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import config
import math
from analysis.pose import ArenaPose
from analysis.pose_utils import put_text
from db_models import ORM, Block, Strike, Trial, Temperature

DEFAULT_OUTPUT_DIR = '/data/Pogona_Pursuit/output'


class MissingStrikeData(Exception):
    """Could not find timestamps for video frames"""


class Loader:
    def __init__(self, strike_db_id, cam_name, is_load_pose=True, is_use_db=True, is_debug=True, orm=None,
                 sec_before=3, sec_after=2):
        self.strike_db_id = strike_db_id
        self.cam_name = cam_name
        self.is_load_pose = is_load_pose
        self.is_use_db = is_use_db
        self.is_debug = is_debug
        self.sec_before = sec_before
        self.sec_after = sec_after
        self.orm = orm if orm is not None else ORM()
        self.frames_delta = None
        self.n_frames_back = None
        self.n_frames_forward = None
        self.bug_traj_strike_id = None
        self.bug_traj_before_strike = None
        self.strike_frame_id = None
        self.video_path = None
        self.avg_temperature = None
        if self.is_load_pose:
            self.dlc_pose = ArenaPose(cam_name, 'deeplabcut', is_use_db=is_use_db, orm=orm)
        self.frames_df: pd.DataFrame = pd.DataFrame()
        self.traj_df: pd.DataFrame = pd.DataFrame(columns=['time', 'x', 'y'])
        self.info = {}
        self.load()

    def __str__(self):
        return f'Strike-Loader:{self.strike_db_id}'

    def load(self):
        with self.orm.session() as s:
            n_tries = 3
            for i in range(n_tries):
                try:
                    strk = s.query(Strike).filter_by(id=self.strike_db_id).first()
                    break
                except Exception as exc:
                    time.sleep(0.2)
                    if i >= n_tries - 1:
                        raise exc
            if strk is None:
                raise MissingStrikeData(f'could not find strike id: {self.strike_db_id}')

            self.info = {k: v for k, v in strk.__dict__.items() if not k.startswith('_')}
            trial = s.query(Trial).filter_by(id=strk.trial_id).first()
            if trial is None:
                raise MissingStrikeData('No trial found in DB')

            self.load_bug_trajectory_data(trial, strk)
            if self.is_load_pose:
                self.load_frames_data(s, trial, strk)
            self.load_temperature(s, trial.block_id)

    def load_bug_trajectory_data(self, trial, strk):
        self.traj_df = pd.DataFrame(trial.bug_trajectory)
        if self.traj_df.empty:
            raise MissingStrikeData('traj_df is empty')
        self.traj_df['time'] = pd.to_datetime(self.traj_df.time).dt.tz_localize(None)
        self.bug_traj_strike_id = (strk.time - self.traj_df.time).dt.total_seconds().abs().idxmin()

        n = self.sec_before / self.traj_df['time'].diff().dt.total_seconds().mean()
        self.bug_traj_before_strike = self.traj_df.loc[self.bug_traj_strike_id-n:self.bug_traj_strike_id].copy()

    def get_bug_traj_around_strike(self, sec_before=None, sec_after=None):
        if self.traj_df.empty:
            return None

        n_before = (sec_before or self.sec_before) / self.traj_df['time'].diff().dt.total_seconds().mean()
        n_after = (sec_after or self.sec_after) / self.traj_df['time'].diff().dt.total_seconds().mean()
        return self.traj_df.loc[self.bug_traj_strike_id - n_before:self.bug_traj_strike_id+n_after].copy()

    def load_frames_data(self, s, trial, strk):
        block = s.query(Block).filter_by(id=trial.block_id).first()
        self.update_info_with_block_data(block)
        for vid in block.videos:
            if vid.cam_name != self.cam_name:
                continue

            frames_times = self.load_frames_times(vid)
            # check whether strike's time is in the loaded frames_times
            if not frames_times.empty and \
                    (frames_times.iloc[0].time <= strk.time <= frames_times.iloc[-1].time):
                # if load pose isn't needed finish here
                self.strike_frame_id = (strk.time - frames_times.time).dt.total_seconds().abs().idxmin()
                if not self.is_load_pose:
                    self.frames_df = frames_times
                # otherwise, load all pose data around strike frame
                else:
                    try:
                        self.load_pose(vid)
                    except Exception as exc:
                        raise MissingStrikeData(str(exc))
                # break since the relevant video was found
                break
            # if strike's time not in frames_times continue to the next video
            else:
                continue

        if self.frames_df.empty:
            raise MissingStrikeData('frames_df is empty after loading')

    def load_pose(self, vid):
        if not self.is_use_db:
            self.set_video_path(vid)
            pose_df = self.dlc_pose.load(video_path=self.video_path, only_load=True)
        else:
            pose_df = self.dlc_pose.load(video_db_id=vid.id)
        first_frame = max(self.strike_frame_id - self.n_frames_back, pose_df.index[0])
        last_frame = min(self.strike_frame_id + self.n_frames_forward, pose_df.index[-1])
        self.frames_df = pose_df.loc[first_frame:last_frame].copy()
        self.frames_df['time'] = pd.to_datetime(self.frames_df.time, unit='s')

    def set_video_path(self, vid):
        video_path = Path(vid.path).resolve()
        # fix for cases in which the analysis runs from other servers
        if DEFAULT_OUTPUT_DIR != config.OUTPUT_DIR and video_path.as_posix().startswith(DEFAULT_OUTPUT_DIR):
            video_path = Path(video_path.as_posix().replace(DEFAULT_OUTPUT_DIR, config.OUTPUT_DIR))
        if not video_path.exists():
            if self.is_debug:
                print(f'Video path does not exist: {video_path}')
            raise Exception(f'Video path {video_path} does not exist')
        self.video_path = video_path

    def update_info_with_block_data(self, blk: Block):
        fields = ['movement_type', 'exit_hole', 'bug_speed']
        self.info.update({k: blk.__dict__.get(k) for k in fields})

    def load_frames_times(self, vid):
        frames_times = self.dlc_pose.load_frames_times(vid.id, vid.path)
        if not frames_times.empty:
            self.frames_delta = np.mean(frames_times.time.diff().dt.total_seconds())
            self.n_frames_back = round(self.sec_before / self.frames_delta)
            self.n_frames_forward = round(self.sec_after / self.frames_delta)
        return frames_times

    def load_temperature(self, s, block_id):
        temps = s.query(Temperature).filter_by(block_id=block_id).all()
        if not temps:
            self.avg_temperature = np.nan
        else:
            self.avg_temperature = np.mean([t.value for t in temps if isinstance(t.value, (int, float))])

    def get_strike_frame(self) -> np.ndarray:
        for _, frame in self.gen_frames_around_strike(0, 1):
            return frame

    def gen_frames(self, frame_ids):
        cap = cv2.VideoCapture(self.video_path.as_posix())
        start_frame, end_frame = frame_ids[0], frame_ids[-1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if i not in frame_ids:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield i, frame
        cap.release()

    def gen_frames_around_strike(self, n_frames_back=None, n_frames_forward=None, center_frame=None, step=1):
        n_frames_back, n_frames_forward = n_frames_back or self.n_frames_back, n_frames_forward or self.n_frames_forward
        center_frame = center_frame or self.strike_frame_id
        start_frame = center_frame - (n_frames_back * step)
        frame_ids = [i for i in range(start_frame, start_frame + step * (n_frames_back + n_frames_forward), step)]
        return self.gen_frames(frame_ids)

    def get_bodypart_pose(self, bodypart):
        return pd.concat([pd.to_datetime(self.frames_df['time'], unit='s'), self.frames_df[bodypart]], axis=1)

    # def load_tongues_out(self):
    #     if not self.is_load_tongue:
    #         return
    #     toa = TongueOutAnalyzer(is_debug=self.is_debug)
    #     cache_path = self.dlc_pose.get_predicted_cache_path(self.video_path).parent
    #     cache_path /= f'{self}_{toa.identifier}_{self.n_frames_back}_{self.n_frames_forward}.pkl'
    #     if cache_path.exists():
    #         res = pd.read_pickle(cache_path)
    #     else:
    #         res = {}
    #         for i, frame in self.gen_frames_around_strike():
    #             label, _ = toa.tr.predict(frame)
    #             res[i] = label == TONGUE_CLASS
    #             # self.frames_df.loc[i, [TONGUE_COL]] = 1
    #             # cv2.imwrite(f'{TONGUE_PREDICTED_DIR}/{self.video_path.stem}_{i}.jpg', frame)
    #         res = pd.Series(res, name=TONGUE_COL)
    #         res.to_pickle(cache_path.as_posix())
    #
    #     self.frames_df = self.frames_df.merge(res, left_index=True, right_index=True, how='left')

    # def plot_strike_events(self, n_frames_back=100, n_frames_forward=20):
    #     plt.figure()
    #     plt.axvline(self.strike_frame_id, color='r')
    #     start_frame = self.strike_frame_id - n_frames_back
    #     end_frame = start_frame + n_frames_back + n_frames_forward
    #     plt.xlim([start_frame, end_frame])
    #     for i in range(start_frame, end_frame):
    #         tongue_val = self.frames_df[TONGUE_COL][i]
    #         if tongue_val == 1:
    #             plt.axvline(i, linestyle='--', color='b')
    #     plt.title(str(self))
    #     plt.show()

    def play_strike(self, n_frames_back=None, n_frames_forward=None, annotations=None):
        n_frames_back = n_frames_back or self.n_frames_back
        n_frames_forward = n_frames_forward or self.n_frames_forward
        nose_df = self.frames_df['nose']
        for i, frame in self.gen_frames_around_strike(n_frames_back, n_frames_forward):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            if i == self.strike_frame_id:
                put_text('Strike Frame', frame, 30, 20)
            if annotations and i in annotations:
                put_text(annotations[i], frame, 30, frame.shape[0]-30)
            if self.is_load_pose:
                self.dlc_pose.predictor.plot_predictions(frame, i, self.frames_df)

            if i in nose_df.index and not np.isnan(nose_df['cam_x'][i]):
                # put_text(f'({nose_df["cam_x"][i]:.0f}, {nose_df["cam_y"][i]:.0f})', frame, 1000, 30)
                # put_text(f'({nose_df["x"][i]:.0f}, {nose_df["y"][i]:.0f})', frame, 1000, 70)
                angle = self.frames_df.loc[i, [("angle", "")]]
                put_text(f'Angle={math.degrees(angle):.0f}', frame, 1000, 30)

            frame = cv2.resize(frame, None, None, fx=0.5, fy=0.5)
            cv2.imshow(str(self), frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def get_block_info(self):
        with self.orm.session() as s:
            strk = s.query(Strike).filter_by(id=self.strike_db_id).first()
            blk = s.query(Block).filter_by(id=strk.block_id).first()
            return blk.__dict__
