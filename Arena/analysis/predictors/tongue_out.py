import cv2
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage, Grayscale
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
if Path('.').parent.name != 'Arena':
    import os
    os.chdir('/data/Pogona_Pursuit/Arena')
from analysis.image_embedding import ResNetPretrained
from db_models import ORM, Experiment, Block, Video, VideoPrediction, Strike, Trial
from analysis.trainer import ClassificationTrainer

DATASET_PATH = Path('/data/Pogona_Pursuit/output/datasets/pogona_tongue/dataset')
TONGUE_PREDICTED_DIR = '/data/Pogona_Pursuit/output/datasets/pogona_tongue/predicted/tongues'
# MODEL_PATH = '/data/Pogona_Pursuit/output/models/tongue_out/20221216_170652'
MODEL_PATH = '/data/Pogona_Pursuit/output/models/tongue_out/20230114_221722'
RESIZE_SHAPE = (480, 640)
THRESHOLD = 0.95
TONGUE_CLASS = 'tongues'


class TongueModel(nn.Module):
    def __init__(self, input_size=512, is_embedded_input=False):
        super().__init__()
        self.is_embedded_input = is_embedded_input
        self.embedding = ResNetPretrained(is_grey=True)
        self.fc1 = nn.Linear(input_size, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 2)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        if not self.is_embedded_input:
            _, x = self.embedding(x)
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x


@dataclass
class TongueTrainer(ClassificationTrainer):
    batch_size = 16
    num_epochs = 30
    targets = ['no_tongues', 'tongues']
    monitored_metric = 'auc'
    transforms = [
        Grayscale(),
        Resize(RESIZE_SHAPE),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ]

    def get_dataset(self):
        dataset = ImageFolder(DATASET_PATH.as_posix(), transform=Compose(self.transforms))
        return dataset

    def get_model(self):
        return TongueModel()

    def predict(self, frame):
        self.model.eval()
        transforms = [ToPILImage()] + self.transforms
        img_tensor = Compose(transforms)(frame).to(self.device)
        outputs = self.model(img_tensor.unsqueeze(0))
        y_pred, y_score = self.predict_proba(outputs)
        label = self.targets[y_pred.item()]
        prob = y_score.item()
        return label, prob


PREDICTION_STACK_DURATION = 0.25  # sec
TONGUE_ACTION_TIMEOUT = 1  # sec
NUM_TONGUES_IN_STACK = 6  # number of predicted tongues in the prediction stack to trigger action


class TongueOutAnalyzer:
    def __init__(self, td: TongueTrainer = None, action_callback=None, identifier=None, is_write_detected_image=False,
                 is_debug=True):
        self.tr = TongueTrainer(model_path=MODEL_PATH, is_debug=is_debug) if td is None else td
        self.identifier = identifier
        self.action_callback = action_callback
        self.is_debug = is_debug
        self.is_write_detected_image = is_write_detected_image
        self.last_action_timestamp = None
        self.last_tongue_detect_timestamp = None
        self.predictions_stack = []
        self.last_predicted_frame = None

    def predict(self, frame, timestamp):
        label, _ = self.tr.predict(frame)
        resized_frame = cv2.resize(frame, RESIZE_SHAPE)
        self.push_to_predictions_stack(label, timestamp)
        is_tongue = label == TONGUE_CLASS
        is_action = self.tongue_detected(frame, timestamp) if is_tongue else False
        return is_action, resized_frame, label

    def push_to_predictions_stack(self, label, timestamp):
        self.predictions_stack.append((label, timestamp))
        for lbl, ts in self.predictions_stack.copy():
            if timestamp - ts > PREDICTION_STACK_DURATION:
                self.predictions_stack.remove((lbl, ts))

    def tongue_detected(self, frame: np.ndarray, timestamp: float):
        self.last_tongue_detect_timestamp = timestamp
        is_action = False
        # detect only if are enough tongue predictions in predictions stack and action timeout reached
        if (sum([x[0] == TONGUE_CLASS for x in self.predictions_stack]) >= NUM_TONGUES_IN_STACK) and \
                (not self.last_action_timestamp or timestamp - self.last_action_timestamp > TONGUE_ACTION_TIMEOUT):
            is_action = True
            self.last_action_timestamp = timestamp
            self.write_detected_frame(frame, timestamp)
        return is_action

    def write_detected_frame(self, frame, timestamp):
        if not self.is_write_detected_image:
            return
        if self.last_predicted_frame is None or np.mean(np.abs(frame - self.last_predicted_frame)) > 15:
            self.last_predicted_frame = frame.copy()
            filename = f'{self.identifier}_{round(timestamp)}' if self.identifier else f'{round(timestamp)}'
            cv2.imwrite(f'{TONGUE_PREDICTED_DIR}/{filename}.jpg', frame)


class TonguesOutVideoAnalyzer(TongueOutAnalyzer):
    def __init__(self, vid_path, **kwargs):
        self.vid_path = Path(vid_path).resolve()
        self.identifier = self.vid_path.stem
        assert self.vid_path.exists(), f'Video file {self.vid_path} does not exist'
        super(TonguesOutVideoAnalyzer, self).__init__(**kwargs)
        self.cap = cv2.VideoCapture(self.vid_path.as_posix())
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.tongue_timestamps = []
        self.action_timestamps = []  # tongue frame ids that trigger action

    def predict_video(self, is_show_video=False):
        for frame_id in tqdm(range(self.n_frames)):
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = frame_id / self.fps
            is_action, _, label = self.predict(frame, timestamp)
            if is_action:
                self.action_timestamps.append(timestamp)
            if is_show_video:
                self.play_labelled_video(frame, label)
        self.cap.release()
        self.plot_video_predictions()

    def plot_video_predictions(self):
        for ts in self.action_timestamps:
            plt.axvline(ts, linestyle='--', color='b')

        for frame_id in self.get_video_strikes_frames_ids(self.vid_path):
            plt.axvline(frame_id/self.fps, color='r')
        plt.show()

    def play_labelled_video(self, frame, label):
        frame = cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.vid_path.name, frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            raise Exception('q was pressed')

    def get_video_strikes_frames_ids(self, vid_path):
        vid_path = Path(vid_path)
        assert vid_path.exists(), f'video file {vid_path} does not exist'
        orm = ORM()
        strike_frame_ids = []
        with orm.session() as s:
            vid = s.query(Video).filter(Video.path.contains(vid_path.name)).first()
            if vid is None:
                raise Exception(f'unable to find video {vid_path.name}')
            start_ts, stop_ts = self.get_video_start_stop(vid)
            frames_array = np.array(list(vid.frames.values()))
            blk = s.query(Block).filter_by(id=vid.block_id).first()
            for strk in blk.strikes:
                strk_ts = datetime.timestamp(strk.time)
                if start_ts <= strk_ts <= stop_ts:
                    frame_id = int(np.argmin(np.abs(frames_array - strk_ts)))
                    strike_frame_ids.append(frame_id)
        return strike_frame_ids

    @staticmethod
    def get_video_start_stop(vid: Video):
        return vid.frames['0'], vid.frames[list(vid.frames.keys())[-1]]


def check_dataset(input_dir=TONGUE_PREDICTED_DIR, is_load=True, min_dist=15, is_delete=False):
    images = []
    input_dir = Path(input_dir)
    img_files = list(input_dir.glob('*.jpg'))
    for p in tqdm(img_files, desc='images read'):
        img = cv2.imread(p.as_posix(), 0)
        if img.shape != (1088, 1456):
            shutil.move(p.as_posix(), (p.parent / 'bad_images').as_posix())
            continue
        img = cv2.resize(img, None, None, fx=0.5, fy=0.5)
        images.append(img)

    groups_dir = input_dir / 'grouped'
    groups_dir.mkdir(exist_ok=True)
    n = len(images)
    cache_file = f'{groups_dir}/m.npz'
    if not is_load:
        with tqdm(total=n, desc='create dist matrix') as pbar:

            def iter1(*l):
                M_ = np.zeros((n, n))
                for i in l:
                    for j in range(i + 1, n):
                        M_[i, j] = np.mean(np.abs(images[i] - images[j]))
                    pbar.update(1)
                return M_

            with ThreadPool() as pool:
                M = pool.starmap(iter1, np.array_split(np.arange(n).astype(int), 10))
            M = np.sum(M, axis=0)
            np.savez(cache_file, M=M)
    else:
        M = np.load(cache_file)['M']

    r, c = np.where((0 < M) & (M < min_dist))

    def get_neighs(i):
        idx = np.where(r == i)[0]
        return c[idx]

    processed_ids = set()
    groups = []
    for i, k in enumerate(tqdm(np.unique(r), desc='grouping frames')):
        if k in processed_ids:
            continue
        neighs = get_neighs(k)
        g = [k] + neighs.tolist()
        processed_ids.update(g)
        for j, gi in enumerate(g):
            p = img_files[gi]
            if not p.exists():
                print(f'File {p.name} does not exist')
                continue
            if is_delete:
                if j > 0:
                    p.unlink()
                    print(f'deleted {p.name}')
            else:
                group_dir_ = groups_dir / str(i)
                group_dir_.mkdir(exist_ok=True)
                shutil.move(p.as_posix(), group_dir_.as_posix())
        groups.append(g)


def clean_wrong_size_images():
    input_dir = Path(DATASET_PATH)
    img_files = list(input_dir.rglob('*.jpg'))
    for p in img_files:
        img = cv2.imread(p.as_posix(), 0)
        if img.shape != (1088, 1456):
            p.unlink()
            print(f'deleted {p} - {img.shape}')


if __name__ == '__main__':
    # vidpath = '/data/Pogona_Pursuit/output/experiments/PV80/20221213/block3/videos/front_20221213T101615.mp4'
    # vidpath = '/data/Pogona_Pursuit/output/experiments/PV80/20221216/block1/videos/front_20221216T155149.mp4'
    # vidpath = '/data/Pogona_Pursuit/output/experiments/PV80/20221215/block2/videos/front_20221215T183722.mp4'
    # tde = TonguesOutVideoAnalyzer(vidpath)
    # tde.predict_video()

    # check_dataset('/data/Pogona_Pursuit/output/datasets/pogona_tongue/dataset/tongues', is_load=False, min_dist=15, is_delete=True)
    # clean_wrong_size_images()

    # TongueTrainer().train(is_plot=True, is_save=True)

    tr = TongueTrainer(model_path=MODEL_PATH, threshold=0.5)
    tr.all_data_evaluation()
    # tr.plot_confusion_matrix()

    # target_ = 'tongues'
    # res = {}
    # images_ = list(Path(f'/data/Pogona_Pursuit/output/datasets/pogona_tongue/dataset/{target_}').glob('*.jpg'))
    # for p in tqdm(images_):
    #     frame_ = cv2.imread(p.as_posix())
    #     label, prob = tr.predict(frame_)
    #     if label != target_:
    #         res[p.stem] = prob
    #
    # print(res)
    # print(f'{len(res)}/{len(images_)} have bad result')