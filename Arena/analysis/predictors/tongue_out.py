import cv2
import shutil
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from tqdm.auto import tqdm
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Lambda, Grayscale, ToPILImage
from torchvision.transforms.functional import crop
import matplotlib.pyplot as plt
if Path('.').resolve().name != 'Arena':
    import os
    os.chdir('../..')
from analysis.image_embedding import ResNetPretrained
from analysis.predictors.base import Predictor
from db_models import ORM, Block, Video
from analysis.trainer import ClassificationTrainer
from analysis.strikes.loader import Loader
from analysis.pose_utils import put_text

DATASET_PATH = Path('/data/Pogona_Pursuit/output/datasets/pogona_tongue/dataset')
TONGUE_PREDICTED_DIR = '/data/Pogona_Pursuit/output/datasets/pogona_tongue/predicted/tongues'
CROPPED_SHAPE = (480, 850)
TONGUE_CLASS = 'tongues'
PREDICTION_STACK_DURATION = 0.25  # sec
TONGUE_ACTION_TIMEOUT = 1  # sec
NUM_TONGUES_IN_STACK = 6  # number of predicted tongues in the prediction stack to trigger action


class TongueOutAnalyzer(Predictor):
    def __init__(self, td=None, action_callback=None, identifier=None, is_write_detected_image=False,
                 is_debug=True):
        super(TongueOutAnalyzer, self).__init__()
        self.tr = TongueTrainer(model_path=self.model_path, is_debug=is_debug) if td is None else td
        self.identifier = identifier
        self.action_callback = action_callback
        self.is_debug = is_debug
        self.is_write_detected_image = is_write_detected_image
        self.last_action_timestamp = None
        self.last_tongue_detect_timestamp = None
        self.predictions_stack = []
        self.last_predicted_frame = None

    def predict(self, frame, timestamp):
        orig_frame = frame.copy()
        label, prob = self.tr.predict(frame)
        self.push_to_predictions_stack(label, timestamp)
        is_tongue = label == TONGUE_CLASS and prob >= self.pred_config.get('threshold')
        is_action = self.tongue_detected(orig_frame, timestamp) if is_tongue else False

        if self.pred_config.get('resize_image'):
            frame = cv2.resize(frame, self.pred_config['resize_image'][::-1])
        frame = put_text(f'P={prob:.2f}', frame, frame.shape[1] - 120, 30, color=(255, 0, 0))

        return is_action, frame, label

    def predict_strike(self, strike_db_id, n_before=30, n_after=10, cols=8, save_frames_above=None):
        ld = Loader(strike_db_id, 'front', is_debug=False)
        frame_ids = []
        n = n_before + n_after - 1
        rows = int(np.ceil(n/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(30, 3*rows))
        axes = axes.flatten()
        for i, (frame_id, frame) in enumerate(ld.gen_frames_around_strike(n_before, n_after)):
            label, prob = self.tr.predict(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            if save_frames_above and prob > save_frames_above:
                cv2.imwrite(f'{TONGUE_PREDICTED_DIR}/{strike_db_id}_{frame_id}.jpg', frame)

            h, w = frame.shape[:2]
            frame = frame[h // 2:, 350:w - 350]
            axes[i].imshow(frame)
            axes[i].set_xticks([]); axes[i].set_yticks([])
            axes[i].set_title(frame_id)
            curr_y = 60
            if label == TONGUE_CLASS:
                axes[i].text(30, curr_y, f'tongue (P={prob:.2f})', color='green')
            else:
                axes[i].text(30, curr_y, f'P={prob:.3f}', color='orange')
            curr_y += 70
            if frame_id == ld.strike_frame_id:
                axes[i].text(30, curr_y, 'Strike Frame', color='red')
            frame_ids.append(frame_id)
        plt.show()

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


def crop_transform(img):
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
    elif isinstance(img, torch.Tensor):
        h, w = img.shape[1:]
    else:
        w, h = img.size[:2]
    w0 = (w // 2) - (CROPPED_SHAPE[1] // 2)
    return crop(img, h - CROPPED_SHAPE[0], w0, CROPPED_SHAPE[0], CROPPED_SHAPE[1])


@dataclass
class TongueTrainer(ClassificationTrainer):
    cropped_shape: tuple = CROPPED_SHAPE
    batch_size = 16
    num_epochs = 30
    targets = ['no_tongues', 'tongues']
    monitored_metric = 'auc'

    def get_dataset(self):
        transforms = [
            Grayscale(),
            Lambda(crop_transform),
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ]
        dataset = ImageFolder(DATASET_PATH.as_posix(), transform=Compose(transforms))
        info = pd.Series(dataset.targets).value_counts().rename({i: v for i, v in enumerate(self.targets)}).to_dict()
        print(f'Loaded tongue dataset with: {info}')
        return dataset

    def get_model(self):
        return TongueModel()

    def predict(self, frame):
        torch.cuda.empty_cache()
        self.model.eval()
        img_tensor = Compose([
            ToPILImage(),
            Lambda(crop_transform),
            Grayscale(),
            ToTensor(),
        ])(frame).to(self.device)
        img_tensor = Compose([
            Normalize((0.5,), (0.5,))
        ])(img_tensor)
        # img_tensor = Lambda(crop_transform)(img_tensor)
        # transforms = [ToPILImage()] + self.transforms
        # img_tensor = Compose(transforms)(frame).to(self.device)
        outputs = self.model(img_tensor.unsqueeze(0))
        y_pred, y_score = self.predict_proba(outputs)
        label = self.targets[y_pred.item()]
        prob = y_score.item()
        return label, prob


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
    for i, k in enumerate(tqdm(np.unique(r), desc='grouping frames', total=len(np.unique(r)))):
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

    # check_dataset(is_load=True, min_dist=35, is_delete=True)
    # clean_wrong_size_images()

    # TongueTrainer().train(is_plot=True, is_save=True)

    toa = TongueOutAnalyzer()
    toa.predict_strike(8432, save_frames_above=0.5)

    # tr = TongueTrainer(model_path=MODEL_PATH, threshold=0.5)
    # tr.all_data_evaluation()
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