import cv2
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
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
from analysis.image_embedding import ResNetPretrained
from db_models import ORM, Experiment, Block, Video, VideoPrediction, Strike, Trial

DATASET_PATH = Path('/data/Pogona_Pursuit/output/datasets/pogona_tongue/dataset')
TONGUE_PREDICTED_DIR = '/data/Pogona_Pursuit/output/datasets/pogona_tongue/predicted/tongues'
# MODEL_PATH = '/data/Pogona_Pursuit/output/models/20221215_091700'
MODEL_PATH = '/data/Pogona_Pursuit/output/models/20221216_170652'
# MODEL_PATH = '/data/Pogona_Pursuit/output/models/20221217_174652'
THRESHOLD = 0.95
TONGUE_CLASS = 'tongues'


class Classifier(nn.Module):
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


class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X.iloc[idx].values, self.y[idx]


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = dict()
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


class TongueDetector:
    def __init__(self, model_path=MODEL_PATH, use_cv2_resize=True, threshold=THRESHOLD):
        self.model = Classifier()
        self.label_encoder = None
        self.use_cv2_resize = use_cv2_resize
        self.resize_dim = (480, 640)
        self.threshold = threshold
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transforms = [
            ToPILImage(),
            Resize(self.resize_dim),
            ToTensor(),
            # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            Normalize((0.5,), (0.5,))
        ]
        if model_path:
            self.load(model_path)

    def inference_init(self):
        self.model.to(torch.device('cpu'))
        self.model.eval()
        im = torch.empty((1, 1, *self.resize_dim), dtype=torch.float, device=torch.device('cpu'))
        self.model(im)
        self.model.to(self.device)
        return self

    @torch.no_grad()
    def predict_image(self, frame: np.ndarray) -> (str, np.ndarray):
        img_tensor, frame = self.transform_frame(frame)
        res = self.model(img_tensor)
        predicted = self._predict(res)
        label = self.index_to_label(predicted.item())
        return label, frame

    def transform_frame(self, frame: np.ndarray):
        if self.use_cv2_resize:
            frame = cv2.resize(frame, self.resize_dim[::-1])
            img_tensor = torch.as_tensor(frame)
            img_tensor = img_tensor.to(self.device)
            img_tensor = img_tensor.unsqueeze(0)  # 1 channel (grey)
            img_tensor = (img_tensor - 127.5) / 127.5
            img_tensor = Normalize((0.5,), (0.5,))(img_tensor)
        else:
            trans_ = self.transforms
            img_tensor = Compose(trans_)(frame)

        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor, frame

    def _predict(self, outputs: torch.Tensor):
        p = F.softmax(outputs, dim=1)
        pmax, predicted = torch.max(p, dim=1)
        predicted[pmax < self.threshold] = self.label_encoder['no_tongues']
        return predicted

    def index_to_label(self, idx):
        for key, i in self.label_encoder.items():
            if i == idx:
                return key

    def train(self, num_epochs=30, is_plot=False):
        best_accuracy, best_model_ = 0.0, None
        train_loader, test_loader = self.get_loaders()
        self.model = Classifier()
        self.model.to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
        print("The model will be running on", self.device, "device")
        metrics = {'loss': [], 'accuracy': []}
        with tqdm(range(num_epochs)) as pbar:
            for _ in pbar:
                mean_loss = self.train_epoch(train_loader, optimizer, loss_fn)
                accuracy = self.test_accuracy(test_loader)
                pbar.desc = f'Accuracy={accuracy:.1f}% (Best={best_accuracy:.1f}%)'
                metrics['loss'].append(mean_loss)
                metrics['accuracy'].append(accuracy)
                if accuracy > best_accuracy:
                    best_model_ = self.model.state_dict()
                    best_accuracy = accuracy

        self.model.load_state_dict(best_model_)
        self.save_model()
        if is_plot:
            self.plot_train_metrics(metrics, num_epochs)
        return self

    def train_epoch(self, train_loader, optimizer, loss_fn):
        """run a train epoch and return the mean loss"""
        loss_ = []
        self.model.train()
        for i, (embs, labels) in enumerate(train_loader, 0):
            embs = embs.to(self.device)
            labels = labels.to(self.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = self.model(embs)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # back-propagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()
            loss_.append(loss.item())
        return np.mean(loss_)

    def get_loaders(self):
        dataset = ImageFolder(DATASET_PATH.as_posix(), transform=Compose([Grayscale()] + self.transforms[1:]))
        self.label_encoder = dataset.class_to_idx
        datasets = train_val_dataset(dataset, val_split=0.25)
        self.print_dataset_info(datasets)

        params = {'batch_size': 16, 'shuffle': True, 'num_workers': 6, 'drop_last': True}
        train_loader = DataLoader(datasets['train'], **params)
        test_loader = DataLoader(datasets['val'], **params)
        return train_loader, test_loader

    @staticmethod
    def print_dataset_info(datasets: dict):
        for key, dataset in datasets.items():
            s = pd.Series([y[1] for y in dataset.samples]).value_counts()
            s.index = s.index.map({v: k for k, v in dataset.class_to_idx.items()})
            text = ', '.join(f'{k}: {v}' for k, v in s.iteritems())
            print(f'{key} dataset distribution: {text}')

    @staticmethod
    def load_labels():
        classes_names = ['tongues', 'no_tongues']
        print(f'loading labels: {classes_names} from directory {DATASET_PATH}')
        classes = {}
        for cl in classes_names:
            img_names = [p.stem for p in (DATASET_PATH / cl).glob('*.jpg')]
            for img_name in img_names:
                classes[img_name] = cl
        return classes

    def load(self, model_path):
        model_path = Path(model_path)
        assert model_path.exists() and model_path.is_dir(), f'{model_path} does not exist or not a directory'
        self.model.load_state_dict(torch.load(model_path / 'clf.pth'))
        # self.model.to(self.device)
        self.label_encoder = torch.load(model_path / 'label_encoder.pth')

    def save_model(self):
        dir_path = Path(f"/data/Pogona_Pursuit/output/models/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        dir_path.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), dir_path / 'clf.pth')
        torch.save(self.label_encoder, dir_path / 'label_encoder.pth')
        print(f'model saved to {dir_path}')

    @staticmethod
    def plot_train_metrics(metrics, num_epochs):
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        epochs = np.arange(1, num_epochs + 1)
        for i, key in enumerate(['loss', 'accuracy']):
            axes[i].plot(epochs, metrics[key])
            axes[i].set_title(key)
        fig.tight_layout()
        plt.show()

    def test_accuracy(self, test_loader):
        self.model.eval()
        accuracy, total = 0.0, 0.0
        with torch.no_grad():
            for data in test_loader:
                images, labels = [x.to(self.device) for x in data]
                # run the model on the test set to predict labels
                outputs = self.model(images)
                # the label with the highest energy will be our prediction
                predicted = self._predict(outputs)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()

        # compute the accuracy over all test images
        accuracy = (100 * accuracy / total)
        return accuracy


PREDICTION_STACK_DURATION = 0.25  # sec
TONGUE_ACTION_TIMEOUT = 1  # sec
NUM_TONGUES_IN_STACK = 6  # number of predicted tongues in the prediction stack to trigger action


class TongueOutAnalyzer:
    def __init__(self, td=None, action_callback=None, identifier=None, is_write_detected_image=False):
        self.td = TongueDetector(use_cv2_resize=True).inference_init() if td is None else td
        self.identifier = identifier
        self.action_callback = action_callback
        self.is_write_detected_image = is_write_detected_image
        self.last_action_timestamp = None
        self.last_tongue_detect_timestamp = None
        self.predictions_stack = []
        self.last_predicted_frame = None

    def predict(self, frame, timestamp):
        label, resized_frame = self.td.predict_image(frame)
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


def check_dataset(input_dir=TONGUE_PREDICTED_DIR, is_load=True, max_dist=15, is_delete=False):
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

    r, c = np.where((0 < M) & (M < max_dist))

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
    import os
    os.chdir('..')
    # td = TongueDetector(use_cv2_resize=True)
    # td.train(is_plot=True, num_epochs=70)
    # vidpath = '/data/Pogona_Pursuit/output/experiments/PV80/20221213/block3/videos/front_20221213T101615.mp4'
    # vidpath = '/data/Pogona_Pursuit/output/experiments/PV80/20221216/block1/videos/front_20221216T155149.mp4'
    vidpath = '/data/Pogona_Pursuit/output/experiments/PV80/20221215/block2/videos/front_20221215T183722.mp4'
    tde = TonguesOutVideoAnalyzer(vidpath)
    tde.predict_video()

    # check_dataset('/data/Pogona_Pursuit/output/datasets/pogona_tongue/dataset/tongues', is_load=False, max_dist=10, is_delete=False)
    # clean_wrong_size_images()