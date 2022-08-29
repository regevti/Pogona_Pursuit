from pathlib import Path
from datetime import datetime
import shutil
import time
import cv2
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms, models
from dataclasses import dataclass, field
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, precision_recall_fscore_support
from scipy.signal import find_peaks
from pose_analysis.strike_predictor.video_dataset import VideoFrameDataset, ImglistToTensor

EXPERIMENTS_DIR = '/media/sil2/Data/regev/Pogona_Pursuit/Arena/day_experiments'


@dataclass
class StrikePredictor:
    train_dir: str
    segment_length: int
    is_save_embedding: bool = True
    is_save_model: bool = True
    threshold: float = 0.9
    batch_size: int = 32
    n_layers: int = 15
    n_epochs: int = 150
    n_classes: int = 0
    hidden_dim: int = 512
    dropout: float = 0.3
    lr: float = 1e-5
    test_size: float = 0.2
    infer_delay: float = 1.0  # seconds
    is_debug: bool = True
    model_id: str = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    rnn_model = None

    def __post_init__(self):
        self.embed_model = ResNetPretrained()
        self.model_id = self.model_id + f'_seg{self.segment_length}'

    def train(self):
        train_ld, val_ld, self.n_classes = self.get_loaders()
        self.rnn_model = HuntStateTagger(n_classes=self.n_classes, n_layers=self.n_layers,
                                         hidden_dim=self.hidden_dim, dropout=self.dropout)
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(self.rnn_model.parameters(), lr=self.lr)

        M = {'train_loss': [], 'val_loss': [], 'accuracy': [], 'best_acc': 0}
        best_model_state = self.rnn_model.state_dict()
        for _ in (tqdm(range(self.n_epochs)) if self.is_debug else range(self.n_epochs)):
            train_loss = self.train_epoch(train_ld, loss_function, optimizer)
            M['train_loss'].append(train_loss)
            val_loss, y_pred, y_true, accuracy = self.val_epoch(val_ld, loss_function)
            M['val_loss'].append(val_loss)
            M['accuracy'].append(accuracy)
            if accuracy > M['best_acc']:
                best_model_state = self.rnn_model.state_dict()
                M['best_acc'] = accuracy
        self.rnn_model.load_state_dict(best_model_state)
        if self.is_save_model:
            self.save()
        if self.is_debug:
            self.plot_train_metrics(M)
            self.evaluate()
        return M["best_acc"]

    def train_epoch(self, train_ld, loss_function, optimizer):
        sum_loss, total = 0.0, 0
        self.rnn_model.train()
        for inputs, y in train_ld:
            current_batch_size = y.shape[0]
            self.rnn_model.zero_grad()  # zero the parameter gradients
            outputs, p = self.rnn_model(inputs.cuda())
            loss = loss_function(outputs, y.squeeze().cuda())
            loss.backward()
            optimizer.step()
            sum_loss += current_batch_size * (loss.item())
            total += current_batch_size
        return sum_loss / total

    def val_epoch(self, val_ld, loss_function):
        sum_loss, total, correct = 0.0, 0, 0
        self.rnn_model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for inputs, y in val_ld:
                current_batch_size = y.shape[0]
                outputs, p = self.rnn_model(inputs.cuda())
                loss = loss_function(outputs, y.squeeze().cuda())
                pred = torch.max(outputs, 1)[1].cpu()
                y_pred.append(pred.numpy())
                y_true.append(y.numpy())
                sum_loss += current_batch_size * (loss.item())
                total += current_batch_size
        y_pred = np.hstack(y_pred)
        y_true = np.vstack(y_true).squeeze()
        accuracy = accuracy_score(y_true, y_pred)
        return sum_loss / total, y_pred, y_true, accuracy

    def evaluate(self):
        X_embed, y_embed = self.embed(is_encode_label=False)
        dataset = TensorDataset(torch.Tensor(X_embed), torch.Tensor(y_embed).type(torch.LongTensor))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.rnn_model.eval()
        y_pred, y_true_multi, y_score = [], [], []
        with torch.no_grad():
            for inputs, y in loader:
                outputs, p = self.rnn_model(inputs.cuda())
                pred = torch.max(outputs, 1)[1].cpu()
                y_pred.append(pred.numpy())
                y_true_multi.append(y.numpy())
                y_score.append(p.cpu().numpy()[:, 0])
        y_pred = np.hstack(y_pred)
        y_true_multi = np.vstack(y_true_multi).squeeze()
        y_pred_multi = y_pred.copy()
        idx1 = np.where((y_pred == 1) & (y_true_multi > 0))[0]
        y_pred_multi[idx1] = y_pred[idx1] * y_true_multi[idx1]
        y_true_binary = self.encode_label(y_true_multi.copy())
        y_score = np.hstack(y_score)

        # prfs = precision_recall_fscore_support(y_true_multi, y_pred_multi)

        fig, axes = plt.subplots(1, 2, figsize=(18, 4))
        plot_roc_curve(axes[0], y_true_binary, y_score)
        plot_precision_recall_curve(axes[1], y_true_binary, y_score)
        plt.show()
        print(classification_report(y_true_multi, y_pred_multi, labels=np.unique(y_true_multi).tolist()))

    def embed(self, frame=None, is_encode_label=True):
        if frame is not None:
            frame = self.preprocess_transform()(frame)
            return self.embed_model.embed_frame(frame.unsqueeze(0))

        if not Path(self.saved_embedded_dataset_path).exists():
            loader = torch.utils.data.DataLoader(dataset=self.get_emb_dataset(), batch_size=1, shuffle=True,
                                                 num_workers=4, pin_memory=False)
            print('Start embedding of train set...')
            X_embed, y_embed = self.embed_model.embed(loader)
            torch.cuda.empty_cache()
            if self.is_save_embedding:
                self.save_embedded_dateset(X_embed, y_embed)
        X_embed, y_embed = self.load_embedded_dataset()
        if is_encode_label:
            y_embed = self.encode_label(y_embed)
        if self.is_debug:
            print(f'Finished embedding. X_embed: {X_embed.shape}, labels: {y_embed.shape}')
        return X_embed, y_embed

    def get_loaders(self):
        X_embed, y_embed = self.embed()
        X_train, X_val, y_train, y_val = train_test_split(X_embed, y_embed, test_size=self.test_size, random_state=42)
        train_ds = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).type(torch.LongTensor))
        val_ds = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val).type(torch.LongTensor))
        train_ld = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_ld = DataLoader(val_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        n_classes = len(np.unique(y_embed))
        return train_ld, val_ld, n_classes

    @staticmethod
    def encode_label(y_embed):
        idx = np.where(y_embed != 0)
        y_embed[idx] = 1
        return y_embed

    def predict_video(self, video_path, is_save_embedded=True):
        t_all = time.time()
        embed_path = self.get_embedded_video_save_path(video_path)
        is_embed = is_save_embedded and Path(embed_path).exists()
        vid = cv2.VideoCapture(video_path)
        fps = vid.get(cv2.CAP_PROP_FPS)
        if not is_embed:
            emb_durations, X_embed = [], []
            n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            X_embed = torch.load(embed_path)
            n_frames = X_embed.shape[0]

        X, rnn_durations = [], []
        probs_df = pd.DataFrame(columns=['start_frame', 'end_frame', 'prob'])
        for i in (tqdm(range(n_frames)) if self.is_debug else range(n_frames)):
            if not is_embed:
                ret, frame = vid.read()
                t0 = time.time()
                X.append(self.embed(frame))
                emb_durations.append(time.time() - t0)
            else:
                X.append(X_embed[i, :, :].unsqueeze(0))
            if len(X) >= self.segment_length:
                t0 = time.time()
                label, probs = self.predict_segment(X)
                rnn_durations.append(time.time() - t0)
                probs_df = pd.concat([probs_df, pd.DataFrame({'start_frame': i - self.segment_length, 'end_frame': i,
                                                              'prob': probs[0]}, index=[0])])
                x = X.pop(0)
                if not is_embed:
                    X_embed.append(x)

        if not is_embed and is_save_embedded:
            X_embed.extend(X)
            torch.save(torch.vstack(X_embed), embed_path)
        probs_df = self.add_real_strikes(Path(video_path).name, probs_df).reset_index(drop=True)
        probs_df = self.calc_predicted_strike_frames(probs_df, fps)
        if self.is_debug:
            if not is_embed:
                print(f'Embedding Duration: {np.mean(emb_durations) * 1000:.1f} ms')
            print(f'RNN Duration: {np.mean(rnn_durations) * 1000:.1f} ms')
            print(f'Time taken for {n_frames} frames: {(time.time()-t_all)/60:.1f} minutes')
        return probs_df

    def get_embedded_video_save_path(self, video_path: str) -> str:
        video_path = Path(video_path)
        return (video_path.parent / f'{video_path.stem}_embed_{self.embed_model.backbone}.pt').as_posix()

    def predict_segment(self, seg: list):
        if len(seg) != self.segment_length:
            print(f'ERROR. segment length must be {self.segment_length}, found: {len(seg)}')
        with torch.no_grad():
            outputs, p = self.rnn_model(torch.hstack(seg))
            pred = torch.max(outputs, 1)[1].squeeze().cpu().numpy()
            probs = p.squeeze().cpu().numpy()
        return pred, probs

    def calc_predicted_strike_frames(self, probs_df, fps):
        frames_delay = int(fps * self.infer_delay)
        delay_end = 0
        probs_df['predicted_strike'] = [0] * len(probs_df)
        for i, row in probs_df.query(f'prob>={self.threshold}').copy().iterrows():
            if row.end_frame <= delay_end:
                continue
            probs_df.loc[i, 'predicted_strike'] = 1
            delay_end = row.end_frame + frames_delay
        return probs_df

    @staticmethod
    def add_real_strikes(vid_name, probs_df):
        probs_df['true_strike'] = [0] * len(probs_df)
        all_strikes = pd.read_csv(f'{EXPERIMENTS_DIR}/all_strikes.csv', index_col=0)
        real_strikes = all_strikes.query(f'video_name=="{vid_name}"')
        for frame_id in real_strikes.strike_frame.values:
            if frame_id in probs_df.end_frame.values:
                probs_df.loc[probs_df.end_frame == frame_id, 'true_strike'] = 1
        return probs_df

    def plot_predicted_video(self, prob_df):
        xs, ys = prob_df.end_frame.values.astype(float), prob_df.prob.values.astype(float)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(xs, ys)
        for frame_id in prob_df.query('true_strike==1').end_frame.values:
            ax.axvline(frame_id, color='crimson', linestyle='--')
        for frame_id in prob_df.query('predicted_strike==1').end_frame.values:
            ax.axvline(frame_id, color='darkgreen', linestyle='--')
        ax.fill_between(xs, ys, where=ys >= self.threshold, interpolate=True, color='darkorange', alpha=0.4)
        ax.fill_between(xs, ys, where=ys <= self.threshold, interpolate=True, color='lightblue', alpha=0.4)

    def get_emb_dataset(self):
        return VideoFrameDataset(
            root_path=self.train_dir,
            annotationfile_path=f'{self.train_dir}/annotations.txt',
            num_segments=1,
            frames_per_segment=118, # maximal lenght
            imagefile_template='img_{:05d}.jpg',
            transform=self.preprocess_transform(is_loader=True),
            is_match_annotations=True
        )

    @staticmethod
    def preprocess_transform(is_loader=False):
        if is_loader:
            t = [ImglistToTensor()]
        else:
            t = [transforms.ToPILImage(), transforms.ToTensor()]
        return transforms.Compose(t)

    @staticmethod
    def plot_train_metrics(metrics):
        fig, axes = plt.subplots(1, 2, figsize=(20, 4))
        axes[0].plot(metrics['train_loss'], label='train')
        axes[0].plot(metrics['val_loss'], label='validation')
        axes[0].legend()
        axes[0].set_title('loss')
        axes[1].plot(metrics['accuracy'])
        axes[1].set_title('Accuracy')
        fig.tight_layout()

    def save_embedded_dateset(self, X_embed, y_embed):
        with open(self.saved_embedded_dataset_path, 'wb') as f:
            pickle.dump({'X_embed': X_embed, 'y_embed': y_embed}, f)

    def load_embedded_dataset(self):
        with open(self.saved_embedded_dataset_path, 'rb') as f:
            d = pickle.load(f)
            X_embed = d['X_embed'][:, -self.segment_length:, :]
            y_embed = d['y_embed']
            del d
        return X_embed, y_embed

    def save(self):
        d = {k: getattr(self, k) for k in self.model_attributes}
        d['model_state_dict'] = self.rnn_model.state_dict()
        torch.save(d, self.saved_model_path)
        if self.is_debug:
            print(f'Model saved to {self.saved_model_path}')

    def load(self, path=None):
        d = torch.load(path or self.saved_model_path)
        state_dict = d.pop('model_state_dict')
        for k, v in d.items():
            setattr(self, k, v)
        self.rnn_model = HuntStateTagger(n_classes=self.n_classes, n_layers=self.n_layers,
                                         hidden_dim=self.hidden_dim, dropout=self.dropout)
        self.rnn_model.load_state_dict(state_dict)
        self.rnn_model.eval()

    @property
    def saved_embedded_dataset_path(self):
        return f'{self.train_dir}/embedded_cache_{self.embed_model.backbone}.pkl'

    @property
    def saved_model_path(self):
        return f'{self.train_dir}/{self.model_id}.pt'

    @property
    def model_attributes(self):
        return [i for i in self.__dict__.keys() if i[:1] != '_' and i not in ['rnn_model', 'embed_model']]


class ResNetPretrained(nn.Module):
    def __init__(self, rescale_size=(224, 224), backbone='resnet50'):
        super().__init__()
        self.backbone = backbone
        model = getattr(models, self.backbone)(pretrained=True)
        # model = models.efficientnet_b4(pretrained=True)
        model.float()
        model.cuda()
        model.eval()
        module_list = list(model.children())
        self.conv5 = nn.Sequential(*module_list[:-2])
        self.pool5 = module_list[-2]
        self.transformer = transforms.Compose([
            transforms.Resize(rescale_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        x = self.transformer(x).cuda()
        res5c = self.conv5(x)
        pool5 = self.pool5(res5c)
        pool5 = pool5.view(pool5.size(0), -1)
        return res5c, pool5

    def embed(self, loader):
        X, y = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(loader):
                X.append(self.embed_frame(inputs.squeeze()))
                y.append(labels)
            return torch.vstack(X).cpu().numpy(), torch.vstack(y).cpu().numpy()

    def embed_frame(self, frame):
        with torch.no_grad():
            if not torch.is_tensor(frame):
                frame = torch.Tensor(frame)
            _, x_emb = self(frame.cuda())
        return x_emb.unsqueeze(0)


class HuntStateTagger(nn.Module):
    def __init__(self, n_classes, n_layers=2, hidden_dim=256, dropout=0.2, embedding_dim=2048):
        super().__init__()
        self.device = torch.device('cuda:0')
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.recurrent = nn.GRU(self.embedding_dim, hidden_dim, n_layers,
                                dropout=dropout, batch_first=True).to(self.device)
        self.hidden2tag = nn.Linear(hidden_dim, n_classes).to(self.device)

    def forward(self, x):
        rnn_out, h = self.recurrent(x)
        y = self.hidden2tag(F.relu(rnn_out[:,-1]))
        out = F.log_softmax(y, dim=1)
        p = F.softmax(y, dim=1)
        return out, p

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden


def plot_roc_curve(ax, y_true, y_score):
    # swap 0->1 and 1->0 to match the roc_curve metrics condition
    y_true1 = np.where((y_true==0)|(y_true==1), y_true^1, y_true)
    fpr, tpr, _ = roc_curve(y_true1, y_score)
    auc = roc_auc_score(y_true1, y_score)
    ax.plot(fpr, tpr)
    ax.set_title('ROC Curve')
    ax.set_ylabel('True-Positive Rate')
    ax.set_xlabel('False-Positive Rate')
    ax.text(0.8, 0.1, f'AUC={auc:.2f}')


def plot_precision_recall_curve(ax, y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    ax.plot(recall, precision)
    ax.set_title('PR Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
