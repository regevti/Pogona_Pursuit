import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.optim.lr_scheduler as lr
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
from captum.attr import GradientShap
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import explained_variance_score, roc_auc_score, balanced_accuracy_score, precision_score, \
    recall_score, confusion_matrix, PrecisionRecallDisplay, RocCurveDisplay

SAVED_MODEL_DIR = '/data/Pogona_Pursuit/output/models'


@dataclass
class Trainer:
    model_path: str = None
    seed: int = 42
    is_debug: bool = True
    batch_size: int = 16
    threshold: float = 0.8
    num_epochs: int = 30
    kfolds: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    monitored_metric: str = 'val_loss'
    monitored_metric_algo: str = 'min'
    is_shuffle_dataset: bool = True
    cache_dir: Path = None

    def __post_init__(self):
        assert self.monitored_metric_algo in ['min', 'max'], f'monitored_metric_algo must be either "min" or "max"'
        torch.manual_seed(self.seed)
        self.model_name = self.get_model_name()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        if self.model_path:
            self.load()
            self.dataset = None
        else:
            self.dataset = self.get_dataset()

    def get_model_name(self):
        model_name = self.__module__
        if model_name == '__main__':
            filename = sys.modules[self.__module__].__file__
            model_name = os.path.splitext(os.path.basename(filename))[0]
        return model_name

    def train_val_dataset(self, dataset, val_split=0.2):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, random_state=self.seed)
        datasets = dict()
        datasets['train'] = Subset(dataset, train_idx)
        datasets['val'] = Subset(dataset, val_idx)
        return datasets

    def train(self, is_plot=False, is_save=True):
        self.print(f"Start train for model {self.model_name}. Train will be running on {self.device} device")
        history = []
        splits = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.seed)
        indices = np.arange(len(self.dataset))
        if self.is_shuffle_dataset:
            indices = np.random.permutation(indices)
        for fold, (train_idx, val_idx) in enumerate(splits.split(indices)):
            f_best_score_, f_best_model_, f_metrics = None, None, dict()
            self.model = self.get_model()
            self.model.to(self.device)
            loss_fn = self.get_loss_fn()
            optimizer = self.get_optimizer()
            scheduler = self.get_scheduler(optimizer)
            train_loader, val_loader = self.get_loaders(indices[train_idx], indices[val_idx])
            with tqdm(range(self.num_epochs)) as pbar:
                for _ in pbar:
                    epoch_metrics = dict()
                    epoch_metrics['train_loss'] = self.train_epoch(train_loader, optimizer, loss_fn)
                    epoch_metrics.update(self.val_epoch(val_loader, loss_fn))
                    score = epoch_metrics[self.monitored_metric]
                    if scheduler is not None:
                        scheduler.step(score)
                    pbar.desc = f'FOLD-#{fold+1}  {self.monitored_metric}={score:.2f} (best={f_best_score_ or 0:.2f})'
                    for k, v in epoch_metrics.items():
                        f_metrics.setdefault(k, []).append(v)
                    if not f_best_score_ or self.is_better_score_(score, f_best_score_):
                        f_best_score_ = score
                        f_best_model_ = self.model.state_dict()
            self.model.load_state_dict(f_best_model_)
            history.append({'model_state': f_best_model_, 'score': f_best_score_, 'metrics': f_metrics})

        chosen_idx = self.get_best_model(history)
        self.print(f'Chosen model is of Fold#{chosen_idx+1}')
        self.model.load_state_dict(history[chosen_idx]['model_state'])
        if is_save:
            self.save_model()
        if is_plot:
            self.summary_plots(history, chosen_idx)
        return self

    def train_epoch(self, train_loader, optimizer, loss_fn):
        """run a train epoch and return the mean loss"""
        loss_ = []
        self.model.train()
        for i, (x, y) in enumerate(train_loader, 0):
            x, y = self.convert_inputs(x, y)
            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = self.model(x)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, y)
            # back-propagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()
            loss_.append(loss.item())
        return np.mean(loss_)

    @torch.no_grad()
    def val_epoch(self, val_loader, loss_fn) -> dict:
        self.model.eval()
        metrics, y_true, y_pred, losses = {}, [], [], []
        for x, y in val_loader:
            x, y = self.convert_inputs(x, y)
            pred = self.model(x)
            loss = loss_fn(pred, y)
            losses.append(loss.item())
            y_pred.append(pred)
            y_true.append(y)
        y_true, y_pred = torch.concat(y_true), torch.concat(y_pred)
        metrics['val_loss'] = np.mean(losses)
        metrics.update(self.evaluate(y_true, y_pred))
        return metrics

    def evaluate(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        return {}

    def convert_inputs(self, x, y):
        return x.to(self.device).float(), y.to(self.device).float()

    def get_loaders(self, train_idx=None, val_idx=None):
        params = {'batch_size': self.batch_size, 'num_workers': 6, 'drop_last': True}
        if train_idx is not None:
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(self.dataset, sampler=train_sampler, **params)
            val_loader = DataLoader(self.dataset, sampler=val_sampler, **params)
        else:
            datasets = self.train_val_dataset(self.dataset, val_split=0.25)
            train_loader = DataLoader(datasets['train'], **params)
            val_loader = DataLoader(datasets['val'], **params)

        return train_loader, val_loader

    def load(self):
        model_path = Path(self.model_path)
        assert model_path.exists() and model_path.is_dir(), f'{model_path} does not exist or not a directory'
        self.model = self.get_model()
        self.model.load_state_dict(torch.load(model_path / 'model.pth'))
        self.model.to(self.device)
        self.print(f'model {self.model_name} load from: {model_path}')
        self.cache_dir = model_path.parent

    def save_model(self):
        dir_path = Path(f"{SAVED_MODEL_DIR}/{self.model_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        dir_path.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), dir_path / 'model.pth')
        self.print(f'model saved to {dir_path}')
        self.cache_dir = dir_path

    def get_dataset(self):
        raise NotImplemented('Must create a method get_dataset')

    def get_model(self):
        raise NotImplemented('Must create a method get_model')

    def get_optimizer(self):
        return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def get_scheduler(self, optimizer):
        return

    def get_loss_fn(self):
        raise NotImplemented('you must specify a get_loss_fn')

    def is_better_score_(self, score, best_so_far):
        if self.monitored_metric_algo == 'min':
            return score < best_so_far
        else:
            return score > best_so_far

    def get_best_model(self, history):
        algo = np.argmin if self.monitored_metric_algo == 'min' else np.argmax
        return int(algo([h['score'] for h in history]))

    def print_dataset_info(self, datasets: dict):
        for key, dataset in datasets.items():
            s = pd.Series([y[1] for y in dataset.samples]).value_counts()
            s.index = s.index.map({v: k for k, v in dataset.class_to_idx.items()})
            text = ', '.join(f'{k}: {v}' for k, v in s.iteritems())
            self.print(f'{key} dataset distribution: {text}')

    def summary_plots(self, history, chosen_idx):
        fig, axes = plt.subplots(2, 3, figsize=(25, 4))
        self.plot_train_metrics(history, chosen_idx, axes[0, :])
        self.all_data_evaluation(axes=axes[1, :])
        if self.cache_dir:
            fig.savefig(self.cache_dir / 'summary_plots.jpg')
        plt.show()

    def plot_train_metrics(self, history, chosen_idx, axes=None):
        chosen_metrics = history[chosen_idx]['metrics']
        # Loss of chosen model
        epochs = np.arange(1, self.num_epochs + 1)
        axes[0].plot(epochs, chosen_metrics['train_loss'], color='k', label='train_loss')
        axes[0].plot(epochs, chosen_metrics['val_loss'], color='purple', label='val_loss')
        axes[0].legend()
        axes[0].set_title('Loss')
        # Chosen Model Validation Metrics
        for k, v in chosen_metrics.items():
            if k in ['train_loss', 'val_loss']:
                continue
            axes[1].plot(epochs, v, label=k)
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        # comparison of monitored metric
        for i, h in enumerate(history):
            axes[2].plot(h['metrics'][self.monitored_metric], label=f'FOLD-{i+1}', color='r' if i == chosen_idx else 'k')
        axes[2].set_title(f'Comparison of {self.monitored_metric} between folds')
        axes[2].legend()

    def all_data_evaluation(self, axes=None):
        pass

    def print(self, msg):
        if self.is_debug:
            print(msg)


@dataclass
class RegressionTrainer(Trainer):
    monitored_metric: str = 'mse'
    monitored_metric_algo: str = 'min'

    def evaluate(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        return {
            'mse': torch.nn.functional.mse_loss(y_true, y_pred).item(),
            'explained_variance': explained_variance_score(y_true.cpu(), y_pred.cpu())
        }

    def get_loss_fn(self):
        return nn.MSELoss()


@dataclass
class ClassificationTrainer(Trainer):
    no_prediction_index: int = 0
    monitored_metric: str = 'accuracy'
    monitored_metric_algo: str = 'max'
    targets = []

    def get_loss_fn(self):
        return nn.CrossEntropyLoss()

    def predict_proba(self, y_pred: torch.Tensor):
        p = F.softmax(y_pred, dim=1)
        pmax, predicted = torch.max(p, dim=1)
        predicted[pmax < self.threshold] = self.no_prediction_index
        return predicted, p[:, 1]

    def evaluate(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        y_pred, y_score = self.predict_proba(y_pred)
        y_true, y_pred, y_score = y_true.cpu().numpy(), y_pred.cpu().numpy(), y_score.cpu().numpy()
        return {
            'auc': roc_auc_score(y_true, y_score),
            'recall': recall_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'accuracy': balanced_accuracy_score(y_true, y_pred)
        }

    def convert_inputs(self, x, y):
        return x.to(self.device), y.to(self.device)

    @staticmethod
    def equalize_dataset(dataset):
        s = pd.Series(dataset.targets)
        labels = s.unique()
        min_count = s.value_counts().iloc[-1]
        min_label = s.value_counts().index[-1]
        samples_ = []
        samples_a = np.array(dataset.samples)
        for lbl in labels:
            if lbl == min_label:
                chosen_idx = s[s == lbl].index
            else:
                chosen_idx = np.random.choice(s[s == lbl].index, min_count)
            samples_.extend(samples_a[chosen_idx].tolist())
        samples_ = [(path, int(k)) for path, k in samples_]
        dataset.samples = samples_
        return dataset

    def all_data_evaluation(self, axes=None):
        if axes is None:
            fig, axes_ = plt.subplots(1, 3, figsize=(18, 4))
        else:
            axes_ = axes
        assert len(axes_) == 3
        self.model.eval()
        dataset = self.equalize_dataset(self.get_dataset())
        y_true, y_pred, y_score = [], [], []
        for x, y in tqdm(dataset):
            outputs = self.model(x.to(self.device).unsqueeze(0))
            label, prob = self.predict_proba(outputs)
            y_true.append(y)
            y_pred.append(label.item())
            y_score.append(prob.item())
        y_true, y_pred, y_score = np.vstack(y_true), np.vstack(y_pred), np.vstack(y_score)
        self.plot_confusion_matrix(y_true, y_pred, ax=axes_[0])
        PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=axes_[1])
        RocCurveDisplay.from_predictions(y_true, y_score, ax=axes_[2])
        if axes is None:
            plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, ax=None):
        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, index=self.targets, columns=self.targets)
        if ax is None:
            fig, ax_ = plt.subplots(figsize=(10, 7))
        else:
            ax_ = ax
        sns.heatmap(df_cm, annot=True, ax=ax_, fmt='.0f')
        ax_.set_ylabel('Ground Truth')
        ax_.set_xlabel('Predicted')
        if ax is None:
            plt.show()
