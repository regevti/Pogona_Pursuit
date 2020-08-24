"""
This module includes generic training and hyper-parameter tuning functions
"""
import pandas as pd
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader

from Prediction.dataset import (
    trial_to_samples,
)  # maybe transfer trial to samples to this module


def create_train_val_test_splits(trials_list, ratios):
    """
    :param trials_list: list of indexed trials in df (tuples from multindex or something else)
    :param ratios: fractions to divide by the list, sums to 1, order: train, val, test
    :return: 3 lists with the trial names for train, val and test
    """
    l = list(trials_list)
    np.random.shuffle(l)

    test_i = round(len(l) * ratios[0])
    val_i = test_i + round(len(l) * ratios[1])

    return l[:test_i], l[test_i:val_i], l[val_i:]


def create_train_val_test_dataloaders(
    df,
    train,
    val,
    test,
    input_labels,
    output_labels,
    input_seq_size,
    output_seq_size,
    batch_size=256,
    shuffle=True,
    num_workers=0,
):
    """
    create 3 dataloaders - train, validation and test, based on the trials list
    :return: list with 3 dataloaders: [train, val, test]
    """
    ret = []

    for trials in [train, val, test]:
        tensor_list_X = []
        tensor_list_Y = []
        for trial in trials:
            X, Y = trial_to_samples(
                df.loc[trial],
                input_labels=input_labels,
                output_labels=output_labels,
                input_seq_size=input_seq_size,
                output_seq_size=output_seq_size,
            )
            tensor_list_X.append(X)
            tensor_list_Y.append(Y)

        all_X = torch.cat(tensor_list_X)
        all_Y = torch.cat(tensor_list_Y)

        dataset = TrajectoriesData(all_X, all_Y)
        ret.append(
            DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
            )
        )

    return ret


class TrajectoriesData(Dataset):
    def __init__(self, X, Y):
        assert X.shape[0] == Y.shape[0]

        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.Y[item]


def create_dataloader(
    X, Y, train_test_ratio=0.8, shuffle=True, batch_size=256, num_workers=0
):
    dataset = TrajectoriesData(X, Y)
    train_size = round(train_test_ratio * len(dataset)) - 1
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return train_loader, test_loader


def calc_ADE(y_hat, y):
    """
    Calculate the mean average displacement error (ADE) of a test set of predictions
    displacement is the L2 norm between the prediction and the truth
    :param y_hat: torch tensor dims (number of sample, out seq size, out size)
    :param y: same
    :return: mean ADE over batch
    """
    if len(y_hat.shape) == 2:
        y_hat = y_hat.unsqueeze(0)
        y = y.unsqueeze(0)

    return torch.mean(torch.norm(y_hat - y, dim=2), dim=1).mean()


def calc_FDE(y_hat, y):
    """
    Calculate the mean final displacement error (FDE) of a test set of predictions
    displacement is the L2 norm between the prediction and the truth
    :param y_hat: torch tensor dims (number of sample, out seq size, out size)
    :param y: same
    :return: mean FDE over batch
    """
    if len(y_hat.shape) == 2:
        y_hat = y_hat.unsqueeze(0)
        y = y.unsqueeze(0)

    return torch.norm(y_hat[:, -1, :] - y[:, -1, :], dim=1).mean()


def grid_input_output(
    model_name,
    df,
    input_seqs,
    output_seqs,
    input_labels,
    output_labels,
    path,
    num_epochs=5000,
):
    """
    Perform 2D grid search over cartesian product of input and output sequence lengths with labels
    assumes input_seqs and output_seqs are iterables
    :return: pandas df with ADE scores
    """

    scores = pd.DataFrame(index=input_seqs, columns=output_seqs)
    train, val, test = create_train_val_test_splits(df.index.unique(), [0.7, 0.2, 0.1])
    count = 0
    num_trains = len(input_seqs) * len(output_seqs)

    for inp_seq in input_seqs:
        for out_seq in output_seqs:
            count += 1
            print("================================================"
            print(f"{count}/{num_trains} Training with input_seq_len={inp_seq}, output_seq_len={out_seq}")

            train_dl, val_dl, test_dl = create_train_val_test_dataloaders(
                df, train, val, test, input_labels, output_labels, inp_seq, out_seq
            )

            net = model_name(len(input_labels), len(output_labels), out_seq)

            _, best_ADE = train_trajectory_model(
                net,
                train_dl,
                val_dl,
                num_epochs,
                path,
                eval_freq=100,
                model_name=f"model_{inp_seq}_{out_seq}",
            )

            scores.loc[inp_seq, out_seq] = best_ADE

    return scores


def train_trajectory_model(
    model,
    dataloader_train,
    dataloader_test,
    epochs,
    path,
    optimizer=None,
    eval_freq=100,
    epoch_print_freq=50,
    model_name="model",
):
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), amsgrad=True
        )  # TODO maybe use other optimizer params
        # criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    best_ADE = None
    best_epoch = None

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0.0
        epoch_start = time.time()
        for i, (x, y) in enumerate(dataloader_train):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            """
            print(f'x shape: {x.shape}, y shape {y.shape}')
            print(f'pred shape: {output.shape}')
            print(f'flat y shape: {flattened_y.shape}')
            """

            loss = calc_ADE(output, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_time = time.time() - epoch_start
        if (epoch - 1) % epoch_print_freq == 0:
            print(
                f"epoch: {epoch}, loss: {epoch_loss:.3f}, epoch time: {epoch_time:.3f}"
            )

        if (epoch - 1) % eval_freq == 0:
            model.eval()
            total_ADE = 0
            count = 0
            for (x, y) in dataloader_test:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    pred = model(x)

                total_ADE += calc_ADE(pred, y)
                count += 1

            ADE = total_ADE / count

            if best_ADE is None or ADE < best_ADE:
                best_ADE = ADE.item()
                best_epoch = epoch
                torch.save(model.state_dict(), path + f"/{model_name}_best.pth")

            print("++++++++++++++++")
            print(f"Eval epoch: {epoch}, Test set mean ADE {ADE:.3f}")
            torch.save(model.state_dict(), path + f"/{model_name}_{epoch}.pth")

    torch.save(model.state_dict(), path + "/model_final.pth")
    print("Finished training")
    return best_epoch, best_ADE
