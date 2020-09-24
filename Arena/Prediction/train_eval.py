"""
This module includes generic training and hyper-parameter tuning functions
"""
import pandas as pd
import numpy as np
import torch
import time
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
#from tqdm.auto import tqdm # python 3.7 issues
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
-------------------- Datasets and Dataloaders -----------------------------
"""


class TrajectoriesData(Dataset):
    def __init__(self, X, Y):
        assert X.shape[0] == Y.shape[0]

        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.Y[item]


class TrajectoriesDataWithHeads(Dataset):
    def __init__(self, X, X_images, Y):
        assert X.shape[0] == Y.shape[0]

        self.X_images = X_images
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return (self.X[item], self.X_images[item]), self.Y[item]


def im2tensor(im, resize=32):
    if im is None:
        ret_tensor = torch.empty(resize, resize)
        ret_tensor[:] = float('nan')
    else:
        ret_tensor = torch.tensor(cv.resize(im, (resize, resize)))
    return ret_tensor


def compose_masks(mask_fns):
    def f(X, Y):
        mask = np.array([True]*X.shape[0])
        for fn in mask_fns:
            temp_mask = fn(X, Y)
            mask = mask & temp_mask
        return mask
    return f


def trial_to_samples(
    trial_df,
    input_labels,
    output_labels,
    input_seq_size,
    output_seq_size,
    input_images=None,
    keep_nans=False
):
    """
    Extract samples from a single trial dataframe to torch 3D tensors, X with dimension
    (sample, input sequence index, single timestep dimension) and Y with dim
    (sample, output sequence index, single timestep dimension)
    Assumes that the data is corrected and transformed

    returns: 2 3D tensors X, Y
    """

    input_2d_arrays = []
    output_2d_arrays = []
    input_3d_images = []

    inds_range = trial_df.shape[0] - input_seq_size - output_seq_size + 1

    input_data = trial_df[input_labels].values
    output_data = trial_df[output_labels].values

    if input_images is not None:
        image_tensors = torch.stack([im2tensor(im, resize=32) for im in input_images])

    for i in range(inds_range):
        inp_seq = input_data[i : i + input_seq_size]
        if input_images is not None:
            img_seq = image_tensors[i : i + input_seq_size]

        out_seq = output_data[i + input_seq_size : i + input_seq_size + output_seq_size]

        if not keep_nans and (np.any(np.isnan(inp_seq)) or np.any(np.isnan(out_seq))):
            continue

        input_2d_arrays.append(inp_seq)
        output_2d_arrays.append(out_seq)

        if input_images is not None:
            input_3d_images.append(img_seq)
    if len(input_2d_tensors) == 0:
        return None, None
    X = np.stack(input_2d_arrays)
    Y = np.stack(output_2d_arrays)

    if input_images is not None:
        X_images = torch.stack(input_3d_images)

    if mask_fn is not None:
        mask = mask_fn(X, Y)
        X = X[mask]
        Y = Y[mask]

        if input_images is not None:
            X_images = X_images[mask]

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    if input_images is not None:
        return (X, X_images), Y

    return X, Y


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
    std_threshold=None,
    keep_lowvar_prob=0.2,
    num_workers=0,
):
    """
    create 3 dataloaders - train, validation and test, based on the trials list
    :return: list with 3 dataloaders: [train, val, test]
    """
    ret = []

    for trials in [train, val, test]:
        if len(trials) == 0:
            ret.append(None)
            continue

        tensor_list_X = []
        tensor_list_Y = []
        for trial in trials:
            X, Y = trial_to_samples(
                df.loc[trial],
                input_labels=input_labels,
                output_labels=output_labels,
                input_seq_size=input_seq_size,
                output_seq_size=output_seq_size,
                std_threshold=std_threshold,
                keep_lowvar_prob=keep_lowvar_prob,
            )
            if X is None:
                continue
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


# TODO is this function necessary?
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


"""
-------------------- Training and Evaluation -----------------------------
"""


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


def weighted_ADE(y_hat, y, weights):
    if len(y_hat.shape) == 2:
        y_hat = y_hat.unsqueeze(0)
        y = y.unsqueeze(0)

    return torch.sum(weights.repeat((y.size(0), 1)) * torch.norm(y_hat - y, dim=2), dim=1).mean()


def randomized_ADE(y_hat, y):
    if len(y_hat.shape) == 2:
        y_hat = y_hat.unsqueeze(0)
        y = y.unsqueeze(0)

    random_k = np.random.randint(1, y_hat.shape[1])
    return calc_ADE(y_hat[:, :random_k], y[:, :random_k])


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


def angle_norm_displacement(target, pred):
    """
    Calculate the mean cumulative displacement of angles and norms for all sequences
    in the batch.

    The total angle displacement is calculated on non-absolute angle diffs so that angle
    corrections can cancel each other out.

    :return: (mean of absolute cumulative norm displacement, mean of absolute cumulative angle displacement)
    """
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    target_diffs = target[:, 1:] - target[:, :-1]
    pred_diffs = pred[:, 1:] - pred[:, :-1]

    target_angle = torch.atan(target_diffs[:, :, 1] / target_diffs[:, :, 0])
    pred_angle = torch.atan(pred_diffs[:, :, 1] / pred_diffs[:, :, 0])
    target_angle_diffs = target_angle[:, 1:] - target_angle[:, :-1]
    pred_angle_diffs = pred_angle[:, 1:] - pred_angle[:, :-1]
    total_angle_displacement = (target_angle_diffs - pred_angle_diffs).sum(1)

    target_norms = torch.norm(target_diffs, dim=2)
    pred_norms = torch.norm(pred_diffs, dim=2)
    total_norm_displacement = (target_norms - pred_norms).sum(1)

    return (torch.abs(total_norm_displacement).mean(),
            torch.abs(total_angle_displacement).mean())


def angle_norm_displacement_loss_fn(weights):
    def fn(output, target):
        n, a = angle_norm_displacement(output, target)
        return weights[0] * n + weights[1] * a

    return fn


def train_trajectory_model(
    model,
    train_dataloader,
    test_dataloader,
    epochs,
    path,
    lr=0.005,
    optimizer=None,
    loss_fn=calc_ADE,
    clip_grad_norm=None,
    eval_freq=100,
    epoch_print_freq=50,
    model_name="model",
    save_each_eval_model=True,
):
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), amsgrad=True, lr=lr
        )  # TODO maybe use other optimizer params

    model.to(device)

    best_ADE = None
    best_epoch = None
    ADEs = {}
    FDEs = {}
    losses = np.empty(epochs)

    for epoch in range(epochs):
        if epoch % eval_freq == 0:
            ADE, FDE = eval_trajectory_model(model, test_dataloader)
            ADEs[epoch] = ADE.item()
            FDEs[epoch] = FDE.item()

            if best_ADE is None or ADE < best_ADE:
                best_ADE = ADE.item()
                best_epoch = epoch
                torch.save(model.state_dict(), path + f"/{model_name}_{epoch}_best.pth")

            print(
                f"### Eval epoch: {epoch}, Test set mean ADE: {ADE:.3f}, mean FDE: {FDE:.3f}"
            )
            if save_each_eval_model:
                torch.save(model.state_dict(), path + f"/{model_name}_{epoch}.pth")

        model.train()

        # a = np.linspace(0.01, 0.2, num=20)[::-1]
        # b = np.array([a[t] * np.prod(1-a[:t]) for t in range(len(a))])[::-1]
        # b = torch.from_numpy(b.copy()).to(device)

        epoch_loss = 0.0
        epoch_start = time.time()
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            if type(x) is list:  # x = (bbox, head image)
                x_bbox, x_head = x
                x_bbox = x_bbox.to(device)
                x_head = x_head.to(device)
                y = y.to(device)
                output = model(x_bbox, x_head)
            else:
                x, y = x.to(device), y.to(device)
                output = model(x)

            # rel_out = output - offset
            # rel_y = y - offset

            # out_pairs = torch.stack((rel_out[:, 1:], rel_out[:, :-1]), dim=3)
            # y_pairs = torch.stack((rel_y[:, 1:], rel_y[:, :-1]), dim=3)
            # gram_out = torch.matmul(out_pairs, out_pairs.transpose(2, 3))
            # gram_y = torch.matmul(y_pairs, y_pairs.transpose(2, 3))
            # loss = (out_pairs - y_pairs).norm(dim=(2, 3)).sum()
            # loss = criterion(rel_out, rel_y)
            # loss = criterion(rel_out, rel_y)
            # loss = weighted_ADE(output, y, b)
            #loss = randomized_ADE(output, y)

            loss = loss_fn(output, y)  # + calc_FDE(output, y)

            loss.backward()

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_dataloader)

        losses[epoch] = avg_loss

        if epoch % epoch_print_freq == 0:
            print(
                f"Epoch: {epoch}, avg loss: {avg_loss:.3f}, epoch time: {epoch_time:.3f}"
            )

    torch.save(model.state_dict(), path + f"/{model_name}_final.pth")
    print("Finished training")
    return best_epoch, best_ADE, losses, ADEs, FDEs


def eval_trajectory_model(model, test_dataloader):
    model.eval()
    total_ADE = 0
    total_FDE = 0
    count = 0
    for (x, y) in test_dataloader:
        with torch.no_grad():
            if type(x) is list:  # x = (bbox, head image)
                x_bbox, x_head = x
                x_bbox = x_bbox.to(device)
                x_head = x_head.to(device)
                y = y.to(device)
                output = model(x_bbox, x_head)
            else:
                x, y = x.to(device), y.to(device)
                output = model(x)

            total_ADE += calc_ADE(output, y)
            total_FDE += calc_FDE(output, y)
            count += 1

    ADE = total_ADE / count
    FDE = total_FDE / count
    return ADE, FDE


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
            print("================================================")
            print(
                f"{count}/{num_trains} Training with input_seq_len={inp_seq}, output_seq_len={out_seq}"
            )

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


def eval_trajectory_predictor(trajectory_predictor, bboxes, show_progress=True):
    saw_non_nan = False
    forecasts = []

    sum_time = 0

    # collect forecasts
    if show_progress:
        it = tqdm(range(bboxes.shape[0]))
    else:
        it = range(bboxes.shape[0])

    for i in it:
        bbox = bboxes[i]

        if not saw_non_nan:
            if not np.any(np.isnan(bbox)):
                saw_non_nan = True
                trajectory_predictor.init_trajectory(bbox)
            forecasts.append(None)
        else:
            history = bboxes[: i + 1]

            start_time = time.time()
            forecast = trajectory_predictor.update_and_predict(history)
            sum_time += time.time() - start_time

            forecasts.append(forecast)

    # analyze results
    sum_ADE = 0
    sum_FDE = 0

    if show_progress:
        it = tqdm(range(len(forecasts) - trajectory_predictor.forecast_horizon))
    else:
        it = range(len(forecasts) - trajectory_predictor.forecast_horizon)

    for i in it:
        forecast = forecasts[i]
        if forecast is None:
            continue

        target = bboxes[i + 1 : i + len(forecast) + 1]
        ADE = calc_ADE(torch.from_numpy(forecast), torch.from_numpy(target)).item()
        FDE = calc_FDE(torch.from_numpy(forecast), torch.from_numpy(target)).item()
        if not np.isnan(ADE):
            sum_ADE += ADE
        if not np.isnan(FDE):
            sum_FDE += FDE

    results = {}
    results["avg ADE"] = sum_ADE / len(forecasts)
    results["avg FDE"] = sum_FDE / len(forecasts)
    results["avg time (ms)"] = sum_time / len(forecasts) * 1000
    return results, forecasts
