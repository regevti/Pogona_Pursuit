"""
This module is responsible for generating sequence data, sampling and masking it using various statistics,
training models and evaluating them.

The first part deals with generating the sequence data from the trials and creating PyTorch Datasets and DataLoaders.
The second part deals with using the DataLoaders to train seq2seq RNN models.
The third part includes Numpy data masking functions, compute some statistic on the sequences array and mask the data
according to some boolean condition.
"""
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm  # python 3.7 issues

# from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
-------------------- Datasets and Dataloaders -----------------------------
"""


class TrajectoriesData(Dataset):
    """
    A PyTorch Dataset for holding (x,y) sequence pairs, i.e, x is the sequence of M detections,
    and y is the sequence of N detections that follows.
    """
    def __init__(self, X, Y):
        """
        :param X: 3D tensor, array of 2D sequences, each sequence of dimension (sequence length, xyxy)
        :param Y: Same
        """
        assert X.shape[0] == Y.shape[0]

        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.Y[item]


def trial_to_samples(
    trial_df,
    input_labels,
    output_labels,
    input_seq_size,
    output_seq_size,
    keep_nans=False,
    mask_fn=None,
):
    """
    Extract samples from a single trial dataframe to torch 3D tensors, X with dimension
    (sample, input sequence index, single timestep dimension) and Y with dim
    (sample, output sequence index, single timestep dimension) using a sliding window
    Assumes that the data is corrected and transformed
    Iteratively creates 3D tensors from trials and finally concatenates them into a single tensor.
    Note: will require large amounts of memeory, alongside the function collect_data from dataset.py, these functions
    need to be re-written to work with larger than memory data, or by loading only small masked subsets of the data.

    :param trial_df: dataframe with the trial's detections
    :param input_labels: list of str, features to be used in the input sequence (xyxy for example)
    :param output_labels: same
    :param input_seq_size: int, length of the X sequence
    :param output_seq_size: int, length of the Y sequence
    :param keep_nans: bool, whether to keep sequences with NaN
    :param mask_fn: boolean bask function, to filter data before creating the tensor
    :return: tuple (X, Y), 3D tensors of sequences
    """

    input_2d_arrays = []
    output_2d_arrays = []

    inds_range = trial_df.shape[0] - input_seq_size - output_seq_size + 1

    input_data = trial_df[input_labels].values
    output_data = trial_df[output_labels].values

    for i in range(inds_range):
        inp_seq = input_data[i : i + input_seq_size]
        out_seq = output_data[i + input_seq_size : i + input_seq_size + output_seq_size]

        if not keep_nans and (np.any(np.isnan(inp_seq)) or np.any(np.isnan(out_seq))):
            continue

        input_2d_arrays.append(inp_seq)
        output_2d_arrays.append(out_seq)

    if len(input_2d_arrays) == 0:
        return None, None
    X = np.stack(input_2d_arrays)
    Y = np.stack(output_2d_arrays)

    if mask_fn is not None:
        mask = mask_fn(X, Y)
        X = X[mask]
        Y = Y[mask]

    # X = torch.from_numpy(X).float()
    # Y = torch.from_numpy(Y).float()

    return X, Y


def create_train_val_test_splits(trials_list, ratios):
    """
    :param trials_list: list of indexed trials in df (tuples from multindex or something else)
    :param ratios: iterable of floats, fractions to divide by the list, sums to 1, order: train, val, test
    :return: 3 lists with the trial names for train, val and test
    """
    trials = list(trials_list)
    np.random.shuffle(trials)

    test_i = round(len(trials) * ratios[0])
    val_i = test_i + round(len(trials) * ratios[1])

    return trials[:test_i], trials[test_i:val_i], trials[val_i:]


def create_samples(
    df,
    mask_fn,
    keep_prob,
    input_labels,
    output_labels,
    input_seq_size,
    output_seq_size,
    keep_nans=False,
):
    """
    Create dictionary of sequence tensors and masks. Allows to generate the sequence data once, and use different masks.
    :param df: pandas detections dataframe to create the sequence data from
    :param mask_fn: boolean masking function
    :param keep_prob: float, percentage of random data to keep in the samples
    :param input_labels: list of str, features to be used in the input sequence (xyxy for example)
    :param output_labels: same
    :param input_seq_size: int, length of the X sequence
    :param output_seq_size: int, length of the Y sequence
    :param keep_nans: bool, whether to keep sequences with NaN
    :return: dictionary of dictionaries, each dictionary contains the X, Y tensors for a trial and it's mask
    """
    trials_dict = dict()
    mask_fn_keep = keep_mask(mask_fn, keep_prob=keep_prob)
    for trial in tqdm(df.index.unique()):
        trial_dict = dict()

        X, Y = trial_to_samples(
            df.loc[trial],
            input_labels=input_labels,
            output_labels=output_labels,
            input_seq_size=input_seq_size,
            output_seq_size=output_seq_size,
            mask_fn=None,
            keep_nans=keep_nans,
        )

        if X is None:
            continue
        trial_mask = mask_fn(X, Y)

        # count_no_nan = (~df.loc[trial, 'x2'].isna()).sum()
        # mask_ratio = trial_mask.sum() / count_no_nan

        mask_ratio = trial_mask.sum() / X.shape[0]

        trial_dict["X"] = X
        trial_dict["Y"] = Y
        trial_dict["mask_ratio"] = mask_ratio
        trial_dict["mask"] = mask_fn_keep(X, Y)

        trials_dict[trial] = trial_dict
    return trials_dict


# functions for updating the trials dict without recreating the sequences again
def update_trial_dict(trial_dict, mask_fn, keep_prob):
    """
    Update the boolean mask for a trials dictionary
    :param trial_dict: an existing trial dictionary from a trials_dict generated by create_samples
    :param mask_fn: a composed boolean mask function
    :param keep_prob: float, percentage of random data to keep in the samples
    :return: no return value, updates in-place
    """
    mask_fn_keep = keep_mask(mask_fn, keep_prob=keep_prob)
    trial_dict["mask"] = mask_fn_keep(trial_dict["X"], trial_dict["Y"])
    trial_dict["mask_ratio"] = trial_dict["mask"].sum() / trial_dict["X"].shape[0]


def update_trials_dict(trials_dict, mask_fn, keep_prob):
    """
    Iterative wrapper for update_trial_dict
    :param trials_dict: trials_dict generated by create_samples
    :param mask_fn: a composed boolean mask function
    :param keep_prob: float, percentage of random data to keep in the samples
    :return: no return value, updates in-place
    """
    for trial_dict in tqdm(trials_dict.values()):
        update_trial_dict(trial_dict, mask_fn, keep_prob)


def split_train_val_test(trials_dict, split=(0.8, 0.2, 0)):
    """
    Splits the trials for test, validation and test sets according to ratios
    :param trials_dict: trials_dict generated by create_samples
    :param split: 3-tuple of floats that sum to 1.0, default: (0.8, 0.2, 0)
    :return: 3 lists of the trials in each set
    """

    train_trials, val_trials, test_trials = [], [], []

    sorted_trials = [
        item[0]
        for item in sorted(
            trials_dict.items(), key=lambda x: x[1]["mask_ratio"], reverse=True
        )
    ]

    splits = np.array(split)
    chunk_size = np.round(1 / min(splits[splits > 0])).astype(int)

    for i in range(0, len(sorted_trials), chunk_size):
        train, val, test = create_train_val_test_splits(
            sorted_trials[i : min(i + chunk_size, len(sorted_trials))], split
        )
        train_trials += train
        val_trials += val
        test_trials += test

    return train_trials, val_trials, test_trials


def create_train_val_test_dataloaders(
    trials_dict,
    train_trials,
    val_trials,
    test_trials,
    train_mask=True,
    val_mask=False,
    test_mask=False,
    batch_size=256,
    shuffle=True,
    num_workers=0,
):
    """
    Create 3 PyTorch DataLoaders: train, validation and test
    :param trials_dict: trials_dict generated by create_samples
    :param train_trials: list of trials (str) for the train set
    :param val_trials: same - validation
    :param test_trials: same - test
    :param train_mask: boolean masking function for the train set
    :param val_mask: same - validation
    :param test_mask: same - test
    :param batch_size: int, number of sequences in batch, a parameter for the DataLoader
    :param shuffle: bool, whether to shuffle the order of the sequences in each batch
    :return: 3 DataLoaders
    """

    ret = []
    for trial_list, to_mask in zip(
        [train_trials, val_trials, test_trials], [train_mask, val_mask, test_mask]
    ):

        tensor_list_X = []
        tensor_list_Y = []

        if len(trial_list) == 0:
            ret.append(None)
            continue

        for trial in trial_list:
            X = trials_dict[trial]["X"]
            Y = trials_dict[trial]["Y"]

            if to_mask:
                X = X[trials_dict[trial]["mask"]]
                Y = Y[trials_dict[trial]["mask"]]
            tensor_list_X.append(torch.from_numpy(X).float())
            tensor_list_Y.append(torch.from_numpy(Y).float())

        all_X = torch.cat(tensor_list_X)
        all_Y = torch.cat(tensor_list_Y)

        dataset = TrajectoriesData(all_X, all_Y)

        if len(dataset) == 0:
            print()
            raise Exception(f"Index in ret: {len(ret)}, Dataset is empty")
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


class TrainingResults:
    """
    Class for holding the training results during training, passed an initialized object to the training function
    """
    def __init__(self, model_name=None):
        self.best_ADE = None
        self.best_epoch = None
        self.ADEs = {}
        self.FDEs = {}
        self.losses = []
        self.model_name = model_name


def calc_ADE(y_hat, y):
    """
    Calculate the mean average displacement error (ADE) of a test set of predictions
    displacement is the L2 norm between the prediction and the truth

    :param y_hat: torch tensor dims (number of sample, out seq size, out size)
    :param y: same
    :return: float, mean ADE over batch
    """
    if len(y_hat.shape) == 2:
        y_hat = y_hat.unsqueeze(0)
        y = y.unsqueeze(0)

    return torch.mean(torch.norm(y_hat - y, dim=2), dim=1).mean()


def weighted_ADE(weights, y_hat, y):
    """
    same as the calc_ADE, but each error term in the sequence is weighted differently
    :param weights: array of floats, weights for each timestep in the sequence
    """
    if len(y_hat.shape) == 2:
        y_hat = y_hat.unsqueeze(0)
        y = y.unsqueeze(0)

    return torch.sum(
        weights.repeat((y.size(0), 1)) * torch.norm(y_hat - y, dim=2), dim=1
    ).mean()


def randomized_ADE(y_hat, y):
    """
    computes the error only up to some random timestep of the sequence
    """
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


# TODO delete?
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

    return (
        torch.abs(total_norm_displacement).mean(),
        torch.abs(total_angle_displacement).mean(),
    )


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
    sched_exp=None,
    results=None,
):
    """
    Training a model

    :param model: initialized PyTorch seq2seq model
    :param train_dataloader: train set DataLoader
    :param test_dataloader: validation set DataLoader
    :param epochs: int, number of epochs (iterations over dataset)
    :param path: folder to save checkpoints in
    :param lr: float, learning rate
    :param optimizer: PyTorch optimizer, Adam with amsgrad by default
    :param loss_fn: loss function (ADE for example)
    :param clip_grad_norm: float, clip gradient if larger than this value
    :param eval_freq: int, frequency of evaluation on the validation set
    :param epoch_print_freq: int, frequency of printing training stats
    :param model_name: str, model name
    :param save_each_eval_model: bool, whether to save model after each evaluation, default False
    :param sched_exp: exponent base for the scheduling sampling training regime, where predictions are replaced with
    ground truth with some probability at each iteration
    :param results: results object
    :return: results object
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), amsgrad=True, lr=lr
        )  # TODO maybe use other optimizer params

    if results is None:
        results = TrainingResults()

    model.to(device)

    for epoch in range(epochs):
        if epoch % eval_freq == 0:
            ADE, FDE = eval_trajectory_model(model, test_dataloader)
            results.ADEs[epoch] = ADE.item()
            results.FDEs[epoch] = FDE.item()

            if results.best_ADE is None or ADE < results.best_ADE:
                results.best_ADE = ADE.item()
                results.best_epoch = epoch
                torch.save(model.state_dict(), path + f"/{model_name}_best.pth")

            print(
                f"### Eval epoch: {epoch}, Test set mean ADE: {ADE:.3f}, mean FDE: {FDE:.3f}",
                end="",
            )
            if results.best_epoch == epoch:
                print(" BEST")
            else:
                print("")

            if save_each_eval_model:
                torch.save(model.state_dict(), path + f"/{model_name}_{epoch}.pth")

        model.train()

        epoch_loss = 0.0
        epoch_start = time.time()

        if sched_exp:
            sched_epsi = sched_exp ** epoch
        else:
            sched_epsi = 0

        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            model.sampling_eps = sched_epsi
            model.target = y
            output = model(x)
            model.sampling_eps = 0

            loss = loss_fn(output, y)

            loss.backward()

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_dataloader)

        results.losses.append(avg_loss)

        if epoch % epoch_print_freq == 0:
            print(
                f"Epoch: {epoch}, avg loss: {avg_loss:.3f}, epoch time: {epoch_time:.3f}",
                end="",
            )

            print(f" epsi: {sched_epsi}") if sched_exp else print("")

    torch.save(model.state_dict(), path + f"/{model_name}_final.pth")
    print("Finished training")
    return results


def eval_trajectory_model(model, test_dataloader):
    """
    Evaluate model on the validation/test set
    :param model: initialized seq2seq model
    :param test_dataloader: sequences DataLoader
    :return: mean ADE, mean FDE
    """
    model.eval()
    total_ADE = 0
    total_FDE = 0
    count = 0
    for (x, y) in test_dataloader:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            output = model(x)

            total_ADE += calc_ADE(output, y)
            total_FDE += calc_FDE(output, y)
            count += 1

    ADE = total_ADE / count
    FDE = total_FDE / count
    return ADE, FDE


def eval_trajectory_predictor(trajectory_predictor, bboxes, show_progress=True):
    """
    Run trajectory_predictor on the bboxes array.
    :param trajectory_predictor: An instance of TrajectoryPredictor
    :param bboxes: A consecutive list of bbox detections (numpy array Nx4+)
    :param show_progress: bool, whether to display tqdm progress bar, default True
    :return: results - a dictionary of evaluation results, forecasts - a list of forecasts for each corresponding bbox
    """

    saw_non_nan = False
    forecasts = []
    times = []

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
            times.append(time.time() - start_time)

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

    results = {"avg ADE": sum_ADE / len(forecasts), "avg FDE": sum_FDE / len(forecasts),
               "avg time (ms)": sum(times) / len(forecasts) * 1000, "times": times}
    return results, forecasts


def get_seq_forecast(dataloader, model):
    """
    Generate Y_hat - the predictions tensor using a seq2seq model
    :param dataloader: test dataloader
    :param model: initialized seq2seq model
    :return: 3D PyTorch predictions tensor
    """
    forecasts_list = []

    with torch.no_grad():
        for x, y in dataloader:
            y_hat = model(x.float().to(model.device)).cpu().numpy()
            forecasts_list.append(y_hat)
    return np.concatenate(forecasts_list)


def eval_traj_predictor_on_trials(trials, predictor, all_df):
    """
    General function using any TrajectoryPredictor (seq2seq or Kalman)
    :param trials: list of trials to evaluate on
    :param predictor: Initialized TrajectoryPredictor object
    :param all_df: unified detections DataFrame
    :return: array of ADEs, array of FDEs, list of lengths of each trial, list of 3D prediction tensors
    """
    all_results = []
    all_forecasts = []
    trial_lens = []

    for trial in tqdm(trials):
        bboxes = all_df.loc[trial][['x1', 'y1', 'x2', 'y2']].values
        results, forecasts = eval_trajectory_predictor(predictor, bboxes, show_progress=False)
        all_results.append(results)
        all_forecasts.append(forecasts)
        trial_lens.append(bboxes.shape[0])

    ADEs = [r['avg ADE'] for r in all_results]
    FDEs = [r['avg FDE'] for r in all_results]

    return ADEs, FDEs, trial_lens, all_forecasts


def all_forecasts_to_seq_data(all_forecasts, inp_seq_len, out_seq_len, out_dim):
    """
    Transform list of 3D tensors to a single 3D arrays, to be used with Kalman predictions
    :param all_forecasts: list of all 3D prediction arrays
    :param inp_seq_len: length of input sequence
    :param out_seq_len: length of output sequence
    :param out_dim: dimension of output
    :return: 3D numpy array
    """
    all_np_forecasts = []
    nans_forecast = np.empty((out_seq_len, out_dim))
    nans_forecast[:] = np.nan

    for forecasts in all_forecasts:
        forecasts_wo_nones = [nans_forecast if f is None else f for f in forecasts]
        np_forecasts = np.stack(forecasts_wo_nones[inp_seq_len - 1:-out_seq_len])
        all_np_forecasts.append(np_forecasts)

    return np.concatenate(all_np_forecasts)

"""
-------------------- Masking functions -----------------------------
The following closure functions return a masking function that computes stats for an array of sequences and then
returns a boolean mask based on some boolean conditions. The functions operate in the same way, and then composed into
a unified mask
"""


def compute_dists(seq_data):
    """
    :param seq_data: 3D sequence array
    :return: l2 distance between first and last coordinates in the sequnce
    """
    return np.linalg.norm(seq_data[:, -1] - seq_data[:, 0], axis=1)


def mask_fl_dist(min_dist, max_dist, by_X=True, by_Y=True):
    def f(X, Y):
        dists_X = compute_dists(X[:, :, :2])
        dists_Y = compute_dists(Y[:, :, :2])

        ret_mask = np.array([True] * X.shape[0])

        if by_X:
            ret_mask = ret_mask & ((dists_X > min_dist) & (dists_X < max_dist))
        if by_Y:
            ret_mask = ret_mask & ((dists_Y > min_dist) & (dists_Y < max_dist))
        return ret_mask

    return f


def compute_speeds(seq_data):
    """
    :param seq_data: 3D sequence array
    :return: mean l2 speed of the sequence
    """
    return np.linalg.norm(np.diff(seq_data, axis=1), axis=2).mean(axis=1)


def mask_speed(min_speed, max_speed, by_X=True, by_Y=True):
    def f(X, Y):
        speeds_X = compute_speeds(X[:, :, :2])
        speeds_Y = compute_speeds(Y[:, :, :2])

        ret_mask = np.array([True] * X.shape[0])

        if by_X:
            ret_mask = ret_mask & ((speeds_X > min_speed) & (speeds_X < max_speed))
        if by_Y:
            ret_mask = ret_mask & ((speeds_Y > min_speed) & (speeds_Y < max_speed))
        return ret_mask

    return f


def compute_zigzagity(seq_data):
    """
    :param seq_data: 3D sequence array
    :return: mean "Zigzagity", which is the angle difference between consecutive velocity vectors in the sequence
    """
    vecs = seq_data[:, 1:] - seq_data[:, :-1]
    u = vecs[:, 1:]
    v = vecs[:, :-1]
    u_norm = np.linalg.norm(u, axis=2)
    v_norm = np.linalg.norm(v, axis=2)
    dotprods = np.einsum(
        "ij,ij->i", u.reshape(-1, u.shape[2]), v.reshape(-1, v.shape[2])
    ).reshape(u.shape[0], u.shape[1])
    angles = np.arccos(dotprods / (u_norm * v_norm))
    return angles.mean(axis=1)


def mask_zgzg(min_zgzg, max_zgzg, by_X=True, by_Y=True):
    def f(X, Y):
        zgzgs_X = compute_zigzagity(X[:, :, :2])
        zgzgs_Y = compute_zigzagity(Y[:, :, :2])

        ret_mask = np.array([True] * X.shape[0])

        if by_X:
            ret_mask = ret_mask & ((zgzgs_X > min_zgzg) & (zgzgs_X < max_zgzg))
        if by_Y:
            ret_mask = ret_mask & ((zgzgs_Y > min_zgzg) & (zgzgs_Y < max_zgzg))
        return ret_mask

    return f


def compute_var(X, Y):
    """

    :param X: 3D X sequence array
    :param Y: 3D Y sequence array
    :return: variance of the sequence
    """
    XY = np.concatenate((X, Y), axis=1)
    XYmeans = XY.mean(axis=1).reshape(XY.shape[0], 1, XY.shape[2])
    XYmeans = np.repeat(XYmeans, XY.shape[1], axis=1)
    XYnorms = np.linalg.norm(XY - XYmeans, axis=2)
    XYvar = (XYnorms ** 2).mean(axis=1)
    return XYvar


def mask_std(min_std, max_std, cols=[0, 1]):
    def f(X, Y):
        variances = compute_var(X[:, :, cols], Y[:, :, cols])
        ret_mask = (variances > min_std ** 2) & (variances < max_std ** 2)
        return ret_mask

    return f


def tile_means(x, seq_len):
    return np.tile(x.reshape(x.shape[0], 1), seq_len)


def comp_sqr_residuals(x):
    return np.sqrt((x ** 2).sum(axis=1))


def compute_batch_r(seq_data):
    """
    Computes Pearson's r correlation between x and y. Assumes X is a 3d tensor, which is stacked pairs of vectors of 2d x-y data.
    Significantly faster than np.corrcoeff or scipy.stats.pearsonr applied to each sample in a for loop, the same numerical result
    """
    if seq_data.shape[2] == 4:
        seq_data = seq_data[:, :, 2:]
    seqs_means_x, seqs_means_y = np.mean(seq_data, axis=1).transpose()
    seq_len = seq_data.shape[1]
    tiled_means_x, tiled_means_y = (
        tile_means(seqs_means_x, seq_len),
        tile_means(seqs_means_y, seq_len),
    )
    x_minus_xbar = seq_data[:, :, 0] - tiled_means_x
    y_minus_ybar = seq_data[:, :, 1] - tiled_means_y
    numers = (x_minus_xbar * y_minus_ybar).sum(1)
    denoms = comp_sqr_residuals(x_minus_xbar) * comp_sqr_residuals(y_minus_ybar)
    return np.abs(numers / denoms)
    #return numers / denoms


def mask_corr(min_corr, max_corr, by_X=True, by_Y=True):
    def f(X, Y):
        corr_X = compute_batch_r(X[:, :, :2])
        corr_Y = compute_batch_r(Y[:, :, :2])

        ret_mask = np.array([True] * X.shape[0])

        if by_X:
            ret_mask = ret_mask & ((corr_X > min_corr) & (corr_X < max_corr))
        if by_Y:
            ret_mask = ret_mask & ((corr_Y > min_corr) & (corr_Y < max_corr))
        return ret_mask

    return f


def compose_masks(mask_fns, invert=False):
    """
    :param mask_fns: list of functions computed by closure functions
    :param invert: bool, whether to take the negative of each mask (not logically sound, need improvement)
    :return: function that returns a boolean mask when operates on 3D tensors
    """
    def f(X, Y):
        mask = np.array([True] * X.shape[0])
        for fn in mask_fns:
            temp_mask = fn(X, Y)
            mask = mask & temp_mask
        if invert:
            return ~mask
        return mask

    return f


def keep_mask(mask_fn, keep_prob=0.2):
    """
    :param mask_fn: a composed boolean masking function
    :param keep_prob: float, percentage of random data to keep
    :return: booelean masking function
    """
    def f(X, Y):
        rand_values = np.random.random(X.shape[0])
        mask = (rand_values < keep_prob) | mask_fn(X, Y)
        return mask

    return f
