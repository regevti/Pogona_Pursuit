"""
This module helps to manage a library of TrajEncDec models, and contains functions for:

- Training a model and storing its parameters and training results in a unified json file.
- Generate a module object using existing parameters.
- Retrieve training and model parameters of exisiting models.
- Plot the results of multiple training sessions.
"""

import os
import json
from datetime import datetime
import re
import matplotlib.pyplot as plt
import traceback

from Prediction import seq2seq_predict
from Prediction import train_eval


# Path to a directory containing model weight files.
WEIGHTS_DIR = "Prediction/traj_models"

# Path to the model parameters json file.
MODELS_JSON_PATH = os.path.join(WEIGHTS_DIR, "models_params.json")

# Dictionary of loss functions.
loss_fns = {
    "ADE": train_eval.calc_ADE,
    # maybe more?
}


def get_weights_path(model_name, suffix="best"):
    """
    Return the path of a model weights file.

    :param model_name: The model's name as used in training
    :param suffix: An additional suffix for the weights file.
                   'best'  - The weights with the best evaluation results.
                   'final' - The last weights generated in training.
                   epoch #  - A specific epoch number (the returned path might not exist for some values).
    """
    return os.path.join(WEIGHTS_DIR, f"{model_name}_{suffix}.pth")


def get_models_dict():
    """
    Return the entire models dictionary, as read from the model parameters json file.
    The dictionary keys are model names.
    """
    with open(MODELS_JSON_PATH, "r") as fp:
        return json.load(fp)


def get_trials_split(model_name):
    """
    Return the train/val/test split of experiment trials that was used to train a model.

    :param model_name: Name of the model that was trained on the returned split.
    :return: (train_trials, val_trials, test_trials), where each element is a list of trial names.
    """
    json_dict = get_models_dict()
    train_trials = json_dict[model_name]["training_params"]["train_trials"]
    val_trials = json_dict[model_name]["training_params"]["val_trials"]
    test_trials = json_dict[model_name]["training_params"]["test_trials"]
    return train_trials, val_trials, test_trials


def build_model(params):
    """
    Construct and return a TrajEncDec module based on the supplied parameters dictionary.
    """
    return seq2seq_predict.TrajEncDec(
        decoder_type=params["decoder_type"],
        rnn_type=params["rnn_type"],
        use_abs_pos=params["use_abs_pos"],
        output_seq_size=params["out_seq_len"],
        hidden_size=params["hidden_size"],
        rnn_layers=params["rnn_layers"],
        tie_enc_dec=params["tie_enc_dec"],
        use_rnn_cell=params["use_rnn_cell"],
        dropout=params["dropout"],
    )


def get_model(model_name):
    """
    Build a TrajEncDec module according to the parameter of an existing model.

    :param model_name: The model name as it appears in the model parameters json file.
    :return: a TrajEncDec module. Does not load trained weights.
    """
    params = get_models_dict()[model_name]
    return build_model(params["network_params"]), params


def update_models_json(model_name, model_dict):
    """
    Add a new model to the model parameters json file.

    :param model_name: Name of the new model.
    :param model_dict: A dictionary of training parameters, network parameters, and training results.
    """
    if os.path.exists(MODELS_JSON_PATH):
        with open(MODELS_JSON_PATH, "r") as fp:
            json_dict = json.load(fp)
    else:
        json_dict = dict()

    json_dict[model_name] = model_dict

    with open(MODELS_JSON_PATH, "w") as fp:
        json.dump(json_dict, fp, indent=4)


def train_model(
    training_params,
    net_params,
    masking_params,
    train_dataloader,
    valid_dataloader,
    num_epochs,
    eval_freq,
    log_freq,
    model=None,
    model_name_suffix=None,
    save_each_eval_model=False,
):
    """
    Train a TrajEncDec model and store its parameters and results in the json file.
    See the predictor_train jupyter notebook for a usage example.

    Training parameters: clip_grad_norm, loss_fn, lr, sched_exp (see train_eval.train_trajectory_model for details)
    Network parameters: See TrajEncDec init function for a list of parameters.

    :param training_params: A dictionary of training parameters.
    :param net_params: A dictionary of network parameters.
    :param masking_params: A dictionary of masking parameters used to subset the dataset (only for storing in the parameters dictionary).
    :param train_dataloader: Training dataloader.
    :param valid_dataloader: Validation dataloader (used for evaluating the weights).
    :param num_epochs: Number of passes over the entire training dataset.
    :param eval_freq: Number of epochs between each evaluation on the validation dataset.
    :param log_freq: Number of epochs between each time the loss data is logged.
    :param model: A TrajEncDec object used for training. Or None to build a new object based on the network parameters.
    :param model_name_suffix: An additional identifying string added to the end of the model name.
    :param save_each_eval_model: When True a weights file is saved after each evaluation, otherwise only the best and
                                 final weights are saved to file.

    :return: (model, results, interrupted), where
             model - The trained TrajEncDec object
             results - a TrainingResults object
             interrupted - True if training was interrupted by a KeyboardInterrupt exception and did not reach the last epoch.
    """
    if model is None:
        model = build_model(net_params)

    date = datetime.now().strftime("%m%d-%H%M")
    model_name = (
        f"{net_params['decoder_type']}_{net_params['rnn_type']}_{date}"
        + f"_i{net_params['inp_seq_len']}_o{net_params['out_seq_len']}"
        + f"_h{net_params['hidden_size']}_l{net_params['rnn_layers']}"
    )

    if model_name_suffix is not None:
        model_name += "_" + model_name_suffix

    results = train_eval.TrainingResults(model_name=model_name)

    print(f"Training model {model_name}:\n")

    interrupted = False

    try:
        train_eval.train_trajectory_model(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=valid_dataloader,
            epochs=num_epochs,
            path=WEIGHTS_DIR,
            clip_grad_norm=training_params["clip_grad_norm"],
            eval_freq=eval_freq,
            loss_fn=loss_fns[training_params["loss_fn"]],
            epoch_print_freq=log_freq,
            model_name=model_name,
            save_each_eval_model=save_each_eval_model,
            lr=training_params["lr"],
            sched_exp=training_params["sched_exp"],
            results=results,
        )

    except KeyboardInterrupt:
        print("Interrupted")
        interrupted = True

    except Exception as e:
        print(e)
        traceback.print_exc()

    finally:
        model_dict = {
            "network_params": net_params,
            "training_params": training_params,
            "masking_params": masking_params,
            "best_epoch": results.best_epoch,
            "best_ADE": results.best_ADE,
            "losses": results.losses,
            "ADEs": results.ADEs,
            "FDEs": results.FDEs,
        }
        update_models_json(model_name, model_dict)

        return model, results, interrupted



def pretty_name(name):
    """Transform a model name to a more fitting name for visualization."""
    return "|".join(re.sub(r"\d{4}-\d{4}", "", name).split("_"))


def plot_train_results(
        results_list,
        title,
        pretty=None,
        fname=None,
        base="../experiments_plots/",
        FDE=False,
        figsize=(14,5),
        legend=True
):
    """
    Plot ADE, FDE and loss as function of epoch for a list of training results.

    :param results_list: a list of TrainingResults objects
    :param title: Plot title
    :param pretty: A function that takes a model name and returns a suitable name for displaying in the legend
    :param fname: Filename for saving the plot. When None a default filename is used.
    :param base: Path to a directory where plots will be stored.
    """
    if FDE:
        fig, axs = plt.subplots(1, 3, figsize=(figsize[0]*2, figsize[1]))
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)

    for ax in axs:
        ax.set_xlabel("Epoch")

    axs[0].set_ylabel("ADE")
    axs[1].set_ylabel("loss")

    if FDE:
        axs[2].set_ylabel("FDE")

    for results in results_list:
        (losses, ADEs, FDEs, name) = (
            results.losses,
            results.ADEs,
            results.FDEs,
            results.model_name,
        )
        if "LSTM" in name:
            lstyle = "--"
        else:
            lstyle = "-"
        if pretty:
            name = pretty(name)

        axs[0].plot(list(ADEs.keys()), list(ADEs.values()), label=name, linestyle=lstyle)
        axs[0].scatter(list(ADEs.keys()), list(ADEs.values()), alpha=0.5, s=10)

        axs[1].plot(losses, label=name, linestyle=lstyle)

        if legend:
            axs[1].legend(fontsize=10)

        if FDE:
            axs[2].plot(list(FDEs.values()), label=name, linestyle=lstyle)
            axs[2].legend(fontsize=10)
    if not fname:
        fn = os.path.join(base, datetime.now().strftime("%m%d-%H%M") + ".jpg")
    else:
        fn = os.path.join(base, fname + ".jpg")
    fig.savefig(fn, dpi=220)
