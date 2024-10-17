# ------------------------------------------------------------------------------------------------------------#
#                             From https://github.com/gpleiss/temperature_scaling                             #
# ------------------------------------------------------------------------------------------------------------#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from betacal import BetaCalibration
from sklearn.calibration import calibration_curve
from src.metrics.metrics_utils import (
    collect_model_predictions_on_set,
    get_models_predictions,
)


def ece(y_test, preds, strategy="uniform"):
    df = pd.DataFrame({"target": y_test, "proba": preds, "bin": np.nan})

    if strategy == "uniform":
        lim_inf = np.linspace(0, 0.9, 10)
        for idx, lim in enumerate(lim_inf):
            df.loc[df["proba"] >= lim, "bin"] = idx

    elif strategy == "quantile":
        pass

    df_bin_groups = pd.concat(
        [df.groupby("bin").mean(), df["bin"].value_counts()], axis=1
    )
    df_bin_groups["ece"] = (df_bin_groups["target"] - df_bin_groups["proba"]).abs() * (
        df_bin_groups["bin"] / df.shape[0]
    )
    return df_bin_groups["ece"].sum()


class ModelBeta:
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model, parameters="abm", device="cpu"):
        super(ModelBeta, self).__init__()
        self.model = model

        self.calibrator = BetaCalibration(parameters)
        self.device = device

        try:
            self.window_1 = model.window_1
            self.window_2 = model.window_2
        except:
            pass

    def eval(self):
        self.model.eval()

    def to(self, device: str = "cpu"):
        self.model.to(device)

    # This function probably should live outside of this class, but whatever
    def fit(self, dataoader, verbose=True):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        test_out_bank, _, test_labels_bank = collect_model_predictions_on_set(
            self.model,
            dataoader,
            model_type=self.model.args["model_type"],
            device=self.device,
            verbose=verbose,
        )

        test_out_flat = torch.vstack(test_out_bank).flatten().numpy()
        test_labels_flat = torch.vstack(test_labels_bank).flatten().numpy()

        self.calibrator.fit(test_out_flat.reshape(-1, 1), test_labels_flat)

        return self

    def get_predictions(self, inputs):
        model_type = self.model.args["model_type"]

        if model_type == "tscp":
            scale, step, alpha = self.model.args["predictions"].values()
        else:
            scale, step, alpha = None, None, None

        preds, _, _ = get_models_predictions(
            inputs=inputs,
            labels=None,
            model=self.model,
            model_type=model_type,
            device=self.device,
            scale=scale,
            step=step,
            alpha=alpha,
        )
        preds = preds.detach().cpu()

        cal_preds = self.calibrator.predict(preds.flatten()).reshape(preds.shape)

        return torch.from_numpy(cal_preds)

    def predict_all(self, dataloader, verbose=False):
        test_out_bank, _, test_labels_bank = collect_model_predictions_on_set(
            self.model,
            dataloader,
            model_type=self.model.args["model_type"],
            device=self.device,
            verbose=verbose,
        )

        preds_flat = torch.vstack(test_out_bank).flatten()
        labels_flat = torch.vstack(test_labels_bank).flatten()

        preds_cal_flat = self.calibrator.predict(preds_flat)

        return preds_cal_flat, labels_flat


# ------------------------------------------------------------------------------------------------------------#
#                                         Utils for calibration                                               #
# ------------------------------------------------------------------------------------------------------------#


def calibrate_single_model(
    cpd_model,
    val_dataloader,
    cal_type="beta",
    parameters_beta="abm",
    lr=1e-2,
    max_iter=50,
    verbose=True,
    device="cpu",
):
    assert cal_type == "beta", f"Unknown calibration type {cal_type}"

    cpd_model.to(device)

    scaled_model = ModelBeta(
        cpd_model,
        parameters=parameters_beta,
        # preprocessor=preprocessor,
        device=device,
    )
    scaled_model.fit(val_dataloader, verbose=verbose)

    return scaled_model


def calibrate_all_models_in_ensemble(
    ensemble_model,
    val_dataloader,
    cal_type,
    lr=1e-2,
    max_iter=50,
    verbose=True,
    device="cpu",
):
    cal_models = []
    for cpd_model in ensemble_model.models_list:
        cal_model = calibrate_single_model(
            cpd_model,
            val_dataloader,
            cal_type,
            lr=lr,
            max_iter=max_iter,
            verbose=verbose,
            device=device,
        )
        cal_models.append(cal_model)

    ensemble_model.models_list = cal_models
    ensemble_model.calibrated = True

    return cal_models



def plot_calibration_curves(
    ens_model,
    test_dataloader,
    model_type="seq2seq",
    calibrated=True,
    device="cpu",
    n_bins=10,
    evaluate=False,
    fontsize=12,
    title=None,
    verbose=False,
    savename=None,
    model_num=None,
):
    if not model_num:
        model_num = len(ens_model.models_list)

    x_ideal = np.linspace(0, 1, n_bins)

    plt.figure(figsize=(8, 6))
    plt.plot(
        x_ideal,
        x_ideal,
        linestyle="--",
        label="Ideal",
        c="black",
        linewidth=2,
    )

    if evaluate:
        ece_list = []

    for i, model in enumerate(ens_model.models_list[:model_num]):
        if calibrated:
            test_out_flat, test_labels_flat = model.predict_all(
                test_dataloader, verbose=verbose
            )
        else:
            test_out_bank, _, test_labels_bank = collect_model_predictions_on_set(
                model,
                test_dataloader,
                model_type=model_type,
                device=device,
                verbose=verbose,
            )
            test_out_flat = torch.vstack(test_out_bank).flatten()
            test_labels_flat = torch.vstack(test_labels_bank).flatten()

        if evaluate:
            try:
                ece_list.append(ece(test_labels_flat.numpy(), test_out_flat.numpy()))
            except AttributeError:
                ece_list.append(ece(test_labels_flat, test_out_flat))
        prob_true, prob_pred = calibration_curve(
            test_labels_flat, test_out_flat, n_bins=n_bins
        )

        plt.plot(
            prob_pred,
            prob_true,
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=1,
            label=f"Model {i}",
        )
    if evaluate:
        bbox = dict(boxstyle="round", fc="blanchedalmond", ec="orange", alpha=0.5)
        plt.text(
            x=0.49,
            y=0.00,
            s="Calibration Error = {:.4f}".format(np.round(np.mean(ece_list), 4)),  # noqa: F523
            fontsize=fontsize,
            bbox=bbox,
        )
    if title:
        plt.title(title, fontsize=fontsize + 2)
    plt.xlabel("Predicted probability", fontsize=fontsize)
    plt.ylabel("Fraction of positives", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize - 1)
    plt.tight_layout()
    if savename:
        plt.savefig(f"pictures/calibration/curves/{savename}.pdf", dpi=300)
    plt.show()
