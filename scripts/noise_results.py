import RNNmpc
import torch
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("--data_dict", type=str, help="Data dictionary.", required=True)

parser.add_argument(
    "--best_hyperparams_dict",
    type=str,
    help="Dictionary containing results of hyperparameter tuning.",
    required=True,
)

parser.add_argument(
    "--dest",
    type=str,
    help="Destination for plot to be saved.",
    required=True,
)

parser.add_argument(
    "--noise_levels",
    type=list[int],
    help="Amount of noise to add to data.",
    required=False,
    default=[0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
)


args = parser.parse_args()
best_params_dict = json.load(open(args.best_hyperparams_dict))
data_dict = json.load(open(args.data_dict))

noise_levels = args.noise_levels
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### ESN
alpha = best_params_dict["ESNForecaster"]["alpha"]
sigma = best_params_dict["ESNForecaster"]["sigma"]
sigma_b = best_params_dict["ESNForecaster"]["sigma_b"]
beta = best_params_dict["ESNForecaster"]["beta"]
rho_sr = best_params_dict["ESNForecaster"]["rho_sr"]

esn_list = []
esn_dict = {}
for noise in noise_levels:
    ret_dict, _ = RNNmpc.utils.forecast_eval(
        "ESNForecaster",
        data_dict=data_dict,
        scaled=True,
        alpha=alpha,
        rho_sr=rho_sr,
        beta=beta,
        sigma_b=sigma_b,
        sigma=sigma,
        noise_level=noise,
        device=device,
    )

    esn_list.append(ret_dict["fcast_dev"])


adam_lr = best_params_dict["LSTMForecaster"]["adam_lr"]
dropout_p = best_params_dict["LSTMForecaster"]["dropout_p"]
lags = best_params_dict["LSTMForecaster"]["lags"]
Nr = best_params_dict["LSTMForecaster"]["Nr"]
lstm_list = []
for noise in noise_levels:
    ret_dict, _ = RNNmpc.utils.forecast_eval(
        "LSTMForecaster",
        data_dict=data_dict,
        scaled=True,
        adam_lr=adam_lr,
        lags=lags,
        hidden_dim=Nr,
        dropout_p=dropout_p,
        noise_level=noise,
        device=device,
    )

    lstm_list.append(ret_dict["fcast_dev"])

adam_lr = best_params_dict["GRUForecaster"]["adam_lr"]
dropout_p = best_params_dict["GRUForecaster"]["dropout_p"]
lags = best_params_dict["GRUForecaster"]["lags"]
Nr = best_params_dict["GRUForecaster"]["Nr"]
gru_list = []
for noise in noise_levels:
    ret_dict, _ = RNNmpc.utils.forecast_eval(
        "GRUForecaster",
        data_dict=data_dict,
        scaled=True,
        adam_lr=adam_lr,
        lags=lags,
        hidden_dim=Nr,
        dropout_p=dropout_p,
        noise_level=noise,
        device=device,
    )

    gru_list.append(ret_dict["fcast_dev"])


beta = best_params_dict["LinearForecaster"]["beta"]
tds = best_params_dict["LinearForecaster"]["tds"]
lin_list = []
for noise in noise_levels:
    ret_dict, _ = RNNmpc.utils.forecast_eval(
        "LinearForecaster",
        data_dict=data_dict,
        scaled=True,
        beta=beta,
        tds=tds,
        noise_level=noise,
        device=device,
    )

    lin_list.append(ret_dict["fcast_dev"])

adam_lr = best_params_dict["FCForecaster"]["adam_lr"]
tds = best_params_dict["FCForecaster"]["tds"]
r_width = best_params_dict["FCForecaster"]["r_width"]
dropout_p = best_params_dict["FCForecaster"]["dropout_p"]

fc_list = []
for noise in noise_levels:
    ret_dict, _ = RNNmpc.utils.forecast_eval(
        "FCForecaster",
        data_dict=data_dict,
        scaled=True,
        r_width=r_width,
        tds=tds,
        dropout_p=dropout_p,
        noise_level=noise,
        device=device,
    )

    fc_list.append(ret_dict["fcast_dev"])

fig, ax = plt.subplots()
ax.semilogy(noise_levels, lin_list, label="DMDc", linewidth=2)
ax.semilogy(noise_levels, fc_list, label="FCN", linewidth=2)
ax.semilogy(noise_levels, gru_list, label="GRU", linewidth=2)
ax.semilogy(noise_levels, lstm_list, label="LSTM", linewidth=2)
ax.semilogy(noise_levels, esn_list, label="ESN", linewidth=2)
ax.tick_params(axis="both", which="major", labelsize=14)
ax.tick_params(axis="both", which="minor", labelsize=14)
ax.set_xlabel("Noise, $\sigma_{STD}$", fontsize=18)
ax.set_ylabel("Error", fontsize=18)
ax.set_xlim([0, max(noise_levels) * 1.35])
ax.legend(fontsize=15, loc="center right")
fig.savefig(args.dest + "/noise.pdf")
