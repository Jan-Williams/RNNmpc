import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from RNNmpc.utils import forecast_eval

plt.style.use("seaborn-v0_8")
import json

parser = argparse.ArgumentParser()

parser.add_argument(
    "--best_hyperparams_dict",
    type=str,
    help="json file containing output of hyperparam_plots.py",
    required=True,
)

parser.add_argument(
    "--data_dict",
    type=str,
    help="Training and evaluation data dictionary.",
    required=True,
)

parser.add_argument(
    "--dest",
    type=str,
    help="Directory to which files are saved.",
    required=True,
)

parser.add_argument("--noise_level", type=float, default=0.00, required=False)

args = parser.parse_args()
best_params_dict = json.load(open(args.best_hyperparams_dict))
data_dict = json.load(open(args.data_dict))
noise_level = args.noise_level
dest = args.dest

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_steps = 5000

U_train = np.array(data_dict["U_train"])[:, -train_steps:]
S_train = np.array(data_dict["S_train"])[:, -train_steps:]
O_train = np.array(data_dict["O_train"])[:, -train_steps:]

U_valid = np.array(data_dict["U_valid"])
S_valid = np.array(data_dict["S_valid"])
O_valid = np.array(data_dict["O_valid"])

U_train = torch.tensor(U_train, dtype=torch.float64)
S_train = torch.tensor(S_train, dtype=torch.float64)
O_train = torch.tensor(O_train, dtype=torch.float64)

U_valid = torch.tensor(U_valid, dtype=torch.float64)
S_valid = torch.tensor(S_valid, dtype=torch.float64)
O_valid = torch.tensor(O_valid, dtype=torch.float64)


concat_U = torch.hstack((U_train, U_valid))
concat_S = torch.hstack((S_train, S_valid))
concat_O = torch.hstack((O_train, O_valid))


### Train models
Nu = U_train.shape[0]
Ns = S_train.shape[0]
No = O_train.shape[0]
fcast_steps = int(data_dict["fcast_lens"] / data_dict["control_disc"])

### Train ESN
alpha = best_params_dict["ESNForecaster"]["alpha"]
sigma_b = best_params_dict["ESNForecaster"]["sigma_b"]
sigma = best_params_dict["ESNForecaster"]["sigma"]
beta = best_params_dict["ESNForecaster"]["beta"]
rho_sr = best_params_dict["ESNForecaster"]["rho_sr"]

_, forecast_esn = forecast_eval(
    "ESNForecaster",
    data_dict=data_dict,
    alpha=alpha,
    sigma_b=sigma_b,
    sigma=sigma,
    beta=beta,
    rho_sr=rho_sr,
    scaled=True,
    noise_level=noise_level,
    device=device,
)

O_valid = O_valid.detach().cpu().numpy()
esn_dev_list = np.linalg.norm(forecast_esn - O_valid, axis=0)

### Train Linear
beta = best_params_dict["LinearForecaster"]["beta"]
tds = best_params_dict["LinearForecaster"]["tds"]

_, forecast_linear = forecast_eval(
    "LinearForecaster",
    data_dict=data_dict,
    beta=beta,
    tds=tds,
    noise_level=noise_level,
    scaled=True,
    device=device,
)
linear_dev_list = np.linalg.norm(forecast_linear - O_valid, axis=0)


## Train FC
tds = best_params_dict["FCForecaster"]["tds"]
lr = best_params_dict["FCForecaster"]["adam_lr"]
r_width = best_params_dict["FCForecaster"]["r_width"]
dropout_p = best_params_dict["FCForecaster"]["dropout_p"]

_, forecast_fc = forecast_eval(
    "FCForecaster",
    data_dict=data_dict,
    adam_lr=lr,
    tds=tds,
    dropout_p=dropout_p,
    noise_level=noise_level,
    scaled=True,
    device=device,
)
fc_dev_list = np.linalg.norm(forecast_fc - O_valid, axis=0)


### Train GRU
hidden_dim = best_params_dict["GRUForecaster"]["Nr"]
lags = best_params_dict["GRUForecaster"]["lags"]
dropout_p = best_params_dict["GRUForecaster"]["dropout_p"]
lr = best_params_dict["GRUForecaster"]["adam_lr"]
Nr = hidden_dim

_, forecast_gru = forecast_eval(
    "GRUForecaster",
    data_dict=data_dict,
    adam_lr=lr,
    lags=lags,
    dropout_p=dropout_p,
    hidden_dim=Nr,
    noise_level=noise_level,
    scaled=True,
    device=device,
)
gru_dev_list = np.linalg.norm(forecast_gru - O_valid, axis=0)

### Train LSTM
hidden_dim = best_params_dict["LSTMForecaster"]["Nr"]
lags = best_params_dict["LSTMForecaster"]["lags"]
dropout_p = best_params_dict["LSTMForecaster"]["dropout_p"]
lr = best_params_dict["LSTMForecaster"]["adam_lr"]
Nr = hidden_dim

_, forecast_lstm = forecast_eval(
    "LSTMForecaster",
    data_dict=data_dict,
    adam_lr=lr,
    lags=lags,
    dropout_p=dropout_p,
    hidden_dim=Nr,
    noise_level=noise_level,
    scaled=True,
    device=device,
)
lstm_dev_list = np.linalg.norm(forecast_lstm - O_valid, axis=0)

U_valid = np.array(data_dict["U_valid"])

results_dict = {
    'U_valid': U_valid.tolist(),
    'lstm_dev': lstm_dev_list.tolist(),
    'esn_dev': esn_dev_list.tolist(),
    'fc_dev': fc_dev_list.tolist(),
    'gru_dev': gru_dev_list.tolist(),
    'lin_dev': linear_dev_list.tolist(),
    'O_valid': O_valid.tolist()
}

with open(args.dest + "/forecast_results.json", "w") as fp:
    json.dump(results_dict, fp)


# fig, ax = plt.subplots(3)
# # fig.suptitle(data_dict['simulator'], fontsize=20)
# for ii in range(O_valid.shape[0]):
#     label = "$x_{index}$"
#     ax[0].plot(
#         t_range,
#         (O_valid[ii] - np.mean(O_valid[ii])) / np.std(O_valid[ii]),
#         label=label.format(index=ii + 1),
#         linewidth=2,
#     )
# for ii in range(10):
#     if ii == 0:
#         ax[0].axvline(
#             x=fcast_steps * data_dict["control_disc"] * ii,
#             linestyle="--",
#             color="k",
#             label="Intervals",
#         )
#     else:
#         ax[0].axvline(
#             x=fcast_steps * data_dict["control_disc"] * ii, linestyle="--", color="k"
#         )
# ax[0].legend(fontsize=15, loc="center right")
# ax[0].set_xlim([-0.1, O_valid.shape[1] * data_dict["control_disc"] * 1.3])
# ax[0].tick_params(axis="both", which="major", labelsize=14)
# ax[0].tick_params(axis="both", which="minor", labelsize=14)

# for ii in range(U_valid.shape[0]):
#     c = "k" if ii == 0 else "tab:red"
#     label = "$u_{index}$"
#     ax[1].plot(t_range, U_valid[ii], linewidth=2, label=label.format(index=ii + 1), c=c)
# ax[0].set_ylabel("Ground Truth", fontsize=18)

# ax[1].set_xlim([-0.1, O_valid.shape[1] * data_dict["control_disc"] * 1.3])
# ax[1].set_ylabel("Control", fontsize=18)
# ax[1].legend(fontsize=15, loc="center right")
# ax[1].tick_params(axis="both", which="major", labelsize=14)
# ax[1].tick_params(axis="both", which="minor", labelsize=14)
# ax[2].semilogy(t_range, linear_dev_list, linewidth=2, label="DMDc")
# ax[2].semilogy(t_range, fc_dev_list, linewidth=2, label="FCN")
# ax[2].semilogy(t_range, gru_dev_list, linewidth=2, label="GRU")
# ax[2].semilogy(t_range, lstm_dev_list, linewidth=2, label="LSTM")
# ax[2].semilogy(t_range, esn_dev_list, linewidth=2, label="ESN")
# ax[2].legend(fontsize=15, loc="center right")
# ax[2].set_ylabel("Error", fontsize=18)
# ax[2].set_xlim([-0.1, O_valid.shape[1] * data_dict["control_disc"] * 1.3])
# ax[2].set_xlabel("$t$", fontsize=18)
# ax[2].set_yticks([1e-3, 1e-1, 1e1])
# ax[2].tick_params(axis="both", which="major", labelsize=14)
# ax[2].tick_params(axis="both", which="minor", labelsize=14)

# plt.tight_layout()

# fig.savefig(dest + "/forecast_plot.pdf")
