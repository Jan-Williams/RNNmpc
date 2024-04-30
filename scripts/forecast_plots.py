import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")
import json
from sklearn.preprocessing import MinMaxScaler
import RNNmpc.Forecasters as Forecaster

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

args = parser.parse_args()
best_params_dict = args.best_hyperparams_dict
best_params_dict = json.load(open(best_params_dict))
data_dict = args.data_dict
data_dict = json.load(open(data_dict))
dest = args.dest

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
t_range = np.arange(0, O_valid.shape[1], 1) * data_dict["control_disc"]

### Train ESN
alpha = best_params_dict["ESNForecaster"]["alpha"]
sigma_b = best_params_dict["ESNForecaster"]["sigma_b"]
sigma = best_params_dict["ESNForecaster"]["sigma"]
beta = best_params_dict["ESNForecaster"]["beta"]
rho_sr = best_params_dict["ESNForecaster"]["rho_sr"]
esn = Forecaster.ESNForecaster(
    Nr=1000,
    Nu=Nu,
    Ns=Ns,
    No=No,
    alpha=alpha,
    sigma_b=sigma_b,
    sigma=sigma,
    rho_sr=rho_sr,
)
esn_r = esn.fit(U_train, S_train, O_train, beta=beta)
forecast_esn = torch.zeros((No, 0)).to("cpu")

for i in range(10):
    fcast_start_index = U_train.shape[1] + i * fcast_steps
    spin_r = esn.spin(
        U_spin=concat_U[:, fcast_start_index - 500 : fcast_start_index],
        S_spin=concat_S[:, fcast_start_index - 500 : fcast_start_index],
    )
    fcast = esn.forecast(
        concat_U[:, fcast_start_index : fcast_start_index + fcast_steps],
        r_k=spin_r,
        s_k=concat_S[:, fcast_start_index : fcast_start_index + 1],
    )
    forecast_esn = torch.hstack((forecast_esn, fcast.to("cpu")))

forecast_esn = forecast_esn.detach().cpu().numpy()
O_valid = O_valid.detach().cpu().numpy()
esn_dev_list = np.linalg.norm(forecast_esn - O_valid, axis=0)

### Train Linear
beta = best_params_dict["LinearForecaster"]["beta"]
tds = best_params_dict["LinearForecaster"]["tds"]

linear = Forecaster.LinearForecaster(Nu=Nu, Ns=Ns, No=No, tds=tds)
linear.fit(U_train, S_train, O_train, beta=beta)

forecast_linear = torch.zeros((No, 0)).to("cpu")

for i in range(10):
    fcast_start_index = U_train.shape[1] + i * fcast_steps
    fcast = linear.forecast(
        U=concat_U[:, fcast_start_index : fcast_start_index + fcast_steps],
        U_spin=concat_U[
            :, fcast_start_index - max(abs(np.array(tds))) + 1 : fcast_start_index
        ],
        S_spin=concat_S[
            :, fcast_start_index - max(abs(np.array(tds))) + 1 : fcast_start_index
        ],
        s_k=concat_S[:, fcast_start_index : fcast_start_index + 1],
    )
    forecast_linear = torch.hstack((forecast_linear, fcast.to("cpu")))

forecast_linear = forecast_linear.detach().cpu().numpy()
linear_dev_list = np.linalg.norm(forecast_linear - O_valid, axis=0)

### Reinitialize for scaled data

U_train = np.array(data_dict["U_train"])[:, -train_steps:]
S_train = np.array(data_dict["S_train"])[:, -train_steps:]
O_train = np.array(data_dict["O_train"])[:, -train_steps:]

U_valid = np.array(data_dict["U_valid"])
S_valid = np.array(data_dict["S_valid"])

sensor_scaler = MinMaxScaler()
S_train = sensor_scaler.fit_transform(S_train.T).T
S_valid = sensor_scaler.transform(S_valid.T).T
O_train = sensor_scaler.transform(O_train.T).T

control_scaler = MinMaxScaler()
U_train = control_scaler.fit_transform(U_train.T).T
U_valid = control_scaler.transform(U_valid.T).T

U_train = torch.tensor(U_train, dtype=torch.float64)
S_train = torch.tensor(S_train, dtype=torch.float64)
O_train = torch.tensor(O_train, dtype=torch.float64)

U_valid = torch.tensor(U_valid, dtype=torch.float64)
S_valid = torch.tensor(S_valid, dtype=torch.float64)

concat_U = torch.hstack((U_train, U_valid))
concat_S = torch.hstack((S_train, S_valid))


### Train FC
tds = best_params_dict["FCForecaster"]["tds"]
lr = best_params_dict["FCForecaster"]["adam_lr"]
r_width = best_params_dict["FCForecaster"]["r_width"]
dropout_p = best_params_dict["FCForecaster"]["dropout_p"]

model = Forecaster.FCForecaster(
    Nu=Nu, Ns=Ns, No=No, tds=tds, r_list=[r_width, r_width], dropout_p=dropout_p
)
model.fit(U_train, S_train, O_train, lr=lr)

forecast_fc = torch.zeros((No, 0)).to("cpu")

for i in range(10):
    fcast_start_index = U_train.shape[1] + i * fcast_steps
    fcast = model.forecast(
        U=concat_U[:, fcast_start_index : fcast_start_index + fcast_steps],
        U_spin=concat_U[
            :, fcast_start_index - max(abs(np.array(tds))) + 1 : fcast_start_index
        ],
        S_spin=concat_S[
            :, fcast_start_index - max(abs(np.array(tds))) + 1 : fcast_start_index
        ],
        s_k=concat_S[:, fcast_start_index : fcast_start_index + 1],
    )
    forecast_fc = torch.hstack((forecast_fc, fcast.to("cpu")))

forecast_fc = forecast_fc.detach().cpu().numpy()
forecast_fc = sensor_scaler.inverse_transform(forecast_fc.T).T
fc_dev_list = np.linalg.norm(forecast_fc - O_valid, axis=0)


### Train GRU
hidden_dim = best_params_dict["GRUForecaster"]["Nr"]
lags = best_params_dict["GRUForecaster"]["lags"]
dropout_p = best_params_dict["GRUForecaster"]["dropout_p"]
lr = best_params_dict["GRUForecaster"]["adam_lr"]
Nr = hidden_dim
model = Forecaster.GRUForecaster(Nr=Nr, Nu=Nu, Ns=Ns, No=No, dropout_p=dropout_p)
out_r = model.fit(U_train, S_train, O_train, lags=lags, lr=lr)

forecast_gru = torch.zeros((No, 0)).to("cpu")

for i in range(10):
    fcast_start_index = U_train.shape[1] + i * fcast_steps
    fcast = model.forecast(
        U=concat_U[:, fcast_start_index : fcast_start_index + fcast_steps],
        U_spin=concat_U[:, fcast_start_index - lags + 1 : fcast_start_index],
        S_spin=concat_S[:, fcast_start_index - lags + 1 : fcast_start_index],
        s_k=concat_S[:, fcast_start_index : fcast_start_index + 1],
    )
    forecast_gru = torch.hstack((forecast_gru, fcast.to("cpu")))

forecast_gru = forecast_gru.detach().cpu().numpy()
forecast_gru = sensor_scaler.inverse_transform(forecast_gru.T).T
gru_dev_list = np.linalg.norm(forecast_gru - O_valid, axis=0)


### Train LSTM
hidden_dim = best_params_dict["LSTMForecaster"]["Nr"]
lags = best_params_dict["LSTMForecaster"]["lags"]
dropout_p = best_params_dict["LSTMForecaster"]["dropout_p"]
lr = best_params_dict["LSTMForecaster"]["adam_lr"]
Nr = hidden_dim
model = Forecaster.LSTMForecaster(Nr=Nr, Nu=Nu, Ns=Ns, No=No, dropout_p=dropout_p)
out_r = model.fit(U_train, S_train, O_train, lags=lags, lr=lr)

forecast_lstm = torch.zeros((No, 0)).to("cpu")

for i in range(10):
    fcast_start_index = U_train.shape[1] + i * fcast_steps
    fcast = model.forecast(
        U=concat_U[:, fcast_start_index : fcast_start_index + fcast_steps],
        U_spin=concat_U[:, fcast_start_index - lags + 1 : fcast_start_index],
        S_spin=concat_S[:, fcast_start_index - lags + 1 : fcast_start_index],
        s_k=concat_S[:, fcast_start_index : fcast_start_index + 1],
    )
    forecast_lstm = torch.hstack((forecast_lstm, fcast.to("cpu")))

forecast_lstm = forecast_lstm.detach().cpu().numpy()
forecast_lstm = sensor_scaler.inverse_transform(forecast_lstm.T).T
lstm_dev_list = np.linalg.norm(forecast_lstm - O_valid, axis=0)

U_valid = control_scaler.inverse_transform(U_valid.detach().numpy().T).T


fig, ax = plt.subplots(3)
# fig.suptitle(data_dict['simulator'], fontsize=20)
for ii in range(O_valid.shape[0]):
    label = "$x_{index}$"
    ax[0].plot(t_range, (O_valid[ii] - np.mean(O_valid[ii])) / np.mean(O_valid[ii]), label=label.format(index=ii + 1), linewidth=2)
for ii in range(10):
    if ii == 0:
        ax[0].axvline(
            x=fcast_steps * data_dict["control_disc"] * ii,
            linestyle="--",
            color="k",
            label="Intervals",
        )
    else:
        ax[0].axvline(
            x=fcast_steps * data_dict["control_disc"] * ii, linestyle="--", color="k"
        )
ax[0].legend(fontsize=15, loc="center right")
ax[0].set_xlim([-0.1, O_valid.shape[1] * data_dict["control_disc"] * 1.3])
ax[0].tick_params(axis="both", which="major", labelsize=14)
ax[0].tick_params(axis="both", which="minor", labelsize=14)

for ii in range(U_valid.shape[0]):
    c = "k" if ii == 0 else "tab:red"
    label = "$u_{index}$"
    ax[1].plot(t_range, U_valid[ii], linewidth=2, label=label.format(index=ii + 1), c=c)
ax[0].set_ylabel("Ground Truth", fontsize=18)

ax[1].set_xlim([-0.1, O_valid.shape[1] * data_dict["control_disc"] * 1.3])
ax[1].set_ylabel("Control", fontsize=18)
ax[1].legend(fontsize=15, loc="center right")
ax[1].tick_params(axis="both", which="major", labelsize=14)
ax[1].tick_params(axis="both", which="minor", labelsize=14)
ax[2].semilogy(t_range, linear_dev_list, linewidth=2, label="DMDc")
ax[2].semilogy(t_range, fc_dev_list, linewidth=2, label="FCN")
ax[2].semilogy(t_range, gru_dev_list, linewidth=2, label="GRU")
ax[2].semilogy(t_range, lstm_dev_list, linewidth=2, label="LSTM")
ax[2].semilogy(t_range, esn_dev_list, linewidth=2, label="ESN")
ax[2].legend(fontsize=15, loc="center right")
ax[2].set_ylabel("Error", fontsize=18)
ax[2].set_xlim([-0.1, O_valid.shape[1] * data_dict["control_disc"] * 1.3])
ax[2].set_xlabel("$t$", fontsize=18)
ax[2].set_yticks([1e-5, 1e-3, 1e-1])
ax[2].tick_params(axis="both", which="major", labelsize=14)
ax[2].tick_params(axis="both", which="minor", labelsize=14)

plt.tight_layout()

fig.savefig(dest + "/forecast_plot.pdf")
