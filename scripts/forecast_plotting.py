import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
plt.style.use('seaborn-v0_8')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--forecast_results",
    type=str,
    required=True,
    help="Output file of forecast_results.py"
)

parser.add_argument(
    "--dest",
    required=True,
    type=str,
    help="Save location."
)

parser.add_argument(
    "--disc",
    type=float,
    required=True,
    help="Discretization of forecast."
)

parser.add_argument(
    "--ticks",
    required=False,
    help="Ticks for error plot.",
    default=[1e-5, 1e-3, 1e-1,],
    nargs='+',
    type=float,
)

args = parser.parse_args()
dest = args.dest
disc = args.disc
ticks = args.ticks
forecast_results = args.forecast_results
forecast_results = json.load(open(forecast_results))

O_valid = np.array(forecast_results['O_valid'])
U_valid = np.array(forecast_results['U_valid'])
t_range = np.arange(0, O_valid.shape[1], 1) * disc

fcast_steps = int(O_valid.shape[1] / 10)

linear_dev_list = np.array(forecast_results['lin_dev'])
esn_dev_list = np.array(forecast_results['esn_dev'])
lstm_dev_list = np.array(forecast_results['lstm_dev'])
gru_dev_list = np.array(forecast_results['gru_dev'])
fc_dev_list = np.array(forecast_results['fc_dev'])


fig, ax = plt.subplots(3)
for ii in range(O_valid.shape[0]):
    label = "$x_{index}$"
    ax[0].plot(
        t_range,
        (O_valid[ii] - np.mean(O_valid[ii])) / np.std(O_valid[ii]),
        label=label.format(index=ii + 1),
        linewidth=2,
    )
for ii in range(10):
    if ii == 0:
        ax[0].axvline(
            x=fcast_steps * disc * ii,
            linestyle="--",
            color="k",
            label="Intervals",
        )
    else:
        ax[0].axvline(
            x=fcast_steps * disc * ii, linestyle="--", color="k"
        )
ax[0].legend(fontsize=15, loc="center right")
ax[0].set_xlim([-0.1, O_valid.shape[1] * disc * 1.3])
ax[0].tick_params(axis="both", which="major", labelsize=14)
ax[0].tick_params(axis="both", which="minor", labelsize=14)

for ii in range(U_valid.shape[0]):
    c = "k" if ii == 0 else "tab:red"
    label = "$u_{index}$"
    ax[1].plot(t_range, U_valid[ii], linewidth=2, label=label.format(index=ii + 1), c=c)
ax[0].set_ylabel("Ground Truth", fontsize=18)

ax[1].set_xlim([-0.1, O_valid.shape[1] * disc * 1.3])
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
ax[2].set_xlim([-0.1, O_valid.shape[1] * disc * 1.3])
ax[2].set_xlabel("$t$", fontsize=18)
ax[2].set_yticks(ticks)
ax[2].tick_params(axis="both", which="major", labelsize=14)
ax[2].tick_params(axis="both", which="minor", labelsize=14)

plt.tight_layout()

fig.savefig(dest + "/forecast_plot.pdf")