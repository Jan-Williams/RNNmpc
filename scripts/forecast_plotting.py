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


fig, ax = plt.subplots(3, figsize=(6.5, 3.9))
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
ax[0].legend(fontsize=10, loc="center right")
ax[0].set_xlim([-0.1, O_valid.shape[1] * disc * 1.3])
ax[0].tick_params(axis="both", which="major", labelsize=12)
ax[0].tick_params(axis="both", which="minor", labelsize=12)

for ii in range(U_valid.shape[0]):
    c = "k" if ii == 0 else "tab:red"
    label = "$u_{index}$"
    ax[1].plot(t_range, U_valid[ii], linewidth=2, label=label.format(index=ii + 1), c=c)
ax[0].set_ylabel("Ground Truth", fontsize=12)
ax[0].set_yticks([-2.5, 0, 2.5])

ax[1].set_xlim([-0.1, O_valid.shape[1] * disc * 1.3])
ax[1].set_ylabel("Control", fontsize=12)
ax[1].legend(fontsize=10, loc="center right")
ax[1].tick_params(axis="both", which="major", labelsize=12)
ax[1].tick_params(axis="both", which="minor", labelsize=12)
ax[2].semilogy(t_range, linear_dev_list, linewidth=2, label="DMDc")
ax[2].semilogy(t_range, fc_dev_list, linewidth=2, label="FCN")
ax[2].semilogy(t_range, gru_dev_list, linewidth=2, label="GRU")
ax[2].semilogy(t_range, lstm_dev_list, linewidth=2, label="LSTM")
ax[2].semilogy(t_range, esn_dev_list, linewidth=2, label="ESN")
ax[2].legend(fontsize=10, loc="center right")
ax[2].set_ylabel("$||x_t - \hat{x}_t||_2$", fontsize=12)
ax[2].set_xlim([-0.1, O_valid.shape[1] * disc * 1.3])
ax[2].set_xlabel("Time, $t$", fontsize=12)
ax[2].set_yticks(ticks)
ax[2].tick_params(axis="both", which="major", labelsize=12)
ax[2].tick_params(axis="both", which="minor", labelsize=12)


ax[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

ax[1].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

plt.tight_layout()

fig.savefig(dest + "/forecast_plot.pdf")