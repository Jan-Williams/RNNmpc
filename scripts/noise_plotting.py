import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-v0_8')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--noise_results_dict",
    type=str,
    help="Path to noise_results output.",
    required=True,
)

parser.add_argument(
    "--dest",
    type=str,
    help="Path to noise_results output.",
    required=True,
)

args = parser.parse_args()
noise_results = json.load(open(args.noise_results_dict))
dest = args.dest

esn_list = np.array(noise_results['esn_list'])
esn_mean_list = esn_list.mean(axis=1)
esn_low_error = np.clip(esn_mean_list - np.quantile(esn_list, q=0.10, axis=1),0,100)
esn_high_error = np.clip(np.quantile(esn_list, q=0.90, axis=1) - esn_mean_list,0,100)

lstm_list = np.array(noise_results['lstm_list'])
lstm_mean_list = lstm_list.mean(axis=1)
lstm_low_error = np.clip(lstm_mean_list - np.quantile(lstm_list, q=0.10, axis=1),0, 100)
lstm_high_error = np.clip(np.quantile(lstm_list, q=0.90, axis=1) - lstm_mean_list,0,100)

gru_list = np.array(noise_results['gru_list'])
gru_mean_list = gru_list.mean(axis=1)
gru_low_error = np.clip(gru_mean_list - np.quantile(gru_list, q=0.10, axis=1),0,100)
gru_high_error = np.clip(np.quantile(gru_list, q=0.90, axis=1) - gru_mean_list,0,100)

fc_list = np.array(noise_results['fc_list'])
fc_mean_list = fc_list.mean(axis=1)
fc_low_error = np.clip(fc_mean_list - np.quantile(fc_list, q=0.10, axis=1), 0, 100)
fc_high_error = np.clip(np.quantile(fc_list, q=0.90, axis=1) - fc_mean_list, 0, 100)

lin_list = np.array(noise_results['lin_list'])
lin_mean_list = lin_list.mean(axis=1)
lin_low_error = np.clip(lin_mean_list - np.quantile(lin_list, q=0.10, axis=1), 0, 100)
lin_high_error = np.clip(np.quantile(lin_list, q=0.90, axis=1) - lin_mean_list,0,100)

noise_levels = np.array(noise_results['noise_levels'])

fig, ax = plt.subplots()

ax.errorbar((noise_levels - 0.002)*100, lin_mean_list, yerr=(lin_low_error, lin_high_error), fmt='.', linewidth=5, markersize=25, label='DMDc')
ax.errorbar((noise_levels - 0.001)*100, fc_mean_list, yerr=(fc_low_error, fc_high_error), fmt='.', linewidth=5, markersize=25, label='FCN')
ax.errorbar((noise_levels)*100, gru_mean_list, yerr=(gru_low_error, gru_high_error), fmt='.', linewidth=5, markersize=25, label='GRU')
ax.errorbar((noise_levels + 0.001)*100, lstm_mean_list, yerr=(lstm_low_error, lstm_high_error), fmt='.', linewidth=5, markersize=25, label='LSTM')
ax.errorbar((noise_levels + 0.002)*100, esn_mean_list, yerr=(esn_low_error, esn_high_error), fmt='.', linewidth=5, markersize=25, label='ESN')
ax.legend(fontsize=20, loc='lower right')
ax.set_xlim([-1, max(noise_levels)*100 * 1.3])
ax.tick_params(axis="both", which="major", labelsize=24)
ax.tick_params(axis="both", which="minor", labelsize=24)
ax.set_ylabel('Error', fontsize=35)
ax.set_xlabel('Noise, $\%\sigma _{train}$', fontsize=35)
ax.set_yscale('log')
plt.tight_layout()
fig.savefig(dest + '/noise_plot.pdf')