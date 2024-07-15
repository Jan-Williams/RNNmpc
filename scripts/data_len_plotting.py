import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
plt.style.use('seaborn-v0_8')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--train_len_results_dict",
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
len_results = json.load(open(args.train_len_results_dict))
dest = args.dest

esn_list = np.array(len_results['esn_list'])
esn_mean_list = esn_list.mean(axis=1)
esn_low_error = np.clip(esn_mean_list - np.quantile(esn_list, q=0.10, axis=1), 0, 100)
esn_high_error = np.clip(np.quantile(esn_list, q=0.90, axis=1) - esn_mean_list, 0, 100)

lstm_list = np.array(len_results['lstm_list'])
lstm_mean_list = lstm_list.mean(axis=1)
lstm_low_error = np.clip(lstm_mean_list - np.quantile(lstm_list, q=0.10, axis=1), 0, 100)
lstm_high_error = np.clip(np.quantile(lstm_list, q=0.90, axis=1) - lstm_mean_list,0,100)

gru_list = np.array(len_results['gru_list'])
gru_mean_list = gru_list.mean(axis=1)
gru_low_error = np.clip(gru_mean_list - np.quantile(gru_list, q=0.10, axis=1),0,100)
gru_high_error = np.clip(np.quantile(gru_list, q=0.90, axis=1) - gru_mean_list,0,100)

fc_list = np.array(len_results['fc_list'])
fc_mean_list = fc_list.mean(axis=1)
fc_low_error = np.clip(fc_mean_list - np.quantile(fc_list, q=0.10, axis=1),0,100)
fc_high_error = np.clip(np.quantile(fc_list, q=0.90, axis=1) - fc_mean_list,0,100)

lin_list = np.array(len_results['lin_list'])
lin_mean_list = lin_list.mean(axis=1)
lin_low_error = np.clip(lin_mean_list - np.quantile(lin_list, q=0.10, axis=1), 0, 100)
lin_high_error = np.clip(np.quantile(lin_list, q=0.90, axis=1) - lin_mean_list, 0, 100)

train_amounts = np.array(len_results['train_amounts'])

fig, ax = plt.subplots(figsize=(3.3,1.9))

ax.errorbar(train_amounts, lin_mean_list, yerr=(lin_low_error, lin_high_error), fmt='^', linewidth=2, markersize=8, label='DMDc', alpha=0.8, mec='black' ,mew=0.5)
ax.errorbar(train_amounts, fc_mean_list, yerr=(fc_low_error, fc_high_error), fmt='s', linewidth=2, markersize=8, label='FCN', alpha=0.8,mec='black' ,mew=0.5)
ax.errorbar(train_amounts, gru_mean_list, yerr=(gru_low_error, gru_high_error), fmt='v', linewidth=2, markersize=8, label='GRU', alpha=0.8,mec='black' ,mew=0.5)
ax.errorbar(train_amounts, lstm_mean_list, yerr=(lstm_low_error, lstm_high_error), fmt='P', linewidth=2, markersize=8, label='LSTM', alpha=0.8,mec='black' ,mew=0.5)
ax.errorbar(train_amounts, esn_mean_list, yerr=(esn_low_error, esn_high_error), fmt='*', linewidth=2, markersize=8, label='ESN', alpha=0.8,mec='black' ,mew=0.5)
# ax.legend(fontsize=24, loc='upper right')

# ax.set_xlim([300, max(train_amounts * 2)])
ax.tick_params(axis="both", which="major", labelsize=12)
ax.tick_params(axis="both", which="minor", labelsize=12)
ax.set_ylabel('Error', fontsize=12)
ax.set_xlabel('Training Samples', fontsize=12)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xticks([500, 2000, 8000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.tight_layout()
fig.savefig(dest + '/train_len_plot.pdf')