import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--hyperparam_dict", type=str, help="Results of hyperparam_searc.py", required=True
)

parser.add_argument(
    "--dest",
    type=str,
    help="Destination to save plots and best hyperparam values.",
    required=True,
)

args = parser.parse_args()
hyperparam_dict = args.hyperparam_dict
dest = args.dest

param_dict = json.load(open(hyperparam_dict))


def plot_perf_distribution(model_type, param_dict):
    best_key = 0
    best_perf = 100
    perf_list = []
    for key in param_dict.keys():
        if param_dict[key]["model_type"] == model_type:
            perf = param_dict[key]["fcast_dev"]
            perf_list.append(perf)
            if perf < best_perf:
                best_perf = perf
                best_key = key
    best_model = param_dict[best_key]
    perf_list = np.clip(perf_list, 0, 1)
    perf_list = np.array(perf_list)
    bin_edges = np.histogram_bin_edges(perf_list, "auto")
    fig, ax = plt.subplots()
    ax.hist(perf_list, bin_edges, density=True)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=14)
    ax.set_ylabel("Density", fontsize=20)
    ax.set_xlabel("Forecast Deviation", fontsize=20)
    ax.set_title(model_type, fontsize=20)
    fig.savefig(dest + "/" + model_type + "hyperparam_dist.pdf")
    return best_model


esn_out = plot_perf_distribution("ESNForecaster", param_dict)
gru_out = plot_perf_distribution("GRUForecaster", param_dict)
lstm_out = plot_perf_distribution("LSTMForecaster", param_dict)
lin_out = plot_perf_distribution("LinearForecaster", param_dict)
fc_out = plot_perf_distribution("FCForecaster", param_dict)

ret_dict = {
    "ESNForecaster": esn_out,
    "GRUForecaster": gru_out,
    "LSTMForecaster": lstm_out,
    "LinearForecaster": lin_out,
    "FCForecaster": fc_out,
}
with open(dest + "/best_hyperparams.json", "w") as fp:
    json.dump(ret_dict, fp)
