import json
from datetime import datetime
import argparse
from RNNmpc.utils import forecast_eval
import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dict",
    type=str,
    help="Location of data dictionary for training and eval.",
    required=True,
)

parser.add_argument(
    "--noise_level",
    type=float,
    help="Noise to add to data",
    required=False,
    default=0.0,
)

parser.add_argument(
    "--dest",
    type=str,
    help="Destination of output dictionary.",
    required=True,
)

args = parser.parse_args()
data_dict = json.load(open(args.data_dict))
noise_level = args.noise_level
dest = args.dest


### Run throught hyperparam combinations
hyperparam_dict = {}
alpha_list = [0.2, 0.4, 0.6, 0.8]
beta_list = [1e-4, 1e-5, 1e-6, 1e-7]
sigma_list = [0.005, 0.01, 0.1, 0.25, 0.5]
sigma_b_list = [0.33, 0.66, 1, 1.33, 1.66]
rho_sr_list = [0.2, 0.4, 0.6, 0.8]
counter = 0
esn_tot_count = (
    len(rho_sr_list)
    * len(sigma_b_list)
    * len(beta_list)
    * len(alpha_list)
    * len(sigma_list)
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Starting ESN tuning: ")
for alpha in alpha_list:
    for beta in beta_list:
        for sigma in sigma_list:
            for sigma_b in sigma_b_list:
                for rho_sr in rho_sr_list:
                    ret_dict, _ = forecast_eval(
                        "ESNForecaster",
                        data_dict=data_dict,
                        alpha=alpha,
                        sigma=sigma,
                        sigma_b=sigma_b,
                        rho_sr=rho_sr,
                        beta=beta,
                        scaled=True,
                        noise_level=noise_level,
                        device=device,
                    )
                    current_datetime = datetime.now().strftime("%F-%T.%f")[:-3]
                    entry_name = "ESN" + ret_dict["simulator"] + current_datetime
                    hyperparam_dict[entry_name] = ret_dict
                    counter += 1
                    if counter % 50 == 0:
                        print(
                            str(counter) + "/" + str(esn_tot_count) + " ESN complete."
                        )

print("ESN tuning complete, starting GRU tuning.")

adam_lr_list = [1e-2, 1e-3, 1e-4]
hidden_dim_list = [16, 32, 64, 128]
lags_list = [5, 10, 20, 30]
dropout_p_list = [0.0, 0.01, 0.02, 0.05, 0.1]
lstm_tot_count = (
    len(adam_lr_list) * len(hidden_dim_list) * len(lags_list) * len(dropout_p_list)
)
counter = 0
for adam_lr in adam_lr_list:
    for hidden_dim in hidden_dim_list:
        for lags in lags_list:
            for dropout_p in dropout_p_list:
                ret_dict, _ = forecast_eval(
                    "GRUForecaster",
                    data_dict=data_dict,
                    scaled=True,
                    adam_lr=adam_lr,
                    hidden_dim=hidden_dim,
                    lags=lags,
                    dropout_p=dropout_p,
                    noise_level=noise_level,
                    device=device,
                )
                current_datetime = datetime.now().strftime("%F-%T.%f")[:-3]
                entry_name = "GRU" + ret_dict["simulator"] + current_datetime
                hyperparam_dict[entry_name] = ret_dict
                counter += 1
                if counter % 50 == 0:
                    print(str(counter) + "/" + str(lstm_tot_count) + " GRU complete.")

print("GRU tuning complete, starting LSTM tuning.")
counter = 0
for adam_lr in adam_lr_list:
    for hidden_dim in hidden_dim_list:
        for lags in lags_list:
            for dropout_p in dropout_p_list:
                ret_dict, _ = forecast_eval(
                    "LSTMForecaster",
                    data_dict=data_dict,
                    scaled=True,
                    adam_lr=adam_lr,
                    hidden_dim=hidden_dim,
                    lags=lags,
                    dropout_p=dropout_p,
                    noise_level=noise_level,
                    device=device,
                )
                current_datetime = datetime.now().strftime("%F-%T.%f")[:-3]
                entry_name = "LSTM" + ret_dict["simulator"] + current_datetime
                hyperparam_dict[entry_name] = ret_dict
                counter += 1
                if counter % 50 == 0:
                    print(str(counter) + "/" + str(lstm_tot_count) + " LSTM complete.")


r_width_list = [10, 25, 50, 75, 100]
tds_list = [1, 5, 10, 15, 20]
adam_lr_list = [1e-2, 1e-3, 1e-4]
dropout_p_list = [0.0, 0.01, 0.02, 0.05, 0.1]
fc_tot_count = (
    len(r_width_list) * len(lags_list) * len(dropout_p_list) * len(adam_lr_list)
)
counter = 0
print("LSTM tuning complete, starting FC tuning.")
for r_width in r_width_list:
    for tds in tds_list:
        for adam_lr in adam_lr_list:
            for dropout_p in dropout_p_list:

                ret_dict, _ = forecast_eval(
                    "FCForecaster",
                    data_dict=data_dict,
                    scaled=True,
                    r_width=r_width,
                    tds=[-i for i in range(1, tds + 1)],
                    adam_lr=adam_lr,
                    dropout_p=dropout_p,
                    noise_level=noise_level,
                    device=device,
                )
                current_datetime = datetime.now().strftime("%F-%T.%f")[:-3]
                entry_name = "FC" + ret_dict["simulator"] + current_datetime
                hyperparam_dict[entry_name] = ret_dict
                counter += 1
                if counter % 50 == 0:
                    print(str(counter) + "/" + str(fc_tot_count) + " FC complete.")

beta_list = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
tds_list = [5, 10, 15, 20]
lin_tot_count = len(beta_list) * len(lags_list)
counter = 0
print("FC tuning complete, starting Linear tuning.")
for beta in beta_list:
    for tds in tds_list:

        ret_dict, _ = forecast_eval(
            "LinearForecaster",
            data_dict=data_dict,
            beta=beta,
            scaled=True,
            tds=[-i for i in range(1, tds + 1)],
            noise_level=noise_level,
            device=device,
        )
        current_datetime = datetime.now().strftime("%F-%T.%f")[:-3]
        entry_name = "Linear" + ret_dict["simulator"] + current_datetime
        hyperparam_dict[entry_name] = ret_dict
        counter += 1
        if counter % 50 == 0:
            print(str(counter) + "/" + str(lin_tot_count) + " Linear complete.")

with open(dest + "/hyperparam_results.json", "w") as fp:
    json.dump(hyperparam_dict, fp)


### Save best performing models
def find_best_model(model_type, param_dict):
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
    return best_model


esn_out = find_best_model("ESNForecaster", hyperparam_dict)
gru_out = find_best_model("GRUForecaster", hyperparam_dict)
lstm_out = find_best_model("LSTMForecaster", hyperparam_dict)
lin_out = find_best_model("LinearForecaster", hyperparam_dict)
fc_out = find_best_model("FCForecaster", hyperparam_dict)

ret_dict = {
    "ESNForecaster": esn_out,
    "GRUForecaster": gru_out,
    "LSTMForecaster": lstm_out,
    "LinearForecaster": lin_out,
    "FCForecaster": fc_out,
}
with open(dest + "/best_hyperparams.json", "w") as fp:
    json.dump(ret_dict, fp)
