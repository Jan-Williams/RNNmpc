import RNNmpc
import matplotlib.pyplot as plt
import argparse
import json
import torch

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
    "--num_models",
    type=int,
    default=32,
    help="Number of models to train for each model type.",
    required=False
)


args = parser.parse_args()
best_params_dict = json.load(open(args.best_hyperparams_dict))
data_dict = json.load(open(args.data_dict))
num_models = args.num_models

train_amounts = [500, 1000, 2000, 4000, 8000, 16000]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### ESN
alpha = best_params_dict["ESNForecaster"]["alpha"]
sigma = best_params_dict["ESNForecaster"]["sigma"]
sigma_b = best_params_dict["ESNForecaster"]["sigma_b"]
beta = best_params_dict["ESNForecaster"]["beta"]
rho_sr = best_params_dict["ESNForecaster"]["rho_sr"]

esn_list = []
esn_dict = {}
counter = 0
model_count = len(train_amounts) * num_models
print("Starting ESN evaluation.")
for train_amount in train_amounts:
    temp_list = []
    for i in range(num_models):
        ret_dict, _ = RNNmpc.utils.forecast_eval(
            "ESNForecaster",
            data_dict=data_dict,
            scaled=True,
            alpha=alpha,
            rho_sr=rho_sr,
            beta=beta,
            sigma_b=sigma_b,
            sigma=sigma,
            train_steps=train_amount,
            device=device,
        )
        counter += 1
        if counter % 50 == 0:
            print(str(counter) + "/" + str(num_models) + " ESN complete.")
        temp_list.append(ret_dict["fcast_dev"])
    esn_list.append(temp_list)


adam_lr = best_params_dict["LSTMForecaster"]["adam_lr"]
dropout_p = best_params_dict["LSTMForecaster"]["dropout_p"]
lags = best_params_dict["LSTMForecaster"]["lags"]
Nr = best_params_dict["LSTMForecaster"]["Nr"]
lstm_list = []
counter = 0
print("Starting LSTM evaluation.")
for train_amount in train_amounts:
    temp_list = []
    for i in range(num_models):
        ret_dict, _ = RNNmpc.utils.forecast_eval(
            "LSTMForecaster",
            data_dict=data_dict,
            scaled=True,
            adam_lr=adam_lr,
            lags=lags,
            hidden_dim=Nr,
            dropout_p=dropout_p,
            train_steps=train_amount,
            device=device,
        )
        counter += 1
        if counter % 50 == 0:
            print(str(counter) + "/" + str(num_models) + " LSTM complete.")
        temp_list.append(ret_dict["fcast_dev"])
    lstm_list.append(temp_list)

adam_lr = best_params_dict["GRUForecaster"]["adam_lr"]
dropout_p = best_params_dict["GRUForecaster"]["dropout_p"]
lags = best_params_dict["GRUForecaster"]["lags"]
Nr = best_params_dict["GRUForecaster"]["Nr"]
gru_list = []
counter = 0
print("Starting GRU evaluation.")
for train_amount in train_amounts:
    temp_list = []
    for i in range(num_models):
        ret_dict, _ = RNNmpc.utils.forecast_eval(
            "GRUForecaster",
            data_dict=data_dict,
            scaled=True,
            adam_lr=adam_lr,
            lags=lags,
            hidden_dim=Nr,
            dropout_p=dropout_p,
            train_steps=train_amount,
            device=device,
        )
        counter += 1
        if counter % 50 == 0:
            print(str(counter) + "/" + str(num_models) + " GRU complete.")
        temp_list.append(ret_dict["fcast_dev"])
    gru_list.append(temp_list)

beta = best_params_dict["LinearForecaster"]["beta"]
tds = best_params_dict["LinearForecaster"]["tds"]
lin_list = []
counter = 0
print("Starting FC evaluation.")
for train_amount in train_amounts:
    temp_list = []
    for i in range(num_models):
        ret_dict, _ = RNNmpc.utils.forecast_eval(
            "LinearForecaster",
            data_dict=data_dict,
            scaled=True,
            beta=beta,
            tds=tds,
            train_steps=train_amount,
            device=device,
        )
        counter += 1
        if counter % 50 == 0:
            print(str(counter) + "/" + str(num_models) + " Linear complete.")
        temp_list.append(ret_dict["fcast_dev"])
    lin_list.append(temp_list)

adam_lr = best_params_dict["FCForecaster"]["adam_lr"]
tds = best_params_dict["FCForecaster"]["tds"]
r_width = best_params_dict["FCForecaster"]["r_width"]
dropout_p = best_params_dict["FCForecaster"]["dropout_p"]

fc_list = []
counter = 0
print("Starting FC evaluation.")
for train_amount in train_amounts:
    temp_list = []
    for i in range(num_models):
        ret_dict, _ = RNNmpc.utils.forecast_eval(
            "FCForecaster",
            data_dict=data_dict,
            scaled=True,
            r_width=r_width,
            tds=tds,
            dropout_p=dropout_p,
            train_steps=train_amount,
            device=device,
        )
        counter += 1
        if counter % 50 == 0:
            print(str(counter) + "/" + str(num_models) + " FC complete.")
        temp_list.append(ret_dict["fcast_dev"])
    fc_list.append(temp_list)
    

results_dict = {
    "esn_list": esn_list,
    "lstm_list": lstm_list,
    "gru_list": gru_list,
    "fc_list": fc_list,
    "lin_list": lin_list,
    "train_amounts": train_amounts
}

with open(args.dest + "/train_len_results.json", "w") as fp:
    json.dump(results_dict, fp)