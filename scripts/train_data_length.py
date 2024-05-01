import RNNmpc
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dict",
    type=str,
    help="Data dictionary.",
    required=True
)

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
    "--train_amounts",
    type=list[int],
    help="Amounts of training data to consider to consider.",
    required=False,
    default=[500, 1000, 2000, 4000, 8000, 16000, 32000]
)



args = parser.parse_args()
best_params_dict = json.load(open(args.best_hyperparams_dict))

train_amounts = args.train_amounts

### ESN
alpha = best_params_dict["ESNForecaster"]["alpha"]
sigma = best_params_dict["ESNForecaster"]["sigma"]
sigma_b = best_params_dict["ESNForecaster"]["sigma_b"]
beta = best_params_dict["ESNForecaster"]["beta"]
rho_sr = best_params_dict["ESNForecaster"]["rho_sr"]

esn_list = []
esn_dict = {}
for train_amount in train_amounts:
    ret_dict, _ = RNNmpc.utils.forecast_eval("ESNForecaster", 
                                          data_dict=args.data_dict,
                                          scaled=False, 
                                          alpha=alpha,
                                          rho_sr=rho_sr,
                                          beta=beta,
                                          sigma_b=sigma_b,
                                          sigma=sigma,
                                          train_steps=train_amount,
                                          )
    
    esn_list.append(ret_dict["fcast_dev"])


adam_lr = best_params_dict["LSTMForecaster"]["adam_lr"]
dropout_p = best_params_dict["LSTMForecaster"]["dropout_p"]
lags = best_params_dict["LSTMForecaster"]["lags"]
Nr = best_params_dict["LSTMForecaster"]["Nr"]
lstm_list = []
for train_amount in train_amounts:
    ret_dict, _ = RNNmpc.utils.forecast_eval("LSTMForecaster", 
                                          data_dict=args.data_dict,
                                          scaled=True, 
                                          adam_lr=adam_lr,
                                          lags=lags,
                                          hidden_dim=Nr,
                                          dropout_p=dropout_p,
                                          train_steps=train_amount,
                                          )
    
    lstm_list.append(ret_dict["fcast_dev"])

adam_lr = best_params_dict["GRUForecaster"]["adam_lr"]
dropout_p = best_params_dict["GRUForecaster"]["dropout_p"]
lags = best_params_dict["GRUForecaster"]["lags"]
Nr = best_params_dict["GRUForecaster"]["Nr"]
gru_list = []
for train_amount in train_amounts:
    ret_dict, _ = RNNmpc.utils.forecast_eval("GRUForecaster", 
                                          data_dict=args.data_dict,
                                          scaled=True, 
                                          adam_lr=adam_lr,
                                          lags=lags,
                                          hidden_dim=Nr,
                                          dropout_p=dropout_p,
                                          train_steps=train_amount,
                                          )
    
    gru_list.append(ret_dict["fcast_dev"])



beta = best_params_dict["LinearForecaster"]["beta"]
tds = best_params_dict["LinearForecaster"]["tds"]
lin_list = []
for train_amount in train_amounts:
    ret_dict, _ = RNNmpc.utils.forecast_eval("LinearForecaster", 
                                          data_dict=args.data_dict,
                                          scaled=False, 
                                          beta=beta,
                                          lags=-min(tds),
                                          train_steps=train_amount,
                                          )
    
    lin_list.append(ret_dict["fcast_dev"])

adam_lr = best_params_dict["FCForecaster"]["adam_lr"]
tds = best_params_dict["FCForecaster"]["tds"]
r_width = best_params_dict["FCForecaster"]["r_width"]
dropout_p = best_params_dict["FCForecaster"]["dropout_p"]

fc_list = []
for train_amount in train_amounts:
    ret_dict, _ = RNNmpc.utils.forecast_eval("FCForecaster", 
                                          data_dict=args.data_dict,
                                          scaled=False, 
                                          r_width=r_width,
                                          lags=-min(tds),
                                          dropout_p=dropout_p,
                                          train_steps=train_amount,
                                          )
    
    fc_list.append(ret_dict["fcast_dev"])

fig, ax = plt.subplots()
ax.semilogy(train_amounts, lin_list, label="DMDc", linewidth=2)
ax.semilogy(train_amounts, fc_list, label="FCN", linewidth=2)
ax.semilogy(train_amounts, gru_list, label='GRU', linewidth=2)
ax.semilogy(train_amounts, lstm_list, label='LSTM', linewidth=2)
ax.semilogy(train_amounts, esn_list, label='ESN', linewidth=2)
ax.tick_params(axis="both", which="major", labelsize=14)
ax.tick_params(axis="both", which="minor", labelsize=14)
ax.set_xlabel('Training Samples', fontsize=18)
ax.set_ylabel('Error', fontsize=18)
ax.set_xlim([0, max(train_amounts) * 1.35])
ax.legend(fontsize=15, loc = 'center right')
fig.savefig(args.dest + '/data_length.pdf')