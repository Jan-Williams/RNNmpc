import RNNmpc.Forecasters as Forecaster
import json
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dict",
    type=str,
    help="Location of data dictionary for training and eval.",
    required=True,
)

parser.add_argument(
    "--dest",
    type=str,
    help="Destination of output dictionary.",
    required = True,
)

args = parser.parse_args()
data_dict = args.data_dict
dest = args.dest

def forecast_eval(
        model_type: str,
        data_dict: str,
        scaled: bool,
        alpha: float = 0.6,
        sigma: float = 0.084,
        sigma_b: float =1.6,
        rho_sr: float=0.8,
        beta: float=1e-8,
        adam_lr: float =0.001,
        dropout_p: float = 0.0,
        lags: int = 10,
        r_width: int = 50,
        hidden_dim: int = 32,
        train_steps: int = 5000,         
    ):

    f = open(data_dict)
    data_dict = json.load(f)

    U_train = np.array(data_dict['U_train'])[:,-train_steps:]
    S_train = np.array(data_dict['S_train'])[:,-train_steps:]
    O_train = np.array(data_dict['O_train'])[:,-train_steps:]

    U_valid = np.array(data_dict['U_valid'])
    S_valid = np.array(data_dict['S_valid'])
    O_valid = np.array(data_dict['O_valid'])

    if scaled:
        sensor_scaler = MinMaxScaler()
        S_train = sensor_scaler.fit_transform(S_train.T).T
        S_valid = sensor_scaler.transform(S_valid.T).T
        O_train = sensor_scaler.transform(O_train.T).T
        O_valid = sensor_scaler.transform(O_valid.T).T

        control_scaler = MinMaxScaler()
        U_train = control_scaler.fit_transform(U_train.T).T
        U_valid = control_scaler.transform(U_valid.T).T

    U_train = torch.tensor(U_train, dtype=torch.float64)
    S_train = torch.tensor(S_train, dtype=torch.float64)
    O_train = torch.tensor(O_train, dtype=torch.float64)

    U_valid = torch.tensor(U_valid, dtype=torch.float64)
    S_valid = torch.tensor(S_valid, dtype=torch.float64)
    O_valid = torch.tensor(O_valid, dtype=torch.float64)

    concat_U = torch.hstack((U_train, U_valid))
    concat_S = torch.hstack((S_train, S_valid))
    concat_O = torch.hstack((S_train, S_valid))

    No = O_valid.shape[0]
    Ns = S_valid.shape[0]
    Nu = U_valid.shape[0]

    fcast_steps = int(data_dict['fcast_lens'] / data_dict['control_disc'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_type == "ESNForecaster":
        ret_dict = {
            "rho_sr":rho_sr,
            "sigma": sigma,
            "sigma_b": sigma_b,
            "alpha": alpha,
            "beta": beta
        }
        Nr = 1000
        model = Forecaster.ESNForecaster(
            Nr=Nr,
            No=No,
            Nu=Nu,
            Ns=Ns,
            rho_sr=rho_sr,
            sigma=sigma,
            sigma_b=sigma_b,
            alpha=alpha
        )
        esn_r = model.fit(U_train, S_train, O_train, beta=beta)
        model.set_device(device)
        tot_forecast = torch.zeros((No, 0)).to("cpu")
        for i in range(10):
            fcast_start_index = U_train.shape[1] + i * fcast_steps
            spin_r = model.spin(
                U_spin=concat_U[:, fcast_start_index-500:fcast_start_index], 
                S_spin=concat_S[:, fcast_start_index-500:fcast_start_index],
            )
            fcast = model.forecast(
                concat_U[:, fcast_start_index:fcast_start_index + fcast_steps],
                r_k=spin_r,
                s_k=concat_S[:, fcast_start_index:fcast_start_index+1])
            tot_forecast = torch.hstack((tot_forecast, fcast.to("cpu")))

    else:
        
        if model_type == "GRUForecaster":
            ret_dict = {
                "Nr": hidden_dim, 
                "dropout_p": dropout_p,
                "lags": lags,
                "adam_lr": adam_lr
            }
            Nr = hidden_dim
            model = Forecaster.GRUForecaster(Nr=Nr, Nu=Nu, Ns=Ns, No=No, dropout_p=dropout_p)
            model.set_device(device)
            out_r = model.fit(U_train, S_train, O_train, lags=lags, lr=adam_lr)

        elif model_type == "LSTMForecaster":
            ret_dict = {
                "Nr": hidden_dim, 
                "dropout_p": dropout_p,
                "lags": lags,
                "adam_lr": adam_lr
            }
            Nr = hidden_dim
            model = Forecaster.LSTMForecaster(Nr=Nr, Nu=Nu, Ns=Ns, No=No, dropout_p=dropout_p)
            model.set_device(device)
            out_r = model.fit(U_train, S_train, O_train, lags=lags, lr=adam_lr)

        elif model_type == "LinearForecaster":
            tds = [-i for i in range(1,lags+1)]
            ret_dict = {
                "beta": beta,
                "tds": tds,
            }
            tds = [-i for i in range(1,lags+1)]
            model = Forecaster.LinearForecaster(Nu=Nu, Ns=Ns, No=No, tds=tds)
            model.set_device(device)
            model.fit(U_train, S_train, O_train, beta=beta)
        else:
            tds = [-i for i in range(1,lags+1)]
            ret_dict = {
                "tds":tds,
                "dropout_p": dropout_p,
                "r_width": r_width,
                "adam_lr": adam_lr
            }
            model = Forecaster.FCForecaster(Nu=Nu, Ns=Ns, No=No, tds=tds, r_list=[r_width]*2, dropout_p=dropout_p)
            model.set_device(device)
            model.fit(U_train, S_train, O_train, lr=adam_lr)

        tot_forecast = torch.zeros((No, 0)).to("cpu")

        for i in range(10):
            fcast_start_index = U_train.shape[1] + i * fcast_steps
            fcast = model.forecast(
                U=concat_U[:, fcast_start_index:fcast_start_index + fcast_steps],
                U_spin=concat_U[:, fcast_start_index - lags + 1:fcast_start_index],
                S_spin=concat_S[:, fcast_start_index - lags + 1:fcast_start_index],
                s_k=concat_S[:, fcast_start_index:fcast_start_index + 1]
            )
            tot_forecast = torch.hstack((tot_forecast, fcast.to("cpu")))

    tot_forecast = tot_forecast.detach().cpu().numpy()
    O_valid = O_valid.detach().cpu().numpy()

    if scaled:
        tot_forecast = sensor_scaler.inverse_transform(tot_forecast.T).T
        O_valid = sensor_scaler.inverse_transform(O_valid.T).T

    fcast_dev = np.linalg.norm((tot_forecast - O_valid) / O_valid) / O_valid.shape[1]

    ret_dict["model_type"] = model_type
    ret_dict["simulator"] = data_dict["simulator"]
    ret_dict["control_disc"] = data_dict["control_disc"]
    ret_dict["train_len"] = train_steps
    ret_dict["model_disc"] = data_dict["model_disc"]
    ret_dict["fcast_dev"] = fcast_dev
    ret_dict["scaled"] = scaled

    return ret_dict

hyperparam_dict = {}
alpha_list = [0.2, 0.4, 0.6, 0.8]
beta_list = [1e-4, 1e-5, 1e-6, 1e-7]
sigma_list = [0.005, 0.01, 0.1, 0.25, 0.5]
sigma_b_list = [0.33, 0.66, 1, 1.33, 1.66]
rho_sr_list = [0.2,0.4, 0.6, 0.8]
counter = 0
esn_tot_count = len(rho_sr_list) * len(sigma_b_list) * len(beta_list) * len(alpha_list) * len(sigma_list)
print("Starting ESN tuning: ")
for alpha in alpha_list:
    for beta in beta_list:
        for sigma in sigma_list:
            for sigma_b in sigma_b_list:
                for rho_sr in rho_sr_list:
                    ret_dict = forecast_eval('ESNForecaster',
                                                data_dict=data_dict,
                                                alpha=alpha,
                                                sigma=sigma,
                                                sigma_b=sigma_b,
                                                rho_sr=rho_sr,
                                                beta=beta,
                                                scaled=False)
                    current_datetime = datetime.now().strftime('%F-%T.%f')[:-3]
                    entry_name = 'ESN' + ret_dict['simulator'] + current_datetime
                    hyperparam_dict[entry_name] = ret_dict
                    counter += 1
                    if counter % 50 == 0:
                        print(str(counter) + '/' + str(esn_tot_count) + ' ESN complete.')

print("ESN tuning complete, starting GRU tuning.")

adam_lr_list = [1e-2, 1e-3, 1e-4]
hidden_dim_list = [16, 32, 64, 128]
lags_list = [5, 10, 20, 30]
dropout_p_list = [0.0, 0.01, 0.02, 0.05, 0.1]
lstm_tot_count = len(adam_lr_list) * len(hidden_dim_list) * len(lags_list) * len(dropout_p_list)
counter = 0
for adam_lr in adam_lr_list:
    for hidden_dim in hidden_dim_list:
        for lags in lags_list:
            for dropout_p in dropout_p_list:
                ret_dict = forecast_eval('GRUForecaster',
                                                data_dict=data_dict,
                                                scaled=True,
                                                adam_lr=adam_lr,
                                                hidden_dim=hidden_dim,
                                                lags=lags,
                                                dropout_p=dropout_p)
                current_datetime = datetime.now().strftime('%F-%T.%f')[:-3]
                entry_name = 'GRU' + ret_dict['simulator'] + current_datetime
                hyperparam_dict[entry_name] = ret_dict
                counter += 1
                if counter % 50 == 0:
                    print(str(counter) + '/' + str(lstm_tot_count) + ' GRU complete.')

print("GRU tuning complete, starting LSTM tuning.")
counter = 0
for adam_lr in adam_lr_list:
    for hidden_dim in hidden_dim_list:
        for lags in lags_list:
            for dropout_p in dropout_p_list:
                ret_dict = forecast_eval('LSTMForecaster',
                                                data_dict=data_dict,
                                                scaled=True,
                                                adam_lr=adam_lr,
                                                hidden_dim=hidden_dim,
                                                lags=lags,
                                                dropout_p=dropout_p)
                current_datetime = datetime.now().strftime('%F-%T.%f')[:-3]
                entry_name = 'LSTM' + ret_dict['simulator'] + current_datetime
                hyperparam_dict[entry_name] = ret_dict
                counter += 1
                if counter % 50 == 0:
                    print(str(counter) + '/' + str(lstm_tot_count) + ' LSTM complete.')


r_width_list = [10, 25, 50, 75, 100]
lags_list = [1, 5, 10, 15, 20]
adam_lr_list = [1e-2, 1e-3, 1e-4]
dropout_p_list = [0.0, 0.01, 0.02, 0.05, 0.1]
fc_tot_count = len(r_width_list) * len(lags_list) * len(dropout_p_list) * len(adam_lr_list)
counter = 0
print("LSTM tuning complete, starting FC tuning.")
for r_width in r_width_list:
    for lags in lags_list:
        for adam_lr in adam_lr_list:
            for dropout_p in dropout_p_list:

                ret_dict = forecast_eval('FCForecaster',
                                                data_dict=data_dict,
                                                scaled=True,
                                                r_width=r_width,
                                                lags=lags,
                                                adam_lr=adam_lr,
                                                dropout_p=dropout_p)
                current_datetime = datetime.now().strftime('%F-%T.%f')[:-3]
                entry_name = 'FC' + ret_dict['simulator'] + current_datetime
                hyperparam_dict[entry_name] = ret_dict
                counter += 1
                if counter % 50 == 0:
                    print(str(counter) + '/' + str(fc_tot_count) + ' FC complete.')

beta_list = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
lags_list = [5, 10, 15, 20]
lin_tot_count = len(beta_list) * len(lags_list)
counter = 0
print("FC tuning complete, starting Linear tuning.")
for beta in beta_list:
    for lags in lags_list:

        ret_dict = forecast_eval('LinearForecaster',
                                                data_dict=data_dict,
                                                scaled=False,
                                                lags=lags)
        current_datetime = datetime.now().strftime('%F-%T.%f')[:-3]
        entry_name = 'Linear' + ret_dict['simulator'] + current_datetime
        hyperparam_dict[entry_name] = ret_dict
        counter += 1
        if counter % 50 == 0:
            print(str(counter) + '/' + str(lin_tot_count) + ' Linear complete.')
        
with open(dest + '/hyperparam_results.json', 'w') as fp:
    json.dump(hyperparam_dict, fp)