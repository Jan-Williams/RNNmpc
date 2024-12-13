import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from RNNmpc import MPController, Forecasters, Simulators

### Note: the function forecast_eval is useful in replicating the results from
### the paper but is likely not as useful for other applications


def forecast_eval(
    model_type: str,
    data_dict: dict,
    scaled: bool,
    alpha: float = 0.6,
    sigma: float = 0.084,
    sigma_b: float = 1.6,
    rho_sr: float = 0.8,
    beta: float = 1e-8,
    adam_lr: float = 0.001,
    dropout_p: float = 0.0,
    lags: int = 10,
    tds: list = [-i for i in range(1, 11)],
    r_width: int = 50,
    hidden_dim: int = 32,
    train_steps: int = 5000,
    noise_level=0.0,
    device=torch.device("cpu"),
):
    """
    Evaluation of RNNmpc.Forecaster model on data output from generate_data.

    Parameters:
    -----------
    model_type: str
        class from RNNmpc.Forecasters
    data_dict: dict
        dictionary containing output from generate data
    scaled: bool
        whether to MinMax scale data for training model
    alpha: float
        alpha parameter for ESN
    sigma: float
        sigma parameter for ESN
    sigma_b: float
        sigma_b parameter for ESN
    beta: float
        beta for either ESN or Linear, depending on model_type
    adam_lr: float
        learning rate for Adam algorithm, if used
    dropout_p: float
        dropout rate during training, if used
    tds: list[int]
        tds for the Linear or FC foercasters
    lags: int
        lags for the RNN models
    r_width: int
        number of nodes in hidden layers for FC models
    hidden_dim: int
        hidden dimension for GRU or LSTM models
    train_steps: int
        number of training steps to use

    Returns:
    ----------
    ret_dict: dict
        dictionary containing fcast_dev and model parameters used
    tot_forecast: np.array
        computed forecast throughout U_valid of data_dict
    """

    U_train = np.array(data_dict["U_train"])[:, -train_steps:]
    for i in range(U_train.shape[0]):
        noise_mag = noise_level * np.std(U_train[i])
        noise_sig = np.random.normal(0, noise_mag, size=U_train[i].shape)
        U_train += noise_sig

    S_train = np.array(data_dict["S_train"])[:, -train_steps:]
    for i in range(S_train.shape[0]):
        noise_mag = noise_level * np.std(S_train[i])
        noise_sig = np.random.normal(0, noise_mag, size=S_train[i].shape)
        S_train += noise_sig
    O_train = np.array(data_dict["O_train"])[:, -train_steps:]
    for i in range(O_train.shape[0]):
        noise_mag = noise_level * np.std(O_train[i])
        noise_sig = np.random.normal(0, noise_mag, size=O_train[i].shape)
        O_train += noise_sig

    U_valid = np.array(data_dict["U_valid"])
    for i in range(U_valid.shape[0]):
        noise_mag = noise_level * np.std(U_valid[i])
        noise_sig = np.random.normal(0, noise_mag, size=U_valid[i].shape)
        U_valid += noise_sig
    S_valid = np.array(data_dict["S_valid"])
    for i in range(S_valid.shape[0]):
        noise_mag = noise_level * np.std(S_valid[i])
        noise_sig = np.random.normal(0, noise_mag, size=S_valid[i].shape)
        S_valid += noise_sig
    O_valid = np.array(data_dict["O_valid"])

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
    concat_O = torch.hstack((O_train, O_valid))

    No = O_valid.shape[0]
    Ns = S_valid.shape[0]
    Nu = U_valid.shape[0]

    fcast_steps = int(data_dict["fcast_lens"] / data_dict["control_disc"])

    if model_type == "ESNForecaster":
        ret_dict = {
            "rho_sr": rho_sr,
            "sigma": sigma,
            "sigma_b": sigma_b,
            "alpha": alpha,
            "beta": beta,
        }
        Nr = 1000
        model = Forecasters.ESNForecaster(
            Nr=Nr,
            No=No,
            Nu=Nu,
            Ns=Ns,
            rho_sr=rho_sr,
            sigma=sigma,
            sigma_b=sigma_b,
            alpha=alpha,
        )
        model.set_device(device)
        esn_r = model.fit(U_train, S_train, O_train, beta=beta, spinup=200)
        tot_forecast = torch.zeros((No, 0)).to("cpu")
        for i in range(10):
            fcast_start_index = U_train.shape[1] + i * fcast_steps
            spin_r = model.spin(
                U_spin=concat_U[:, fcast_start_index - 200 : fcast_start_index],
                S_spin=concat_S[:, fcast_start_index - 200 : fcast_start_index],
            )
            fcast = model.forecast(
                concat_U[:, fcast_start_index : fcast_start_index + fcast_steps],
                r_k=spin_r,
                s_k=concat_S[:, fcast_start_index : fcast_start_index + 1],
            )
            tot_forecast = torch.hstack((tot_forecast, fcast.to("cpu")))

    else:

        if model_type == "GRUForecaster":
            ret_dict = {
                "Nr": hidden_dim,
                "dropout_p": dropout_p,
                "lags": lags,
                "adam_lr": adam_lr,
            }
            Nr = hidden_dim
            model = Forecasters.GRUForecaster(
                Nr=Nr, Nu=Nu, Ns=Ns, No=No, dropout_p=dropout_p
            )
            model.set_device(device)
            out_r = model.fit(
                U_train, S_train, O_train, lags=lags, lr=adam_lr, num_epochs=5000
            )

        elif model_type == "LSTMForecaster":
            ret_dict = {
                "Nr": hidden_dim,
                "dropout_p": dropout_p,
                "lags": lags,
                "adam_lr": adam_lr,
            }
            Nr = hidden_dim
            model = Forecasters.LSTMForecaster(
                Nr=Nr, Nu=Nu, Ns=Ns, No=No, dropout_p=dropout_p
            )
            model.set_device(device)
            out_r = model.fit(
                U_train, S_train, O_train, lags=lags, lr=adam_lr, num_epochs=5000
            )

        elif model_type == "LinearForecaster":
            ret_dict = {
                "beta": beta,
                "tds": tds,
            }
            lags = max(-np.array(tds))
            model = Forecasters.LinearForecaster(Nu=Nu, Ns=Ns, No=No, tds=tds)
            model.set_device(device)
            model.fit(
                U_train,
                S_train,
                O_train,
                beta=beta,
            )
        else:
            ret_dict = {
                "tds": tds,
                "dropout_p": dropout_p,
                "r_width": r_width,
                "adam_lr": adam_lr,
            }
            lags = max(-np.array(tds))
            model = Forecasters.FCForecaster(
                Nu=Nu, Ns=Ns, No=No, tds=tds, r_list=[r_width] * 2, dropout_p=dropout_p
            )
            model.set_device(device)
            model.fit(U_train, S_train, O_train, lr=adam_lr, num_epochs=5000)

        tot_forecast = torch.zeros((No, 0)).to("cpu")

        for i in range(10):
            fcast_start_index = U_train.shape[1] + i * fcast_steps
            fcast = model.forecast(
                U=concat_U[:, fcast_start_index : fcast_start_index + fcast_steps],
                U_spin=concat_U[:, fcast_start_index - lags + 1 : fcast_start_index],
                S_spin=concat_S[:, fcast_start_index - lags + 1 : fcast_start_index],
                s_k=concat_S[:, fcast_start_index : fcast_start_index + 1],
            )
            tot_forecast = torch.hstack((tot_forecast, fcast.to("cpu")))

    tot_forecast = tot_forecast.detach().cpu().numpy()
    O_valid = O_valid.detach().cpu().numpy()

    if scaled:
        tot_forecast = sensor_scaler.inverse_transform(tot_forecast.T).T
        O_valid = sensor_scaler.inverse_transform(O_valid.T).T

    fcast_dev = np.mean(np.linalg.norm((tot_forecast - O_valid), axis=0))

    ret_dict["model_type"] = model_type
    ret_dict["simulator"] = data_dict["simulator"]
    ret_dict["control_disc"] = data_dict["control_disc"]
    ret_dict["train_len"] = train_steps
    ret_dict["model_disc"] = data_dict["model_disc"]
    ret_dict["fcast_dev"] = fcast_dev
    ret_dict["scaled"] = scaled
    ret_dict["noise_level"] = noise_level

    return ret_dict, tot_forecast


def closed_loop_sim(simulator, forecaster, controller_params, ref_traj, forecast_horizon, control_horizon, num_steps):
    """
    Perform a closed-loop simulation using the given simulator and controller.

    Parameters:
    -----------
    simulator: object
        The simulation environment. Must have `initial_state` and `simulate` methods.
    forecaster: object
        The forecasting model for the controller. Must have a `forecast` method.
    controller_params: dict
        Parameters for the controller, including penalties and bounds.
    ref_traj: torch.DoubleTensor
        The reference trajectory to track, dims (No, num_steps).
    forecast_horizon: int
        The forecasting horizon for the controller.
    control_horizon: int
        The control horizon for the controller.
    num_steps: int
        The total number of steps to simulate.

    Returns:
    --------
    results: dict
        Contains the control inputs and sensor outputs during the simulation.
    """
    # Error handling for input parameters
    if not callable(getattr(simulator, "initial_state", None)) or not callable(getattr(simulator, "simulate", None)):
        raise TypeError("simulator must have `initial_state` and `simulate` methods.")
    if not callable(getattr(forecaster, "forecast", None)):
        raise TypeError("forecaster must have a `forecast` method.")
    if not isinstance(controller_params, dict):
        raise TypeError("controller_params must be a dictionary.")
    if not isinstance(ref_traj, torch.DoubleTensor):
        raise TypeError("ref_traj must be a torch.DoubleTensor.")
    if not isinstance(forecast_horizon, int) or forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be a positive integer.")
    if not isinstance(control_horizon, int) or control_horizon <= 0:
        raise ValueError("control_horizon must be a positive integer.")
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer.")

    # Initialize scaling objects
    sensor_scaler = MinMaxScaler()
    control_scaler = MinMaxScaler()

    # Fit control scaler with expected control range (e.g., [0, 1])
    control_scaler.fit(np.array([[0], [1]]))  # Assuming control values are normalized between 0 and 1

    # Fit sensor scaler with the reference trajectory
    sensor_scaler.fit(ref_traj.T)

    Nu, Ns = simulator.control_dim, simulator.sensor_dim

    # Scale reference trajectory
    ref_traj_scaled = torch.tensor(sensor_scaler.fit_transform(ref_traj.T).T, dtype=torch.float64)

    # Initialize simulator states
    s_k = simulator.initial_state()
    s_k_scaled = torch.tensor(sensor_scaler.transform(s_k.numpy().T).T, dtype=torch.float64)

    # Initialize control variables
    U_act_scaled = torch.ones((Nu, forecast_horizon), requires_grad=True) * 0.5

    # Initialize lists for logging results
    U_list = torch.zeros((Nu, 0))
    S_list = torch.zeros((Ns, 0))

    # Instantiate controller
    controller = MPController(forecaster=forecaster, **controller_params)

    for t_step in range(num_steps):
        if t_step % control_horizon == 0:
            # Prepare the control sequence
            U_act_temp = torch.zeros((Nu, forecast_horizon))
            U_act_temp[:, :control_horizon] = U_act_scaled[:, forecast_horizon - control_horizon:].data
            U_act_temp[:, control_horizon:] = U_act_scaled[:, -1].data.unsqueeze(1)
            U_act_scaled = U_act_temp.clone()
            U_act_scaled.requires_grad = True

            # Compute optimal control actions
            # Compute optimal control actions
            if isinstance(forecaster, Forecasters.ESNForecaster):
                r_k = torch.zeros((Ns, 1), dtype=torch.float64)  # Initialize r_k as needed
                U_act_scaled = controller.compute_act(
                    U=U_act_scaled,
                    ref_vals=ref_traj_scaled[:, t_step : t_step + forecast_horizon],
                    s_k=s_k_scaled,
                    U_last=U_act_scaled[:, control_horizon - 1 : control_horizon].clone(),
                    r_k=r_k,
                ).clone()
            else:
                # Prepare U_spin and S_spin
                spin_start = max(0, t_step - forecast_horizon + 1)
                U_spin = U_list[:, spin_start:t_step] if t_step > 0 else torch.zeros((Nu, forecast_horizon))
                S_spin = S_list[:, spin_start:t_step] if t_step > 0 else torch.zeros((Ns, forecast_horizon))

                U_act_scaled = controller.compute_act(
                    U=U_act_scaled,
                    ref_vals=ref_traj_scaled[:, t_step : t_step + forecast_horizon],
                    s_k=s_k_scaled,
                    U_last=U_act_scaled[:, control_horizon - 1 : control_horizon].clone(),
                    U_spin=U_spin,
                    S_spin=S_spin,
                ).clone()


        # Apply control input and advance simulator
        u_k_scaled = U_act_scaled[:, t_step % control_horizon : t_step % control_horizon + 1]
        u_k = torch.tensor(control_scaler.inverse_transform(u_k_scaled.numpy().T).T, dtype=torch.float64)

        s_k = simulator.simulate(U=u_k, x0=s_k)
        s_k_scaled = torch.tensor(sensor_scaler.transform(s_k.numpy().T).T, dtype=torch.float64)

        # Log results
        S_list = torch.hstack((S_list, s_k))
        U_list = torch.hstack((U_list, u_k))

    results = {
        "U": U_list,
        "S": S_list,
        "ref_traj": ref_traj,
    }

    return results
