import RNNmpc.Forecasters as Forecaster
import RNNmpc.Simulators as Simulator
from RNNmpc import MPController
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
        model = Forecaster.ESNForecaster(
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
            model = Forecaster.GRUForecaster(
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
            model = Forecaster.LSTMForecaster(
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
            model = Forecaster.LinearForecaster(Nu=Nu, Ns=Ns, No=No, tds=tds)
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
            model = Forecaster.FCForecaster(
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

def closed_loop_sim(environment, reference_trajectory, model_type, train_data, hyperparams, control_params):
    """
    Perform a closed-loop simulation using MPC and a surrogate model.

    Args:
        environment (str): Simulation environment (e.g., 'LorenzControl', 'TwoTankControl').
        reference_trajectory (np.ndarray): Desired reference trajectory for the system.
        model_type (str): Type of surrogate model (e.g., 'LSTMForecaster', 'GRUForecaster').
        train_data (dict): Dictionary containing 'U_train', 'S_train', and 'O_train'.
        hyperparams (dict): Dictionary of hyperparameters for the chosen forecaster.
        control_params (dict): Parameters for the controller and simulation.

    Returns:
        dict: A dictionary containing results of the controlled simulation.

    Raises:
        ValueError: If inputs are invalid or unsupported configurations are detected.
    """

    # Validate inputs
    valid_environments = ["LorenzControl", "TwoTankControl", "StirredTankControl", "SpringMassControl"]
    if environment not in valid_environments:
        raise ValueError(
            f"Invalid environment '{environment}'. Supported environments: {valid_environments}"
        )

    valid_models = ["LSTMForecaster", "GRUForecaster", "LinearForecaster", "FCForecaster", "ESNForecaster"]
    if model_type not in valid_models:
        raise ValueError(
            f"Invalid model type '{model_type}'. Supported models: {valid_models}"
        )

    if not isinstance(reference_trajectory, np.ndarray):
        raise TypeError("reference_trajectory must be a NumPy array.")

    if not isinstance(train_data, dict) or not all(key in train_data for key in ["U_train", "S_train", "O_train"]):
        raise ValueError(
            "train_data must be a dictionary containing keys: 'U_train', 'S_train', and 'O_train'."
        )

    if not isinstance(hyperparams, dict):
        raise TypeError("hyperparams must be a dictionary.")

    if not isinstance(control_params, dict):
        raise TypeError("control_params must be a dictionary.")

    # Ensure control_params has required keys
    required_control_keys = ["dev", "u_1", "u_2", "control_horizon", "forecast_horizon"]
    for key in required_control_keys:
        if key not in control_params:
            raise ValueError(f"Missing control parameter: '{key}' in control_params.")

    # Initialize scalers
    sensor_scaler = MinMaxScaler()
    control_scaler = MinMaxScaler()

    # Extract and validate training data
    try:
        U_train = np.array(train_data['U_train'])
        S_train = np.array(train_data['S_train'])
        O_train = np.array(train_data['O_train'])
    except KeyError as e:
        raise ValueError(f"Missing required key in train_data: {e}")
    except Exception as e:
        raise ValueError(f"Error processing training data: {e}")

    # Ensure training data dimensions match
    if U_train.ndim != 2 or S_train.ndim != 2 or O_train.ndim != 2:
        raise ValueError("Training data must be 2D arrays.")

    control_scaler.fit(U_train.T)
    sensor_scaler.fit(S_train.T)

    U_train_scaled = torch.tensor(control_scaler.transform(U_train.T).T, dtype=torch.float64)
    S_train_scaled = torch.tensor(sensor_scaler.transform(S_train.T).T, dtype=torch.float64)
    O_train_scaled = torch.tensor(sensor_scaler.transform(O_train.T).T, dtype=torch.float64)

    # Initialize simulation environment
    try:
        if environment == "LorenzControl":
            sim = Simulator.LorenzControl(control_disc=0.01)
        elif environment == "TwoTankControl":
            sim = Simulator.TwoTankControl()
        elif environment == "StirredTankControl":
            sim = Simulator.StirredTankControl()
        elif environment == "SpringMassControl":
            sim = Simulator.SpringMassControl(model_disc=0.1, control_disc=0.1)
        else:
            raise ValueError("Invalid simulation environment.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize simulation environment: {e}")

    ref_traj_scaled = torch.tensor(sensor_scaler.transform(reference_trajectory.T).T, dtype=torch.float64)

    # Determine simulation dimensions
    try:
        Nu, Ns, No = sim.input_output_dims()
    except Exception as e:
        raise RuntimeError(f"Failed to determine simulation dimensions: {e}")

    # Initialize forecaster model
    try:
        if model_type == "LSTMForecaster":
            model = Forecaster.LSTMForecaster(
                Nr=hyperparams['Nr'], Nu=Nu, Ns=Ns, No=No, dropout_p=hyperparams['dropout_p']
            )
        elif model_type == "GRUForecaster":
            model = Forecaster.GRUForecaster(
                Nr=hyperparams['Nr'], Nu=Nu, Ns=Ns, No=No, dropout_p=hyperparams['dropout_p']
            )
        elif model_type == "LinearForecaster":
            model = Forecaster.LinearForecaster(Nu=Nu, Ns=Ns, No=No, tds=hyperparams['tds'])
        elif model_type == "FCForecaster":
            model = Forecaster.FCForecaster(
                Nu=Nu, Ns=Ns, No=No, tds=hyperparams['tds'],
                r_list=[hyperparams['r_width']] * 2, dropout_p=hyperparams['dropout_p']
            )
        elif model_type == "ESNForecaster":
            model = Forecaster.ESNForecaster(
                Nr=1000, Nu=Nu, Ns=Ns, No=No,
                rho_sr=hyperparams['rho_sr'], alpha=hyperparams['alpha'],
                sigma=hyperparams['sigma'], sigma_b=hyperparams['sigma_b']
            )
        else:
            raise ValueError("Unsupported model type.")
    except KeyError as e:
        raise ValueError(f"Missing hyperparameter for model {model_type}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize forecaster model: {e}")

    # Train the model
    try:
        if model_type in ["LSTMForecaster", "GRUForecaster"]:
            model.set_device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            model.fit(U_train_scaled, S_train_scaled, O_train_scaled, lags=hyperparams['lags'], lr=hyperparams['adam_lr'])
            model.set_device("cpu")
        elif model_type == "LinearForecaster":
            model.fit(U_train_scaled, S_train_scaled, O_train_scaled, beta=hyperparams['beta'])
        elif model_type == "FCForecaster":
            model.set_device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            model.fit(U_train_scaled, S_train_scaled, O_train_scaled, lr=hyperparams['adam_lr'], num_epochs=5000)
            model.set_device("cpu")
        elif model_type == "ESNForecaster":
            model.fit(U_train_scaled, S_train_scaled, O_train_scaled, beta=hyperparams['beta'], spinup=300)
    except Exception as e:
        raise RuntimeError(f"Failed to train forecaster model {model_type}: {e}")

    # Configure the MPC controller
    try:
        controller = MPController(
            forecaster=model,
            dev=control_params.get('dev', 100),
            u_1=control_params.get('u_1', 1),
            u_2=control_params.get('u_2', 20),
            soft_bounds=(0.05, 0.95),
            hard_bounds=(0, 1)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to configure MPC controller: {e}")

    # Prepare for closed-loop simulation
    control_horizon = control_params.get('control_horizon', 20)
    forecast_horizon = control_params.get('forecast_horizon', 50)

    # Initialize lists to store simulation results
    S_list = []
    U_list = []

    # Initial state
    s_k = torch.zeros((Ns, 1), dtype=torch.float64)

    # Perform simulation
    for t_step in range(1600):
        if t_step % control_horizon == 0:
            U_act_scaled = controller.compute_act(
                U=torch.ones((Nu, forecast_horizon), dtype=torch.float64) * 0.5,
                ref_vals=ref_traj_scaled[:, t_step:t_step + forecast_horizon],
                s_k=s_k,
                r_k=torch.zeros((model.Nr, 1), dtype=torch.float64)
            )
        u_k = control_scaler.inverse_transform(U_act_scaled.detach().numpy().T).T
        s_k = sim.simulate(U=torch.tensor(u_k, dtype=torch.float64))

        # Collect results
        S_list.append(s_k.detach().numpy())
        U_list.append(u_k)

    # Final output
    return {
        "controlled_states": np.hstack(S_list),
        "applied_controls": np.hstack(U_list),
        "reference_trajectory": reference_trajectory
    }
