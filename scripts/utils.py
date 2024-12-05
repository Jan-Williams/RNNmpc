import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from RNNmpc import Forecasters, Simulators, MPController


def closed_loop_sim(environment, reference_trajectory, model_type, train_data, hyperparams, control_params):
    """
    Perform a closed-loop simulation using MPC and a surrogate model.

    Args:
        environment (str): Simulation environment (e.g., 'LorenzControl', 'TwoTankControl').
        reference_trajectory (np.ndarray): Desired reference trajectory for the system.
        model_type (str): Type of forecaster model (e.g., 'LSTMForecaster', 'GRUForecaster').
        train_data (dict): Dictionary containing 'U_train', 'S_train', and 'O_train'.
        hyperparams (dict): Dictionary of hyperparameters for the chosen model type.
        control_params (dict): Parameters for the controller and simulation.

    Returns:
        dict: A dictionary containing results of the controlled simulation.
    """

    # Initialize scalers
    sensor_scaler = MinMaxScaler()
    control_scaler = MinMaxScaler()

    # Extract and scale training data
    U_train = np.array(train_data['U_train'])
    S_train = np.array(train_data['S_train'])
    O_train = np.array(train_data['O_train'])

    control_scaler.fit(U_train.T)
    sensor_scaler.fit(S_train.T)

    U_train_scaled = torch.tensor(control_scaler.transform(U_train.T).T, dtype=torch.float64)
    S_train_scaled = torch.tensor(sensor_scaler.transform(S_train.T).T, dtype=torch.float64)
    O_train_scaled = torch.tensor(sensor_scaler.transform(O_train.T).T, dtype=torch.float64)

    # Initialize simulation environment
    if environment == "LorenzControl":
        sim = Simulators.LorenzControl(control_disc=0.01)
    elif environment == "TwoTankControl":
        sim = Simulators.TwoTankControl()
    elif environment == "StirredTankControl":
        sim = Simulators.StirredTankControl()
    elif environment == "SpringMassControl":
        sim = Simulators.SpringMassControl(model_disc=0.1, control_disc=0.1)
    elif environment == "CylinderControl":
        sim = Simulators.CylinderControl(model_disc=0.0025, control_disc=0.1, restart=None)
    else:
        raise ValueError("Invalid simulation environment.")

    ref_traj_scaled = torch.tensor(sensor_scaler.transform(reference_trajectory.T).T, dtype=torch.float64)

    # Determine simulation dimensions
    Nu, Ns, No = sim.input_output_dims()

    # Initialize forecaster model
    if model_type == "LSTMForecaster":
        model = Forecasters.LSTMForecaster(
            Nr=hyperparams['Nr'], Nu=Nu, Ns=Ns, No=No, dropout_p=hyperparams['dropout_p']
        )
        model.set_device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        model.fit(U_train_scaled, S_train_scaled, O_train_scaled, lags=hyperparams['lags'], lr=hyperparams['adam_lr'])
        model.set_device("cpu")
    elif model_type == "GRUForecaster":
        model = Forecasters.GRUForecaster(
            Nr=hyperparams['Nr'], Nu=Nu, Ns=Ns, No=No, dropout_p=hyperparams['dropout_p']
        )
        model.set_device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        model.fit(U_train_scaled, S_train_scaled, O_train_scaled, lags=hyperparams['lags'], lr=hyperparams['adam_lr'])
        model.set_device("cpu")
    elif model_type == "LinearForecaster":
        model = Forecasters.LinearForecaster(Nu=Nu, Ns=Ns, No=No, tds=hyperparams['tds'])
        model.fit(U_train_scaled, S_train_scaled, O_train_scaled, beta=hyperparams['beta'])
    elif model_type == "FCForecaster":
        model = Forecasters.FCForecaster(
            Nu=Nu, Ns=Ns, No=No, tds=hyperparams['tds'], 
            r_list=[hyperparams['r_width']] * 2, dropout_p=hyperparams['dropout_p']
        )
        model.set_device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        model.fit(U_train_scaled, S_train_scaled, O_train_scaled, lr=hyperparams['adam_lr'], num_epochs=5000)
        model.set_device("cpu")
    elif model_type == "ESNForecaster":
        model = Forecasters.ESNForecaster(
            Nr=1000, Nu=Nu, Ns=Ns, No=No, 
            rho_sr=hyperparams['rho_sr'], alpha=hyperparams['alpha'],
            sigma=hyperparams['sigma'], sigma_b=hyperparams['sigma_b']
        )
        model.fit(U=U_train_scaled, S=S_train_scaled, O=O_train_scaled, beta=hyperparams['beta'], spinup=300)
    else:
        raise ValueError("Unsupported model type.")

    # Configure the MPC controller
    controller = MPController(
        forecaster=model,
        dev=control_params.get('dev', 100),
        u_1=control_params.get('u_1', 1),
        u_2=control_params.get('u_2', 20),
        soft_bounds=(0.05, 0.95),
        hard_bounds=(0, 1)
    )

    # Prepare for closed-loop simulation
    control_horizon = control_params.get('control_horizon', 20)
    forecast_horizon = control_params.get('forecast_horizon', 50)
    S_list, U_list = [], []
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

    return {
        "controlled_states": np.hstack(S_list),
        "applied_controls": np.hstack(U_list),
        "reference_trajectory": reference_trajectory
    }
