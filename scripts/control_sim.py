import argparse
import json
import torch
import numpy as np
from RNNmpc import Forecasters
from RNNmpc import Simulators
from sklearn.preprocessing import MinMaxScaler
from RNNmpc import MPController
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument(
    "--sim",
    type=str,
    help="Simulation environment to control.",
    required=True,
)

parser.add_argument(
    "--model_type",
    type=str,
    help="Forecaster to use in MPC.",
    required=True,
)

parser.add_argument(
    "--train_data",
    type=str,
    help="Data dict to use for training.",
    required=True,
)

parser.add_argument(
    "--hyperparams_dict",
    type=str,
    help="Hyperparameter results dict to use.",
    required=True
)

parser.add_argument(
    "--dest",
    type=str,
    help="Save location for results dictionary.",
    required=True,
)

parser.add_argument(
    "--dev",
    type=float,
    help="Dev penalty.",
    required=False,
    default=100,
)

parser.add_argument(
    "--u_1",
    type=float,
    help="u_1 penalty.",
    required=False,
    default=1,
)

parser.add_argument(
    "--u_2",
    type=float,
    help="u_2 penalty.",
    required=False,
    default=20,
)

parser.add_argument(
    "--control_horizon",
    type=int,
    help="control horizon.",
    required=False,
    default=20,
)

parser.add_argument(
    "--forecast_horizon",
    type=int,
    help="forecast horizon.",
    required=False,
    default=50,
)

args = parser.parse_args()
train_data_dict = json.load(open(args.train_data))
hyperparams_dict = json.load(open(args.hyperparams_dict))
sim = args.sim
dest = args.dest
model_type = args.model_type
control_horizon = args.control_horizon
forecast_horizon = args.forecast_horizon
dev = args.dev
u_1 = args.u_1
u_2 = args.u_2

sensor_scaler = MinMaxScaler()
control_scaler = MinMaxScaler()

U_train = np.array(train_data_dict['U_train'])
S_train = np.array(train_data_dict['S_train'])
O_train = np.array(train_data_dict['O_train'])

control_scaler.fit(U_train.T)
sensor_scaler.fit(S_train.T)

U_train_scaled = torch.tensor(control_scaler.transform(U_train.T).T)
S_train_scaled = torch.tensor(sensor_scaler.transform(S_train.T).T)
O_train_scaled = torch.tensor(sensor_scaler.transform(O_train.T).T)

if sim == "LorenzControl":
    Nu = 1
    Ns = 3
    No = 3
    sim = Simulators.LorenzControl(control_disc=0.01)
    ref_traj1 = np.zeros((3, 500))
    ref_traj_scaled1 = torch.tensor(sensor_scaler.transform(ref_traj1.T).T)

    ref_traj2 = np.zeros((3, 500))
    ref_traj2[0] = np.sqrt(72)
    ref_traj2[1] = np.sqrt(72)
    ref_traj2[2] = 27

    ref_traj_scaled2 = torch.tensor(sensor_scaler.transform(ref_traj2.T).T)

    ref_traj3 = np.zeros((3, 700))
    ref_traj3[0] = -np.sqrt(72)
    ref_traj3[1] = -np.sqrt(72)
    ref_traj3[2] = 27

    ref_traj_scaled3 = torch.tensor(sensor_scaler.transform(ref_traj3.T).T)

    ref_traj_scaled = torch.hstack((ref_traj_scaled1, ref_traj_scaled2, ref_traj_scaled3))
    ref_traj = np.hstack((ref_traj1, ref_traj2, ref_traj3))


elif sim == "TwoTankControl":
    Nu = 2
    Ns = 2
    No = 2
    sim = Simulators.TwoTankControl()

    ref_traj1 = np.ones((2, 500)) * 0.25
    ref_traj_scaled1 = torch.tensor(sensor_scaler.transform(ref_traj1.T).T)

    ref_traj2 = np.ones((2, 500)) * 0.5

    ref_traj_scaled2 = torch.tensor(sensor_scaler.transform(ref_traj2.T).T)

    ref_traj3 = np.ones((2, 700)) * 0.1

    ref_traj_scaled3 = torch.tensor(sensor_scaler.transform(ref_traj3.T).T)

    ref_traj_scaled = torch.hstack((ref_traj_scaled1, ref_traj_scaled2, ref_traj_scaled3))
    ref_traj = np.hstack((ref_traj1, ref_traj2, ref_traj3))

elif sim == "StirredTankControl":
    Nu = 1
    Ns = 2
    No = 2
    sim = Simulators.StirredTankControl()

    ref_traj1 = np.ones((2, 500)) 
    ref_traj1[0,:] = 0.87725294608097
    ref_traj1[1,:] = 324.475443431599
    ref_traj_scaled1 = torch.tensor(sensor_scaler.transform(ref_traj1.T).T)

    ref_traj2 = np.ones((2, 500))
    ref_traj2[0,:] = 0.87725294608097 + 0.01 * 0.87725294608097
    ref_traj2[1,:] = 324.475443431599 - 0.0066 * 324.475443431599
    ref_traj_scaled2 = torch.tensor(sensor_scaler.transform(ref_traj2.T).T)

    ref_traj3 = np.ones((2, 700)) * 0.1
    ref_traj3[0,:] = 0.87725294608097 - 0.01 * 0.87725294608097
    ref_traj3[1,:] = 324.475443431599 + 0.0066 * 324.475443431599
    ref_traj_scaled3 = torch.tensor(sensor_scaler.transform(ref_traj3.T).T)

    ref_traj_scaled = torch.hstack((ref_traj_scaled1, ref_traj_scaled2, ref_traj_scaled3))
    ref_traj = np.hstack((ref_traj1, ref_traj2, ref_traj3))

elif sim == "SpringMassControl":
    Nu = 1
    Ns = 2
    No = 2
    sim = Simulators.SpringMassControl(model_disc=0.1, control_disc=0.1)

    ref_traj1 = np.ones((2, 500)) * -1.25
    ref_traj_scaled1 = torch.tensor(sensor_scaler.transform(ref_traj1.T).T)

    ref_traj2 = np.ones((2, 500)) * 0

    ref_traj_scaled2 = torch.tensor(sensor_scaler.transform(ref_traj2.T).T)

    ref_traj3 = np.ones((2, 700)) * 1.25

    ref_traj_scaled3 = torch.tensor(sensor_scaler.transform(ref_traj3.T).T)

    ref_traj_scaled = torch.hstack((ref_traj_scaled1, ref_traj_scaled2, ref_traj_scaled3))
    ref_traj = np.hstack((ref_traj1, ref_traj2, ref_traj3))

    s_k_scaled = O_train_scaled[:,-1:]
    s_k = torch.tensor(O_train[:,-1:])

elif sim == "CylinderControl":
    Nu = 1
    Ns = 1
    No = 1
    ref_traj1 = np.zeros((1, 500))
    ref_traj2 = np.ones((1,500)) * 0
    ref_traj3 = np.ones((1,700)) * 0 
    ref_traj = np.hstack((ref_traj1, ref_traj2, ref_traj3))
    ref_traj_scaled = torch.tensor(sensor_scaler.transform(ref_traj.T).T)
    restart = '/home/jmpw1/Documents/Control/rc-mpc/transient_run.h5'
    print('done')
    sim = Simulators.CylinderControl(model_disc=0.0025, control_disc=0.1, restart=restart)
    X_out = sim.simulate(torch.zeros(1,633))[0:1,:]
    fig1, ax1 = plt.subplots(1)
    ax1.plot(X_out[0].detach().numpy())
    plt.savefig('temptestspinup.pdf')
    print('done')
    s_k = X_out[:, -1:]
    s_k_scaled = torch.tensor(sensor_scaler.transform(X_out[:,-1:].detach().numpy().T).T, dtype=torch.float64)
else:
    raise ValueError("Invalid sim environment.")


if model_type == "ESNForecaster":
    alpha = hyperparams_dict['ESNForecaster']['alpha']
    rho_sr = hyperparams_dict['ESNForecaster']['rho_sr']
    sigma = hyperparams_dict['ESNForecaster']['sigma']
    sigma_b = hyperparams_dict['ESNForecaster']['sigma_b']
    beta = hyperparams_dict['ESNForecaster']['beta']

    Nr = 1000
    model = Forecasters.ESNForecaster(Nr=Nr, Nu=Nu, Ns=Ns, No=No, rho_sr=rho_sr, alpha=alpha, sigma=sigma, sigma_b=sigma_b)
    r_train = model.fit(U=U_train_scaled, S=S_train_scaled, O=O_train_scaled, beta=beta, spinup=300)

elif model_type == "LSTMForecaster":
    lags = hyperparams_dict['LSTMForecaster']['lags']
    adam_lr = hyperparams_dict['LSTMForecaster']['adam_lr']
    dropout_p = hyperparams_dict['LSTMForecaster']['dropout_p']
    Nr = hyperparams_dict['LSTMForecaster']['Nr']

    model = Forecasters.LSTMForecaster(Nr=Nr, Nu=Nu, Ns=Ns, No=No, dropout_p=dropout_p)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.set_device(device)
    r_train = model.fit(U=U_train_scaled, S=S_train_scaled, O=O_train_scaled, lags=lags, lr=adam_lr, num_epochs=5000)
    model.set_device("cpu")

elif model_type == "GRUForecaster":
    lags = hyperparams_dict['GRUForecaster']['lags']
    adam_lr = hyperparams_dict['GRUForecaster']['adam_lr']
    dropout_p = hyperparams_dict['GRUForecaster']['dropout_p']
    Nr = hyperparams_dict['GRUForecaster']['Nr']

    model = Forecasters.GRUForecaster(Nr=Nr, Nu=Nu, Ns=Ns, No=No, dropout_p=dropout_p)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.set_device(device)
    r_train = model.fit(U=U_train_scaled, S=S_train_scaled, O=O_train_scaled, lags=lags, lr=adam_lr, num_epochs=5000)
    model.set_device("cpu")

elif model_type == "LinearForecaster":
    beta = hyperparams_dict['LinearForecaster']['beta']
    tds = hyperparams_dict['LinearForecaster']['tds']

    lags = max(-np.array(tds))
    model = Forecasters.LinearForecaster(Nu=Nu, Ns=Ns, No=No, tds=tds)
    model.fit(
        U_train_scaled,
        S_train_scaled,
        O_train_scaled,
        beta=beta,
    )

elif model_type == "FCForecaster":
    r_width = hyperparams_dict['FCForecaster']['r_width']
    adam_lr = hyperparams_dict['FCForecaster']['adam_lr']
    dropout_p = hyperparams_dict['FCForecaster']['dropout_p']
    tds = hyperparams_dict['FCForecaster']['tds']

    lags = max(-np.array(tds))
    model = Forecasters.FCForecaster(
        Nu=Nu, Ns=Ns, No=No, tds=tds, r_list=[r_width] * 2, dropout_p=dropout_p
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.set_device(device)
    model.fit(U=U_train_scaled, S=S_train_scaled, O=O_train_scaled, lr=adam_lr, num_epochs=5000)
    model.set_device('cpu')

else:
    raise ValueError('Invalid forecaster model.')

controller = MPController(forecaster=model, dev=dev, u_1=u_1, u_2=u_2, soft_bounds=(0.05,0.95), hard_bounds=(0,1))

if model_type == 'ESNForecaster':
    if isinstance(sim, Simulators.SpringMassControl):
        r_k = model.spin(U_spin=torch.zeros((2,300), dtype=torch.float64), S_spin=torch.zeros((2,300), dtype=torch.float64))
        s_k_scaled = torch.tensor(sensor_scaler.transform(np.array([[0], [0]]).T).T)
        s_k = torch.tensor([[0],[0]], dtype=torch.float64)
        x_k = torch.tensor([[0], [0], [0], [0]], dtype=torch.float64)
    if isinstance(sim, Simulators.CylinderControl):
        X_out_scaled = sensor_scaler.transform(X_out.detach().numpy().T).T
        X_out_scaled = torch.tensor(X_out_scaled)
        r_k = model.spin(U_spin=torch.ones_like(X_out_scaled) * 0.5, S_spin=X_out_scaled)
    else:
        s_k_scaled = O_train_scaled[:,-1:]
        s_k = torch.tensor(O_train[:,-1:])
        r_k = model.spin(U_spin=U_train_scaled[:, -300:], S_spin=S_train_scaled)
    
    U_act_scaled = torch.ones((Nu, forecast_horizon), requires_grad=True) * 0.5
    control_step = 0
    U_list = torch.zeros((Nu,0))
    S_list = torch.zeros((Ns,0))
    U_list_scaled = torch.zeros((Nu,0))
    S_list_scaled = torch.zeros((Ns,0))
    for t_step in range(1600):
        if t_step % control_horizon == 0:
            U_act_temp = torch.zeros((Nu,forecast_horizon))

            U_act_temp[:,:control_horizon] = U_act_scaled[:,forecast_horizon - control_horizon:].data
            U_act_temp[:, control_horizon:] = U_act_scaled[:, -1].data.reshape(Nu, -1)

            U_act_scaled = U_act_temp
            U_act_scaled.requires_grad = True
            r_k = r_k.detach()

            U_act_scaled = controller.compute_act(U=U_act_scaled, 
                                                ref_vals=ref_traj_scaled[:,t_step:t_step+forecast_horizon], 
                                                s_k=s_k_scaled, U_last=U_act_scaled[:,control_horizon:control_horizon+1].clone(),
                                                r_k=r_k,
                                                ).clone()
            control_step = 0

        u_k_scaled = U_act_scaled[:, control_step:control_step+1]
        r_k = model.advance(u_k=u_k_scaled, s_k=s_k_scaled, r_k=r_k)


        u_k = control_scaler.inverse_transform(u_k_scaled.detach().numpy().T).T
        u_k = torch.tensor(u_k)

        if isinstance(sim, Simulators.SpringMassControl):
            x_k = sim.simulate(U=u_k, x0=x_k)
            s_k = x_k[[0,2],:]
        elif isinstance(sim, Simulators.CylinderControl):
            s_k = sim.simulate(U=u_k)[0:1, :]
        else:
            s_k = sim.simulate(U=u_k, x0=s_k)
        s_k_scaled = sensor_scaler.transform(s_k.detach().numpy().T).T
        s_k_scaled = torch.tensor(s_k_scaled)

        S_list = torch.hstack((S_list, s_k))
        S_list_scaled = torch.hstack((S_list_scaled, s_k_scaled))

        U_list = torch.hstack((U_list, u_k))
        U_list_scaled = torch.hstack((U_list_scaled, u_k_scaled))

        control_step += 1

else:
    if isinstance(sim, Simulators.SpringMassControl):
        s_k_scaled = torch.tensor(sensor_scaler.transform(np.array([[0], [0]]).T).T)
        s_k = torch.tensor([[0],[0]], dtype=torch.float64)
        x_k = torch.tensor([[0], [0], [0], [0]], dtype=torch.float64)
        U_list_scaled = torch.zeros((1, lags-1), dtype=torch.float64)
        S_list_scaled = torch.zeros((2, lags-1), dtype=torch.float64)
    else:
        s_k_scaled = O_train_scaled[:,-1:]
        s_k = torch.tensor(O_train[:,-1:])
        U_list_scaled = U_train_scaled[:,-lags+1:]
        S_list_scaled = S_train_scaled[:,-lags+1:]
    U_act_scaled = torch.ones((Nu, forecast_horizon), requires_grad=True) * 0.5
    control_step = 0
    U_list = torch.zeros((Nu,0))
    S_list = torch.zeros((Ns,0))
    
    for t_step in range(1600):
        if t_step % control_horizon == 0:
            U_act_temp = torch.zeros((Nu,forecast_horizon))

            U_act_temp[:,:control_horizon] = U_act_scaled[:,forecast_horizon - control_horizon:].data
            U_act_temp[:, control_horizon:] = U_act_scaled[:, -1].data.reshape(Nu, -1)

            U_act_scaled = U_act_temp
            U_act_scaled.requires_grad = True

            U_act_scaled = controller.compute_act(U=U_act_scaled, 
                                                ref_vals=ref_traj_scaled[:,t_step:t_step+forecast_horizon], 
                                                s_k=s_k_scaled, 
                                                U_last=U_act_scaled[:,control_horizon:control_horizon+1].clone(),
                                                U_spin=U_list_scaled[:,-lags+1:].detach(),
                                                S_spin=S_list_scaled[:, -lags+1:].detach(),
                                                ).clone()
            control_step = 0


        u_k_scaled = U_act_scaled[:, control_step:control_step+1]


        u_k = control_scaler.inverse_transform(u_k_scaled.detach().numpy().T).T
        u_k = torch.tensor(u_k)

        if isinstance(sim, Simulators.SpringMassControl):
            x_k = sim.simulate(U=u_k, x0=x_k)
            s_k = x_k[[0,2],:]
        else:
            s_k = sim.simulate(U=u_k, x0=s_k)
        s_k_scaled = sensor_scaler.transform(s_k.detach().numpy().T).T
        s_k_scaled = torch.tensor(s_k_scaled)

        S_list = torch.hstack((S_list, s_k))
        S_list_scaled = torch.hstack((S_list_scaled, s_k_scaled))

        U_list = torch.hstack((U_list, u_k))
        U_list_scaled = torch.hstack((U_list_scaled, u_k_scaled))

        control_step += 1

ret_dict = {
    "model_type": model_type,
    "sim": args.sim,
    "S_controlled": S_list.detach().numpy().tolist(),
    "ref_traj": ref_traj.tolist(),
    "dev": dev,
    "u_1": u_1,
    "u_2": u_2,
    "U": U_list.detach().numpy().tolist(),
}

with open(args.dest + "/control_results_cyl_2" + model_type + ".json", "w") as fp:
    json.dump(ret_dict, fp)