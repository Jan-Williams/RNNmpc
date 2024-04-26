import RNNmpc.Simulators as Simulators
import numpy as np
import torch
from copy import deepcopy
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--simulator",
    type=str,
    help="Options are class names in Simulators.",
    required=True,
)

parser.add_argument(
    "--train_len",
    type=float,
    help="Length of training data in seconds.",
    required=True,
)

parser.add_argument(
    "--fcast_lens",
    type=float,
    help="Valid dataset consists of 10 forecasts of length fcast_len.",
    required=True,
)

parser.add_argument(
    "--switching_period",
    type=float,
    help="Time interval between control changes in training input.",
    required=True,
)

parser.add_argument(
    "--model_disc",
    type=float,
    help="Time step of underlying simulator.",
    required=True,
)

parser.add_argument(
    "--control_disc",
    type=float,
    help="Time step of controller model.",
    required=True,
)

parser.add_argument(
    "--filter_len",
    type=int,
    help="Width of smoothing filter applied to control signal.",
    required=True,
)

parser.add_argument(
    "--dest",
    type=str,
    help="Destination to save output dictionary.",
    required=True,
)


args = parser.parse_args()

simulator = args.simulator
train_len = args.train_len
fcast_lens = args.fcast_lens
switching_period = args.switching_period
model_disc = args.model_disc
control_disc = args.control_disc
filter_len = args.filter_len
dest = args.dest

if args.simulator == "SpringMassControl":
    # For the data generated in the paper:
    # simulator = SpringMassControl
    # train_len = 5000
    # fcast_lens = 5
    # switching_period = 0.5
    # model_disc = 0.1
    # control_disc = 0.1
    # filter_len = 2
    sim = Simulators.SpringMassControl(model_disc=model_disc, control_disc=control_disc)
    Nu = 1
    Ns = 2
    No = 2
    amplitudes = [0.0, 0.5, 1.0, 1.5, 2.0]
    x0 = torch.tensor([[1], [0], [1], [0]])
    dist_min = -3
    dist_max = 3
    fcast_freq = 2

elif simulator == "TwoTankControl":
    # For the data generated in the paper:
    # simulator = TwoTankControl
    # train_len = 50000
    # fcast_lens = 50
    # switching_period = 50
    # model_disc = 1.0
    # control_disc = 1.0
    # filter_len = 10
    sim = Simulators.TwoTankControl(model_disc=model_disc, control_disc=control_disc)
    Nu = 2
    Ns = 2
    No = 2
    amplitudes = [0, 0.25, 0.5, 0.75, 1.0]
    x0 = sim.default_x0
    dist_min = 0.0
    dist_max = 0.4
    fcast_freq = 0.25

elif simulator == "StirredTankControl":
    # For the data generated in the paper:
    # simulator = StirredTankControl
    # train_len = 5000
    # fcast_lens = 5
    # switching_period = 0.5
    # model_disc = 0.1
    # control_disc = 0.1
    # filter_len = 2
    sim = Simulators.StirredTankControl(
        model_disc=model_disc, control_disc=control_disc
    )
    Nu = 1
    Ns = 2
    No = 2
    amplitudes = [0, 0.5, 1.0, 1.5, 2.0]
    dist_min = 297
    dist_max = 303
    fcast_freq = 2
    x0 = sim.default_x0

elif simulator == "LorenzControl":
    # For the data generated in the paper:
    # simulator = LorenzControl
    # train_len = 500
    # fcast_lens = 0.5
    # switching_period = 0.05
    # control_disc = 0.01
    # filter_len = 2
    sim = Simulators.LorenzControl(control_disc=control_disc)
    Nu = 1
    Ns = 3
    No = 3
    amplitudes = [0, 0.5, 1.0, 1.5, 2.0]
    x0 = sim.default_x0
    dist_min = -50
    dist_max = 50
    fcast_freq = 20

elif simulator == "CylinderControl":
    sim = Simulators.CylinderControl(
        model_disc=model_disc,
        control_disc=control_disc,
    )
    Nu = 1
    Ns = 2
    No = 2
    amplitudes = [0, 0.33, 0.66, 1, 1.33]
    dist_min = -np.pi / 2
    dist_max = np.pi / 2
    fcast_freq = 2


else:
    raise Exception("Simulator must be the name of a class in Simulators.py.")

train_control_vals = np.random.uniform(
    dist_min, dist_max, size=(Nu, int(train_len / switching_period))
)
train_control_sig = np.repeat(
    train_control_vals, int(switching_period / control_disc), axis=1
)
smoothing_filter = np.ones(filter_len) / filter_len
temp_list = []
for dim in range(Nu):
    temp = np.convolve(train_control_sig[dim], smoothing_filter, mode="valid")
    temp_list.append(np.convolve(temp, smoothing_filter, mode="valid"))

train_control_sig = torch.tensor(np.array(temp_list), dtype=torch.float64)
train_control_mean = torch.mean(train_control_sig, axis=1).reshape(-1, 1)
train_control_std = torch.std(train_control_sig, axis=1).reshape(-1, 1)
train_control_sig_len = train_control_sig.shape[1]

tot_sig = deepcopy(train_control_sig)
for amp in range(len(amplitudes)):
    amps = amplitudes[amp] if amp % 2 else -amplitudes[amp]
    temp_const_sig = train_control_mean + amps * train_control_std * torch.ones(
        (train_control_mean.shape[0], int(fcast_lens / control_disc)),
        dtype=torch.float64,
    )

    t_range = torch.arange(0, fcast_lens, control_disc)
    temp_sin_sig = train_control_mean + amps * train_control_std * torch.sin(
        t_range * fcast_freq
    ) * torch.ones(
        (train_control_mean.shape[0], int(fcast_lens / control_disc)),
        dtype=torch.float64,
    )
    tot_sig = torch.hstack((tot_sig, temp_const_sig, temp_sin_sig))

if simulator == "CylinderControl":
    tot_out = sim.simulate(tot_sig)
else:
    tot_out = sim.simulate(tot_sig, x0)


U_train = tot_sig[:, 1:train_control_sig_len]
S_train = tot_out[:, : train_control_sig_len - 1]
O_train = tot_out[:, 1:train_control_sig_len]

U_valid = tot_sig[:, train_control_sig_len:]
S_valid = tot_out[:, train_control_sig_len - 1 : -1]
O_valid = tot_out[:, train_control_sig_len:]

return_dict = {
    "U_train": U_train.detach().numpy().tolist(),
    "U_valid": U_valid.detach().numpy().tolist(),
    "S_train": S_train.detach().numpy().tolist(),
    "S_valid": S_valid.detach().numpy().tolist(),
    "O_train": O_train.detach().numpy().tolist(),
    "O_valid": O_valid.detach().numpy().tolist(),
    "simulator": simulator,
    "train_len": train_len,
    "fcast_lens": fcast_lens,
    "switching_period": switching_period,
    "model_disc": model_disc,
    "control_disc": control_disc,
    "filter_len": filter_len,
}

with open(dest + "/data.json", "w+") as fp:
    json.dump(return_dict, fp)
