from neuromancer import psl
import numpy as np
import torch
from copy import deepcopy
import json


class neuromancerControl:
    """Simulation object for stirred tank system.

    Attributes:
    -----------
    model_disc: float
        underlying model dsicretization
    control_disc: float
        control discretization
    sys: psl.nonautonomous.CSTR
        underlying simulation environment
    model_steps: int
        number of model timesteps per control timestep
    default_x0: np.array
        reasonable initial condition from which to simulate

    Methods:
    -----------
    control_step(U,x0)
        advance system one control timestep
    simulate(U, x0)
        simulate system under control input U
    generate_data()
        generate a trajectory for training
    """

    def __init__(self, sys, model_disc: float = 0.1, control_disc: float = 0.1) -> None:
        """Initialize stirred tank system with specified control and model
        discretizations.

        Parameters:
        -----------
        model_disc: float
            underlying model dsicretization
        control_disc: float
            control discretization
        """
        self.model_disc = model_disc
        self.control_disc = control_disc
        self.sys = sys
        self.sys.ts = model_disc
        self.model_steps = int(control_disc / model_disc)
        self.default_x0 = np.array(sys.params[0]['x0'])
        self.nx = len(sys.params[0]['x0'])
        self.nu = sys.get_U(1).size

    def control_step(self, U: np.array, x0: np.array) -> np.array:
        """Advance system one control step.

        Parameters:
        ----------
        U: np.array
            control for control model step, dims (Nu, 1)
        x0: np.array
            initial condition for control step, dims (2,1)

        Returns:
        ----------
        x0: np.array
            system state advanced by one control step, dims (2,1)
        """
        out_dict = self.sys.simulate(x0=x0.flatten(), U=U.T)
        return out_dict["X"][-1:, :].T

    def simulate(
        self, U: torch.DoubleTensor, x0: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Simulate the system under given control U.

        Parameters:
        -----------
        U: torch.DoubleTensor
            control inputs for simulation, dims (Nu, t_steps)
        x0: torch.DoubleTensor
            initial condition, dims (2, 1)

        Returns:
        ----------
        X_list: torch.DoubleTensor
            evolution of system under control inputs, dims (2, t_steps)
        """
        t_steps = U.shape[1]
        X_list = np.empty((x0.shape[0], t_steps))
        for step in range(t_steps):
            U_rep = np.repeat(U[:, step : step + 1], self.model_steps, axis=1)
            if step < t_steps - 1:
                U_rep = np.hstack((U_rep, U[:, step + 1 : step + 2]))
            else:
                U_rep = np.hstack((U_rep, torch.zeros((self.nu, 1))))
            x0 = self.control_step(x0=x0, U=U_rep)
            X_list[:, step : step + 1] = x0
        X_list = torch.tensor(X_list, dtype=torch.float64)
        return X_list
    

    
    def generate_data(self, train_len: float, switching_period: float,
                      filter_len: int, dist_min: float, dist_max: float,
                      train_percentage: float):
        """Generate random trajectories for training a neural network

        Parameters:
        -----------
        train_len: float
            length of training data in seconds
        switching_period: float
            Time interval between control changes in training input
        filter_len: int
            Width of smoothing filter applied to control signal
        dist_min: float
            Minimum value of initial states
        dist_max: float
            Maximum value of initial states
        train_percentage: float
            percent of trajectory to use for training

        Outputs:
        -----------
        return_dict: dictionary
            Dictionary of trajectory data and metadata

            U_train: list
                control inputs of training data trajectory
            U_valid: list
                control inputs of validation data trajectory
            S_train: list
                State vectors of training data trajectory
            S_valid: list
                State vectors of validation data trajectory
            O_train: list
                Time shifted state vectors of training data trajectory
            O_valid: list
                Time shifted state vectors of validation data trajectory
        """

        #make sure that dist_min != dist_max so there is a std deviation
        if dist_min == dist_max:
            raise ValueError("dist_min cannot equal dist_max")
        if dist_min > dist_max:
            raise ValueError("dist_min cannot be greater than dist_max")
        if switching_period <= 0:
            raise ValueError("The switching period must be positive non-zero")

        #generate random control inputs
        train_control_vals = np.random.uniform(
            dist_min, dist_max, size = (self.nu, int(train_len / switching_period))
        )
        train_control_sig = np.repeat(
            train_control_vals, int(switching_period / self.control_disc)
        )
        train_control_sig = train_control_vals

        train_control_sig = torch.tensor(np.array(train_control_sig), dtype=torch.float64)
        train_control_sig_len = int(np.round(train_control_sig.shape[1]*train_percentage/100))

        tot_sig = deepcopy(train_control_sig)
        tot_sig = tot_sig.reshape([self.nu,-1])
        tot_out = self.simulate(tot_sig,torch.DoubleTensor(self.default_x0))
        
        U_train = tot_sig[:, :train_control_sig_len]
        S_train = tot_out[:, :train_control_sig_len]
        O_train = tot_out[:, 1:train_control_sig_len + 1]

        U_valid = tot_sig[:, train_control_sig_len:]
        S_valid = tot_out[:, train_control_sig_len:-1]
        O_valid = tot_out[:, train_control_sig_len + 1:]

        return_dict = {
            "U_train": U_train.detach().numpy().tolist(),
            "U_valid": U_valid.detach().numpy().tolist(),
            "S_train": S_train.detach().numpy().tolist(),
            "S_valid": S_valid.detach().numpy().tolist(),
            "O_train": O_train.detach().numpy().tolist(),
            "O_valid": O_valid.detach().numpy().tolist(),
            #"simulator": simulator, #should this be returned???
            "train_len": train_len,
            #"fcast_lens": fcast_lens,
            "switching_period": switching_period,
            "model_disc": self.model_disc,
            "control_disc": self.control_disc,
            "filter_len": filter_len,
        }

        return return_dict
        #with open(dest + "/data.json", "w+") as fp:
        #    json.dump(return_dict, fp)