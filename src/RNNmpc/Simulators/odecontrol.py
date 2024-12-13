"""simulator class for general ODEs"""

from copy import deepcopy
import numpy as np
import torch
import scipy.integrate


class OdeControl:
    """Simulation object for ODE systems.

    Attributes:
    -----------
    control_disc: float
        control discretization
    rtol: float
        relative tolerance for scipy.integrate.solve_ivp
    atol: float
        absolute tolerance for scipy.integrate.solve_ivp
    default_x0: np.array
        reasonable initial condition from which to simulate

    Methods:
    -----------
    control_step(U,x0)
        advance system one control timestep
    simulate(U, x0)
        simulate system under control input U
    generate_data(num_trajectories,Tfinal)
        generate training data for network
    """

    def __init__(
        self, dynamics, x0: np.array, nu: int, control_disc: float = 0.1
    ) -> None:
        """Initialize system with specified control discretization.

        No model discretization is used because we use the variable
        time-stepper in scipy.integrate.solve_ivp.

        Parameters:
        -----------
        dynamics: function
            dynamics function to be used.
        x0: np.array
            initial condition
        nu: int
            size of control vector
        control_disc: float
            control discretization
        """
        self.dynamics = dynamics
        self.control_disc = control_disc
        self.rtol = 1e-12
        self.atol = 1e-12
        self.default_x0 = torch.tensor(x0, dtype=torch.float64)
        self.nu = nu
        self.nx = len(x0)

        try:
            if np.shape(dynamics(0, x0, np.zeros(nu))) != np.shape(x0):
                raise ValueError(
                    "initial conditions vector is not the correct length"
                )
        except Exception as exc:
            raise ValueError(
                "initial conditions vector is not the correct length"
            ) from exc

        try:
            dynamics(0, x0, np.zeros(nu))
        except Exception as exc:
            raise ValueError("nu is too small") from exc

        try:
            dynamics(0, x0, np.zeros(nu - 1))
            raise ValueError("nu is too big")
        except Exception as exc:
            pass

    def control_step(self, U: np.array, x0: np.array) -> np.array:
        """Advance system one control step.

        Parameters:
        ----------
        U: np.array
            control for control model step, dims (Nu, 1)
        x0: np.array
            initial condition for control step, dims (3,1)

        Returns:
        ----------
        x0: np.array
            system state advanced by one control step, dims (3,1)
        """
        x0 = x0.flatten()
        sol = scipy.integrate.solve_ivp(
            self.dynamics,
            t_span=[0, self.control_disc + 1e-8],
            y0=x0,
            args=(U,),
            rtol=self.rtol,
            atol=self.atol,
            t_eval=np.arange(0, 2 * self.control_disc, self.control_disc),
        )

        return sol.y[:, -1]

    def simulate(
        self, U: torch.DoubleTensor, x0: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Simulate the system under given control U.

        Parameters:
        -----------
        U: torch.DoubleTensor
            control inputs for simulation, dims (Nu, t_steps)
        x0: torch.DoubleTensor
            initial condition, dims (3, 1)

        Returns:
        ----------
        X_list: torch.DoubleTensor
            evolution of system under control inputs, dims (3, t_steps)
        """
        U = U.detach().numpy()
        x0 = x0.detach().numpy()
        t_steps = U.shape[1]
        X_list = np.empty((x0.size, t_steps))
        X_list[:, 0] = x0
        for step in range(1, t_steps):
            x0 = self.control_step(U[:, step], x0)
            X_list[:, step] = x0
        X_list = torch.tensor(X_list, dtype=torch.float64)
        return X_list

    def generate_data(
        self,
        train_len: float,
        switching_period: float,
        dist_min: float,
        dist_max: float,
        train_percentage: float,
    ):
        """Generate random trajectory for training a neural network

        Parameters:
        -----------
        train_len: float
            length of training data in seconds
        switching_period: float
            Time interval between control changes in training input
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

            U_train: np.array
                control inputs of training data trajectory
            U_valid: np.array
                control inputs of validation data trajectory
            S_train: np.array
                State vectors of training data trajectory
            S_valid: np.array
                State vectors of validation data trajectory
            O_train: np.array
                Time shifted state vectors of training data trajectory
            O_valid: np.array
                Time shifted state vectors of validation data trajectory
        """

        # make sure that dist_min != dist_max so there is a std deviation
        if dist_min == dist_max:
            raise ValueError("dist_min cannot equal dist_max")
        if dist_min > dist_max:
            raise ValueError("dist_min cannot be greater than dist_max")

        # generate random control inputs
        train_control_vals = np.random.uniform(
            dist_min,
            dist_max,
            size=(self.nu, int(train_len / switching_period)),
        )

        train_control_sig = torch.tensor(
            np.array(train_control_vals), dtype=torch.float64
        )
        train_control_sig_len = int(
            train_control_sig.shape[1] * train_percentage / 100
        )

        tot_sig = deepcopy(train_control_sig)
        tot_out = self.simulate(
            tot_sig, torch.DoubleTensor(self.default_x0).squeeze()
        )

        # create data mats of control inputs, states and time shifted states
        U_train = tot_sig[:, :train_control_sig_len]
        S_train = tot_out[:, :train_control_sig_len]
        O_train = tot_out[:, 1: train_control_sig_len + 1]

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
            "train_len": train_len,
            "switching_period": switching_period,
            "control_disc": self.control_disc,
        }

        return return_dict
