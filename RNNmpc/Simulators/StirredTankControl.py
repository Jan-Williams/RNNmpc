from neuromancer import psl
import numpy as np
import torch 

class StirredTankControl:
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
    """

    def __init__(self, model_disc: float = 0.1, control_disc: float = 0.1) -> None:
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
        self.sys = psl.nonautonomous.CSTR()
        self.sys.ts = model_disc
        self.model_steps = int(control_disc / model_disc)
        self.default_x0 = np.array([[0.8773696], [324.24042]])
        # Default control distribution should be taken as U(297, 303)

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
                U_rep = np.hstack((U_rep, torch.zeros((1, 1))))
            x0 = self.control_step(x0=x0, U=U_rep)
            X_list[:, step : step + 1] = x0
        X_list = torch.tensor(X_list, dtype=torch.float64)
        return X_list