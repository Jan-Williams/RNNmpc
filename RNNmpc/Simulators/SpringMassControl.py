import control
import numpy as np
import torch

class SpringMassControl:
    """Simulation object for spring mass system.

    Attributes:
    -----------
    model_disc: float
        underlying model dsicretization
    control_disc: float
        control discretization
    A: np.array
        discretized A matrix
    B: np.array
        discretized B matrix
    C: np.array
        C matrix
    model_steps: int
        number of mmodel timesteps per control timestep

    Methods:
    -----------
    model_step(U,x0)
        advance system one model timestep
    control_step(U,x0)
        advance system one control timestep
    simulate(U, x0)
        simulate system under control input U
    """

    def __init__(self, model_disc: float, control_disc: float = None) -> None:
        """Initialize spring mass model with control and model discretizations.

        Parameters:
        ----------
        model_disc: float
            underlying model dsicretization
        control_disc: float
            control discretization
        A: np.array
            discretized A matrix
        B: np.array
            discretized B matrix
        C: np.array
            C matrix
        """
        if control_disc is None:
            control_disc = model_disc

        self.model_disc = model_disc
        self.control_disc = control_disc
        self.model_steps = int(control_disc / model_disc)

        k1 = 2
        k2 = 2
        b = 0.2
        m1 = 1
        m2 = 1

        A_cts = np.array(
            [
                [0, 1, 0, 0],
                [-2 * k2 / m2, -b / m2, k2 / m2, 0],
                [0, 0, 0, 1],
                [k2 / m1, 0, -k1 / m1, -b / m1],
            ]
        )

        B_cts = np.array([[0], [1 / m2], [0], [0]])

        C_cts = np.array([[1, 0, 1, 0]])

        sys_cts = control.StateSpace(A_cts, B_cts, C_cts, 0)

        sys_dis = control.c2d(sys_cts, model_disc)

        self.A = sys_dis.A
        self.B = sys_dis.B
        self.C = sys_dis.C

    def model_step(self, U: np.array, x0: np.array) -> np.array:
        """Advance system one model step.

        Parameters:
        -----------
        U: np.array
            Control for model step, dims (Nu, 1)
        x0: np.array
            initial condition for model step, dims (4, 1)

        Returns:
        ----------
        x0: np.array
            system state advance by one model step, dims (4,1)
        """
        x0 = self.A @ x0 + self.B @ U.reshape(1, 1)
        return x0

    def control_step(self, U: np.array, x0: np.array) -> np.array:
        """Advance system one control step.

        Parameters:
        ----------
        U: np.array
            control for control model step, dims (Nu, 1)
        x0: np.array
            initial condition for control step, dims (4,1)

        Returns:
        ----------
        x0: np.array
            system state advance by one control step, dims (4,1)
        """
        for small_step in range(int(self.control_disc / self.model_disc)):
            x0 = self.model_step(U, x0)
        return x0

    def simulate(
        self, U: torch.DoubleTensor, x0: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Simulate the system under given control U.

        Parameters:
        -----------
        U: torch.DoubleTensor
            control inputs for simulation, dims (Nu, t_steps)
        x0: torch.DoubleTensor
            initial condition, dims (4, 1)

        Returns:
        ----------
        X_list: torch.DoubleTensor
            evolution of system under control inputs, dims (4, t_steps)
        """
        t_steps = U.size(1)
        X_list = np.empty((4, t_steps))
        U = U.detach().numpy()
        x0 = x0.detach().numpy()
        for step in range(t_steps):

            x0 = self.control_step(U[:, step], x0)
            X_list[:, step] = x0.flatten()
        X_list = torch.tensor(X_list, dtype=torch.float64)
        return X_list