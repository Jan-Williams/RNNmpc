import numpy as np
import torch
import scipy.integrate

class LorenzControl:
    """Simulation object for Lorenz control system.

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
    """

    def __init__(self, control_disc: float = 0.01) -> None:
        """Initialize stirred tank system with specified control discretization.

        No model discretization is used because we use the variable time-stepper in
        scipy.integrate.solve_ivp.

        Parameters:
        -----------
        control_disc: float
            control discretization
        """
        self.control_disc = control_disc
        self.rtol = 1e-12
        self.atol = 1e-12
        self.default_x0 = torch.tensor(np.array([[1], [1], [1]]), dtype=torch.float64)

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
            self.lorenz_deriv,
            t_span=[0, self.control_disc + 1e-8],
            y0=x0,
            args=(U,),
            rtol=self.rtol,
            atol=self.atol,
            t_eval=np.arange(0, 2 * self.control_disc, self.control_disc),
        )

        return sol.y[:, -1]

    @staticmethod
    def lorenz_deriv(t, x0, u) -> list[float]:
        """Derivative function of lorenz system for scipy.integrate.solve_ivp."""
        sigma = 10
        beta = 8 / 3
        rho = 28
        x1_dot = sigma * (x0[1] - x0[0]) + u
        x2_dot = x0[0] * (rho - x0[2]) - x0[1]
        x3_dot = x0[0] * x0[1] - beta * x0[2]
        return [x1_dot, x2_dot, x3_dot]

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
        X_list = np.empty((3, t_steps))
        for step in range(t_steps):
            x0 = self.control_step(U[:, step], x0)
            X_list[:, step] = x0
        X_list = torch.tensor(X_list, dtype=torch.float64)
        return X_list