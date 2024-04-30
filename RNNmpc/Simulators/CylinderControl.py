import numpy as np
import hydrogym.firedrake as hgym
import torch
import importlib


class CylinderControl:
    """Wrapper for hydrogym's flow past a cylinder (rotary).

    Attributes:
    -----------
    model_disc: float
        discretization used for underlying simulation, default 0.0025
    control_disc: float
        discretization used for control, default 0.1
    config: dict
        configuration dict for hydrogym FlowEnv
    env: FlowEnv
        hydrogym FlowEnv

    Methods:
    -----------
    model_step(U)
        advance simulation one sim time step with given control action
    control_step(U)
        advance simulation one control time step with constant given action
    simulate(U)
        simulate the system, keeping observations at the control discretization
    """

    def __init__(
        self, model_disc: float = 0.0025, control_disc: float = 0.1, restart: str = None
    ) -> None:
        """Initialize environment with model and control discretizations.

        Parameters:
        ----------
        model_disc: float
            discretization used for underlying simulation, default 0.0025
        control_disc: float
            discretization used for control, default 0.01
        config: dict
            config dictionary for hydrogym RotaryCylinder
        """
        self.model_disc = model_disc
        self.control_disc = control_disc
        if restart is None:
            config_dict = {
                "flow": hgym.RotaryCylinder,
                "flow_config": {
                    "dt": model_disc,
                    "mesh": "medium",
                },
                "solver": hgym.SemiImplicitBDF,
                "solver_config": {"dt": model_disc},
            }
        else:
            config_dict = {
                "flow": hgym.RotaryCylinder,
                "flow_config": {
                    "restart": restart,
                    "dt": model_disc,
                    "mesh": "medium",
                },
                "solver": hgym.SemiImplicitBDF,
                "solver_config": {"dt": model_disc},
            }
        self.config = config_dict
        self.env = hgym.FlowEnv(self.config)

    def model_step(self, U: torch.DoubleTensor) -> torch.DoubleTensor:
        """Advance system one model step.

        Parameters:
        -----------
        U: torch.DoubleTensor
            Control for model step, dims (Nu, 1)

        Returns:
        ----------
        obs: torch.DoubleTensor
            system state advance by one model step, dims (2,1)
        """
        (lift, drag), _, _, _ = self.env.step(U.detach().numpy().reshape(-1))
        obs = torch.tensor(np.vstack((lift, drag)), dtype=torch.float64)
        return obs

    def control_step(self, U: torch.DoubleTensor) -> torch.DoubleTensor:
        """Advance system one control step.

        Parameters:
        ----------
        U: np.array
            control for control model step, dims (Nu, 1)

        Returns:
        ----------
        obs: torch.DoubleTensor
            system state advance by one control step, dims (2,1)
        """
        for _ in range(int(self.control_disc / self.model_disc)):
            obs = self.model_step(U)
        return obs

    def simulate(self, U: torch.DoubleTensor) -> torch.DoubleTensor:
        """Simulate the system under given control U.

        Parameters:
        -----------
        U: torch.DoubleTensor
            control inputs for simulation, dims (Nu, t_steps)

        Returns:
        ----------
        X_list: torch.DoubleTensor
            evolution of system under control inputs, dims (2, t_steps)
        """
        t_steps = U.size(1)
        X_list = np.empty((2, t_steps))
        for step in range(t_steps):
            obs = self.control_step(U[:, step])
            X_list[:, step] = obs.flatten()
        X_list = torch.tensor(X_list, dtype=torch.float64)
        return X_list
