import torch
from . import Forecasters

class MPController:
    """LBFGS solver for minimizing MPC objective function.

    Attributes:
    ----------
    forecaster
        surrogate model for forecasting, from RNNmpc.Forecasters
    dev: float
        deviation from reference trajectory penalty
    u_1: float
        control penalty for magnitude of control input
    u_2: float
        control penalty for magnitude of control input derivative
    
    Methods:
    ----------
    objective_fn(U, r_0, s_k, ref_vals, S_spin, U_spin, U_last)
        evaluate MPC objective function
    compute_act(U, r_0, s_k, ref_vals, S_spin, U_spin, U_last)
        compute actuation to minimize objective function
    """

    def __init__(self, forecaster, dev, u_1, u_2) -> None:
        """
        Parameters:
        ----------
        forecaster
            surrogate model for forecasting
        dev: float
            deviation from reference trajectory penalty
        u_1: float
            control penalty for magnitude of control input
        u_2: float
            control penalty for magnitude of control input derivative
        """
        self.forecaster = forecaster
        self.dev = dev
        self.u_1 = u_1
        self.u_2 = u_2

    def objective_fn(
            self,
            U: torch.DoubleTensor,
            ref_vals: torch.DoubleTensor,
            s_k: torch.DoubleTensor,
            U_last: torch.DoubleTensor,
            **kwargs,
    ) -> torch.DoubleTensor:
        """Compute objective function cost.

        Parameters:
        -----------
        U: torch.DoubleTensor
            control actuation during forecast, dims (Nu, fcast_len)
        ref_vals: torch.DoubleTensor
            reference trajectory to track, dims (No, fcast_len)
        s_k: torch.DoubleTensor
            current sensor measuremnets, dims (Ns, 1)
        U_last: torch.DoubleTensor
            last control input, dims (Nu,1)
        **kwargs
            if ESN controller: 
                specify r_k, current reservoir state, dims (Nr, 1)
            else:
                specify U_spin, S_spin
        """

        if isinstance(self.forecaster, Forecasters.ESNForecaster):
            r_k = kwargs['r_k']
            fcast = self.forecaster.forecast(U=U, r_k=r_k, s_k=s_k)
        else:
            U_spin = kwargs['U_spin']
            S_spin = kwargs['S_spin']
            fcast = self.forecaster.forecast(U=U, U_spin=U_spin, S_spin=S_spin, s_k=s_k)
        
        cost = self.dev * torch.linalg.norm(fcast - ref_vals)
        cost += self.u_1 * torch.linalg.norm(U)
        cost += self.u_2 * torch.linalg.norm(torch.diff(U))
        cost += self.u_2 * torch.linalg.norm(U[:,0] - U_last[:, -1])
        print(cost)
        return cost
    
    def compute_act(
        self,
        U: torch.DoubleTensor,
        ref_vals: torch.DoubleTensor,
        s_k: torch.DoubleTensor,
        U_last: torch.DoubleTensor,
        bounds: tuple = None,
        **kwargs,
    ) -> torch.DoubleTensor:
        """Compute control action
        
        Parameters:
        ----------
        U: torch.DoubleTensor
            control actuation during forecast, dims (Nu, fcast_len)
        ref_vals: torch.DoubleTensor
            reference trajectory to track, dims (No, fcast_len)
        s_k: torch.DoubleTensor
            current sensor measuremnets, dims (Ns, 1)
        U_last: torch.DoubleTensor
            last control input, dims (Nu,1)
        **kwargs
            if ESN controller: 
                specify r_k, current reservoir state, dims (Nr, 1)
            else:
                specify U_spin, S_spin

        Returns:
        U: torch.DoubleTensor
            control action to minimize objective fn, dims (Nu, fcast_len)
        """
        lbfgs = torch.optim.LBFGS(
            [U], history_size=15, max_iter=1000, line_search_fn="strong_wolfe",
        )
        if isinstance(self.forecaster, Forecasters.ESNForecaster):
            r_k = kwargs['r_k']
            def closure():
                lbfgs.zero_grad()
                objective = self.objective_fn(U=U, r_k=r_k, s_k=s_k, ref_vals=ref_vals, U_last=U_last)
                objective.backward()
                return objective
        
        else:
            U_spin = kwargs['U_spin']
            S_spin = kwargs['S_spin']

            def closure():
                lbfgs.zero_grad()
                objective = self.objective_fn(
                    U=U,
                    s_k=s_k,
                    S_spin=S_spin,
                    U_spin=U_spin,
                    ref_vals=ref_vals,
                    U_last = U_last,
                )
                objective.backward()
                return objective

        lbfgs.step(closure)

        if bounds is not None:
            with torch.no_grad():
                U = U[:,:].clamp(bounds[0], bounds[1]).clone()
        return U



        # adam = torch.optim.Adam(
        #     [U], lr=0.001
        # )
        # if isinstance(self.forecaster, Forecasters.ESNForecaster):
        #     r_k = kwargs['r_k']

        #     for i in range(100):
        #         adam.zero_grad()
        #         objective = self.objective_fn(U=U, r_k=r_k, s_k=s_k, ref_vals=ref_vals, U_last=U_last)
        #         objective.backward()
        #         adam.step()
        #         print(U)

        # return U