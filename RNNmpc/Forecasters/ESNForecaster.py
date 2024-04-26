import numpy as np
import torch
from torch import nn
from RNNmpc.Forecasters.Reservoir import Reservoir

class ESNForecaster(nn.Module):
    """ESN model for forecasting given known control inputs.

    ESNForecaster revolves around the use of a Reservoir for advancing the
    reservoir state of an ESN, but adds the trainable linear output layer.

    Attributes:
    -----------
    Nr: int
        reservoir dimension
    Nu: int
        dimension of control inputs
    Ns: int
        dimension of sensor inputs
    No: int
        dimension of outputs
    rho_sr: float
        spectral radius of reservoir matrix A
    rho_A: float
        density of reservoir matrix A
    sigma: float
        scaling of input matrix B
    alpha: float
        leak rate parameter
    sigma_b: float
        additive bias
    dtype: torch.dtype
        dtype for all layers
    res: Reservoir
        underlying reservoir state
    C: nn.Linear
        learnable output layer
    device: torch.device
        device to initialize hidden states to, default cpu

    Methods:
    ----------
    forward(input, r_0)
        final approximated outputs given sequence of inputs and initial reservoir state
    fit(U, S, O, spinup, beta)
        function for training the linear output layer given control inputs and
        sensor inputs/outputs
    forecast(U, r_0)
        forecast given sequence of control inputs and initial reservoir state
    set_device(device)
        move model and initializations to device

    """

    def __init__(
        self,
        Nr: int,
        Nu: int,
        Ns: int,
        No: int,
        rho_sr: float = 0.8,
        rho_A: float = 0.02,
        sigma: float = 0.084,
        alpha: float = 0.6,
        sigma_b: float = 1.6,
        dtype=torch.float64,
    ) -> None:
        """
        Parameters:
        -----------
        Nr: int
            reservoir dimension
        Nu: int
            dimension of control inputs
        Ns: int
            dimension of sensor inputs
        No: int
            dimension of outputs
        rho_sr: float
            spectral radius of reservoir matrix A
        rho_A: float
            density of reservoir matrix A
        sigma: float
            scaling of input matrix B
        alpha: float
            leak rate parameter
        sigma_b: float
            additive bias
        dtype: torch.dtype
            dtype for all layer
        """
        super(ESNForecaster, self).__init__()
        self.Nr = Nr
        self.No = No
        self.Nu = Nu
        self.Ns = Ns
        self.res = Reservoir(Nr, Nu, Ns, rho_sr, rho_A, sigma, alpha, sigma_b, dtype)
        self.C = nn.Linear(Nr, No, bias=False, dtype=dtype)
        self.device = torch.device("cpu")

    def forward(
        self, input: torch.DoubleTensor, r_0: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Compute final approximated output given sequence of inputs and
        initial reservoir state.

        Parameters:
        ----------
        input: torch.DoubleTensor
            sequence of stacked control and sensor measurements, dims
            (batch, seq_len, Ns + Nu)
        r_0: torch.DoubleTensor
            initial reservoir state, dims (1, batch, Nr)

        Returns:
        ----------
        output: torch.DoubleTensor
            final approximated outputs, dims (batch, No)
        """

        _, r_n = self.res(input, r_0)
        output = r_n[0]
        output = self.C(output)
        return output

    def fit(self, U, S, O, spinup=500, beta=1e-6) -> torch.DoubleTensor:
        """Train linear output layer.

        Parameters:
        ----------
        U: torch.DoubleTensor
            control input sequence, dims (Nu, seq_len)
        S: torch.DoubleTensor
            sensor input sequence, dims (Ns, seq_len)
        O: torch.DoubleTensor
            sensor output sequence, dims (No, seq_len)
        spinup: int
            burn-in sequence length to discard in training
        beta: float
            Tikhonov regularization parameter for Ridge regression

        Returns:
        ---------
        r_k: torch.DoubleTensor
            final hidden state, dims (Nr, 1)
        """
        input = torch.vstack((U, S)).to(self.device)
        input = input.T.reshape(1, S.size(1), self.res.Ns + self.res.Nu)
        r_0 = torch.zeros(
            (1, 1, self.res.Nr), dtype=torch.float64, requires_grad=False
        ).to(self.device)
        reservoir, _ = self.res(input, r_0)
        reservoir = torch.squeeze(reservoir[0]).T[:, spinup:]
        lhs = reservoir @ reservoir.T + beta * np.eye(self.res.Nr)
        rhs = reservoir @ O[:, spinup:].T
        self.Cmat = torch.linalg.lstsq(
            lhs.to(dtype=torch.float64), rhs.to(dtype=torch.float64), rcond=None
        )[0].T
        self.C.weight.data = self.Cmat.to(self.device)
        self.R_train = reservoir
        r_k = self.R_train[:, -2:-1]
        return r_k

    def forecast(
        self, U: torch.DoubleTensor, r_k: torch.DoubleTensor, s_k: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Forecast given control input sequence and initial reservoir state.

        Parameters:
        ----------
        U: torch.DoubleTensor
            control inputs under which to forecast, dims (Nu, forecast_len)
        r_k: torch.DoubleTensor
            initial reservoir state, dims (Nr, 1)
        s_k: torch.DoubleTensor
            current sensor measurements
        Returns:
        ----------
        fcast: torch.DoubleTensor
            forecast of system, dims (No, forecast_len)
        """
        self.eval()
        U = U.to(self.device)
        r_k = r_k.reshape(1, 1, self.res.Nr).to(self.device)
        O = torch.empty(
            (U.size(1), self.No), requires_grad=False, dtype=torch.float64
        ).to(self.device)
        # o_k = torch.squeeze(self.C(r_k))
        o_k = torch.squeeze(s_k.to(self.device)).reshape(self.Ns)
        for t_step in range(U.size(1)):
            u_k = U[:, t_step]
            input_k = torch.hstack((u_k, o_k[: self.Ns])).reshape(1, 1, -1)
            _, r_k = self.res(input_k, r_k)
            o_k = torch.squeeze(self.C(r_k)).reshape(self.No)
            O[t_step] = torch.squeeze(o_k)
        fcast = O.T
        return fcast

    def spin(
        self, U_spin: torch.DoubleTensor, S_spin: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        U_spin = U_spin.to(self.device)
        S_spin = S_spin.to(self.device)
        t_steps = U_spin.shape[1]
        r_k = torch.zeros((1, 1, self.Nr), dtype=torch.float64).to(self.device)
        for step in range(t_steps):
            u_k = U_spin[:, step : step + 1]
            s_k = S_spin[:, step : step + 1]
            r_k = self.advance(u_k, s_k, r_k)
        return r_k

    def advance(
        self, u_k: torch.DoubleTensor, s_k: torch.DoubleTensor, r_k: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Advance reservoir one-step.

        Parameters:
        ----------
        u_k: torch.DoubleTensor
            control values, dims (Nu, 1)
        s_k: torch.DoubleTensor
            input sensor values, dims (Ns, 1)
        r_k: torch.DoubleTensor
            current reservoir state, dims (Nr, 1)

        Returns:
        ----------
        r_k: torch.DoubleTensor
            next reservoir state
        """
        u_k = u_k.to(self.device)
        s_k = s_k.to(self.device)
        r_k = r_k.to(self.device)
        inputs = torch.vstack((u_k, s_k)).reshape(1, 1, -1)
        r_k = self.res(inputs, r_k.reshape(1, 1, -1))[1].reshape(-1, 1)
        return r_k

    def set_device(self, device):
        """Move model and any initializations performed to desired device.

        Parameters:
        ----------
        device: torch.device
            device to move to
        """
        self.to(device)
        self.device = device