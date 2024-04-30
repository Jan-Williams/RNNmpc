import numpy as np
import torch
from torch import nn


class LinearForecaster(nn.Module):
    """DMDc based linear model for forecasting.

    Attributes:
    -----------
    Nu: int
        dimension of control inputs
    Ns: int
        dimension of sensor inputs
    No: int
        dimension of sensor outputs
    tds: list
        list of time delays to incorporate
    dtype: torch.dtype
        dtype of model
    G: nn.Linear
        forward, linear operator
    in_features: int
        dimension of input
    device: torch.device
        device to which tensors are initialized, default cpu

    Methods:
    ----------
    stack_inputs(U, S)
        Helper function to stack time-delays for inputs.
    fit(U,S,O)
        Train linear module to learn map from previous measurements/control
    forward(s_k)
        forward pass of the linear module
    forecast(U, s_k, U_spin, S_spin)
        Forecast under a given control input.
    set_device(device)
        move model and initializations to device
    """

    def __init__(
        self,
        Nu: int,
        Ns: int,
        No: int,
        tds: list = [-1],
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super(LinearForecaster, self).__init__()

        self.Nu = Nu
        self.Ns = Ns
        self.No = No
        self.tds = tds
        self.in_features = (Nu + Ns) * len(tds)
        self.G = nn.Linear(
            in_features=self.in_features, out_features=No, dtype=dtype, bias=False
        )
        self.dtype = dtype
        self.device = torch.device("cpu")

    def stack_inputs(
        self, U: torch.DoubleTensor, S: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Helper function to stack time-delays for inputs.

        Parameters:
        -----------
        U: torch.DoubleTensor
            control input to stack, (Nu, >max(abs(tds)))
        S: torch.DoubleTensor
            input sensor time history to stack, dims (Ns, > max(abs(tds)))

        Returns:
        -----------
        input_k: torch.DoubleTensor
            input to feed to network, dims (batch, (Ns + Nu) * len(tds))
        """
        input_k = torch.vstack((U, S))
        input_k = input_k[:, self.tds].flatten().reshape(1, -1)
        return input_k

    def fit(
        self,
        U: torch.DoubleTensor,
        S: torch.DoubleTensor,
        O: torch.DoubleTensor,
        beta: float = 0,
    ) -> None:
        """Train linear module to map from previous measurements/control inputs
        to subsequent measurements.

        Parameters:
        -----------
        U: torch.DoubleTensor
            control inputs, dims (Nu, train_len)
        S: torch.DoubleTensor
            sensor inputs, dims (Ns, train_len)
        O: torch.DoubleTensor
            sensor outputs, dims (No, train_len)
        beta: float
            Tikhonov regularization parameter for fitting

        """
        U = U.to(self.device)
        S = S.to(self.device)
        O = O.to(self.device)
        outputs = O.T
        N_samples = U.size(1)
        lags = abs(np.min(self.tds))
        N_retained = N_samples + np.min(self.tds) + 1

        all_in = torch.zeros((N_retained, self.in_features), dtype=self.dtype)
        all_out = torch.zeros((N_retained, self.No), dtype=self.dtype)
        for i in range(N_retained):

            all_in[i] = self.stack_inputs(U[:, i : i + lags], S[:, i : i + lags])

            all_out[i] = outputs[i + lags - 1]

        all_out = all_out.T
        all_in = all_in.T
        lhs = all_in @ all_in.T + beta * torch.eye(self.in_features)
        rhs = all_in @ all_out.T

        self.G_mat = torch.linalg.lstsq(
            lhs.to(dtype=torch.float64), rhs.to(dtype=torch.float64), rcond=None
        )[0].T
        self.G.weight.data = self.G_mat.to(self.device)

    def forward(self, s_k: torch.DoubleTensor) -> torch.DoubleTensor:
        """Forward pass of the linear module.

        Parameters:
        -----------
        s_k: torch.DoubleTensor
            stacked time-delayed inputs, dims (batch_size, (Ns + Nu) * len(tds))

        Returns:
        ---------
        o_k: torch.DoubleTensor
            output sensor measurements, dims (batch_size, No)
        """
        o_k = self.G(s_k)

        return o_k

    def forecast(
        self,
        U: torch.DoubleTensor,
        s_k: torch.DoubleTensor,
        U_spin: torch.DoubleTensor,
        S_spin: torch.DoubleTensor,
    ) -> torch.DoubleTensor:
        """Forecast under a given control input.

        Parameters:
        -----------
        U: torch.DoubleTensor
            control input under which to forecast, dims (Nu, fcast_len)
        s_k: torch.DoubleTensor
            current sensor recordings, dims (Ns, 1)
        U_spin: torch.DoubleTensor
            previous control inputs, dims (Nu, >max(abs(tds)))
        S_spin: torch.DoubleTensor
            previous sensor recordings, dims (Ns, >max(abs(tds)))
        """
        self.eval()
        U = U.to(self.device)
        s_k = s_k.to(self.device)
        U_spin = U_spin.to(self.device)
        S_spin = S_spin.to(self.device)
        fcast_len = U.shape[1]
        U_tot = torch.hstack((U_spin, U))
        S_spin = torch.hstack((S_spin, s_k))
        fcast = torch.zeros((self.No, fcast_len))
        for step in range(fcast_len):
            input_k = self.stack_inputs(U_tot[:, : step + U_spin.shape[1] + 1], S_spin)
            s_k = self(input_k)
            S_spin = torch.hstack((S_spin, s_k.T[: self.Ns, :]))
            fcast[:, step : step + 1] = s_k.T
        return fcast

    def set_device(self, device):
        """Move model and any initializations performed to desired device.

        Parameters:
        ----------
        device: torch.device
            device to move to
        """
        self.to(device)
        self.device = device
