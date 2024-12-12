import numpy as np
import torch
from torch import nn


class Reservoir(nn.Module):
    """Reservoir module for use with ESNForecaster.

    This module provies a torch-like API for advancing the reservoir state of an
    ESNForecaster to be used for control. It follows the batch_first convention,
    requiring input shapes to be (batch, seq_len, input_dim).

    Attributes:
    ----------
    Nr: int
        reservoir dimension
    Nu: int
        dimension of control inputs
    Ns: int
        dimension of sensor inputs
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
    A: nn.Linear
        reservoir update layer
    B: nn.Linear
        reservoir input layer
    device: torch.device
        device to initialize hidden states to, default cpu

    Methods:
    ----------
    fill_A()
        builds weights for reservoir update layer
    fill_B()
        builds weights for reservoir input layer
    cell_forward(input, r_0)
        helper function for advancing one step
    forward(input, r_0)
        advances reservoir
    set_device(device)
        move model and initializations to device
    """

    def __init__(
        self,
        Nr: int,
        Nu: int,
        Ns: int,
        rho_sr: float = 0.8,
        rho_A: float = 0.02,
        sigma: float = 0.084,
        alpha: float = 0.6,
        sigma_b: float = 1.6,
        dtype=torch.float64,
    ) -> None:
        """
        Parameters:
        ----------
        Nr: int
            reservoir dimension
        Nu: int
            dimension of control inputs
        Ns: int
            dimension of sensor inputs
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
        """
        super(Reservoir, self).__init__()
        self.Nr = Nr
        self.Nu = Nu
        self.Ns = Ns
        self.rho_sr = rho_sr
        self.rho_A = rho_A
        self.sigma = sigma
        self.alpha = alpha
        self.sigma_b = sigma_b
        self.dtype = dtype

        self.A = nn.Linear(
            in_features=self.Nr, out_features=self.Nr, bias=False, dtype=dtype
        )

        self.B = nn.Linear(
            in_features=self.Ns + self.Nu,
            out_features=self.Nr,
            bias=False,
            dtype=self.dtype,
        )

        self.fill_A()
        self.fill_B()

        for param in self.A.parameters():
            param.requires_grad_(False)

        for param in self.B.parameters():
            param.requires_grad_(False)

        self.device = torch.device("cpu")

    def fill_A(self) -> None:
        """Helper function to instantiate reservoir matrix of ESN."""

        inc_indices = np.random.choice(
            self.Nr * self.Nr,
            (int(self.Nr * self.Nr * self.rho_A),),
            replace=False,
        )
        A = np.zeros(self.Nr * self.Nr)
        A[inc_indices] = np.random.uniform(-1, 1, size=(len(inc_indices)))
        A = A.reshape(self.Nr, self.Nr)

        while_count = 0

        while np.linalg.matrix_rank(A) < self.Nr and while_count < 10:
            inc_indices = np.random.choice(
                self.Nr * self.Nr,
                (int(self.Nr * self.Nr * self.rho_A),),
                replace=False,
            )
            A = np.zeros(self.Nr * self.Nr)
            A[inc_indices] = 1
            A = A.reshape(self.Nr, self.Nr)

        A = A * (self.rho_sr / np.max(np.abs(np.linalg.eigvals(A))))
        self.A.weight.data = torch.tensor(A, dtype=self.dtype)

    def fill_B(self) -> None:
        """Helper function to instantiate input matrix of ESN."""
        B = np.random.uniform(
            -self.sigma, self.sigma, size=(self.Nr, self.Ns + self.Nu)
        )
        self.B.weight.data = torch.tensor(B, dtype=self.dtype)

    def cell_forward(
        self, input: torch.DoubleTensor, r_0: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Advance reservoir one step.

        Parameters:
        ----------
        input: torch.DoubleTensor
            input to advance reservoir, dims (batch, Ns + Nu)
        r_0: torch.DoubleTensor
            current reservoir state, dims (batch, Nr)

        Returns:
        ----------
        r_0: torch.DoubleTensor
            advanced reservoir state, dims (batch, Nr)
        """
        r_0 = (1 - self.alpha) * r_0 + self.alpha * torch.tanh(
            self.A(r_0)
            + self.B(input)
            + self.sigma_b * torch.ones_like(r_0, requires_grad=False)
        )
        return r_0

    def forward(
        self, input: torch.DoubleTensor, r_0: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Advance reservoir state multiple steps.

        Parameters:
        ----------
        input: torch.DoubleTensor
            input to advance reservoir, dims (batch, seq_len, Ns + Nu)
        r_0: torch.DoubleTensor
            current reservoir state, dims (1, batch, Nr)

        Returns:
        ----------
        reservoir: torch.DoubleTensor
            sequence of advanced reservoir states, dims (batch, seq_len, Nr)
        r_0: torch.DoubleTensor
            final reservoir state, dims (1, batch, Nr)
        """

        seq_len = input.size(1)
        batch = input.size(0)
        reservoir = torch.empty(
            (batch, seq_len, self.Nr), dtype=self.dtype, requires_grad=False
        )
        for step in range(seq_len):
            r_0 = self.cell_forward(input[:, step, : self.Ns + self.Nu], r_0[0])
            r_0 = torch.unsqueeze(r_0, 0)
            reservoir[:, step, :] = r_0
        return reservoir, r_0

    def set_device(self, device):
        """Move model and any initializations performed to desired device.

        Parameters:
        ----------
        device: torch.device
            device to move to
        """
        self.to(device)
        self.device = device
