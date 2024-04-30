import numpy as np
import torch
from torch import nn
from RNNmpc.Forecasters.Training import fit_fc


class FCForecaster(nn.Module):
    """Fully connected model for forecasting given known control inputs.

    Attributes:
    ----------
    Nu: int
        number of control inputs
    Ns: int
        number of sensor inputs
    No: int
        number of sensor outputs
    r_list: list
        list of hidden layer widths
    tds: list
        list of time delay lengths to include
    layer_list: nn.ModuleList
        module list of fully connected layers
    dropout: nn.Dropout
        dropout layer to be used between connected layers
    in_features: int
        dimension of input
    device: torch.device
        device to initialize tensors to, default cpu

    Methods:
    ----------
    stack_inputs(U, S)
        Helper function to stack time-delays for inputs.
    forward(s_k)
        output final approximated outputs from input sequence and initial hidden state
    fit(U, S, O, lags, num_epochs, lr, patience, train_split)
        train the fully-connected forecaster model
    forecast(U, U_spin, S_spin)
        forecast given control inputs and burn in data
    set_device(device)
        move model and initializations to device
    """

    def __init__(
        self,
        Nu: int,
        Ns: int,
        No: int,
        r_list: list,
        tds: list,
        dropout_p: float = 0,
        dtype: torch.dtype = torch.float64,
    ):
        """
        Parameters:
        ----------
        r_list: list
            list of hidden layer widths
        Nu: int
            dimension of control inputs
        Ns: int
            dimension of sensor inputs
        No: int
            dimension of outputs
        tds: list
            time delays to include in model
        dropout_p: float
            probability of dropout during training between linear layers
        dtype: torch.dtype
            dtype for all layers
        """
        super(FCForecaster, self).__init__()
        self.Ns = Ns
        self.Nu = Nu
        self.No = No
        self.r_list = r_list
        self.tds = tds
        self.in_features = Ns * len(tds) + Nu * len(tds)
        self.dtype = dtype
        self.dropout = nn.Dropout(dropout_p)

        self.layer_list = nn.ModuleList()
        self.layer_list.append(
            nn.Linear(in_features=self.in_features, out_features=r_list[0], dtype=dtype)
        )
        for lay in range(len(r_list)):
            self.layer_list.append(nn.Linear(r_list[lay - 1], r_list[lay], dtype=dtype))
        self.layer_list.append(nn.Linear(r_list[-1], self.No, dtype=dtype))
        self.nonlin = nn.functional.elu

        self.device = torch.device("cpu")

    def forward(self, s_k: torch.DoubleTensor) -> torch.DoubleTensor:
        """Perform one-step forecast.

        Parameters:
        ----------
        s_k: torch.DoubleTensor
            stacked, time delay inputs, dims (batch, (Ns + Nu) * len(tds))

        Returns:
        ----------
        s_k: torch.DoubleTensor
            one step forecast of output measurements, dims (batch, No)
        """
        for lay in range(len(self.layer_list) - 1):
            s_k = self.layer_list[lay](s_k)
            s_k = self.dropout(s_k)
            s_k = self.nonlin(s_k)
        s_k = self.layer_list[-1](s_k)
        return s_k

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
        U_spin = U_spin.to(self.device)
        S_spin = torch.hstack((S_spin, s_k)).to(self.device)
        U_spin = torch.hstack((U_spin, U[:, 0:1]))
        s_k = s_k.to(self.device)
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
        num_epochs: int = 500,
        lr: float = 1e-3,
        patience: int = 5,
        train_split: float = 0.8,
    ) -> None:
        """Function for training the network.

        Parameters:
        ----------
        model: nn.Module
            instantiated forecasting model to train
        U: torch.DoubleTensor
            sequence of control inputs, dims (Nu, seq_len)
        S: torch.DoubleTensor
            sequence of input measurements, dims (Ns, seq_len)
        O: torch.DoubleTensor
            sequence of output measurements, dims (No, seq_len)
        num_epochs: int
            max number of epochs for training
        lr: float
            learning rate for Adam optimizer
        patience: int
            patience for early stopping criteria
        train_split: float
            proportion of data to use in training
        """
        fit_fc(self, U, S, O, num_epochs, lr, patience, train_split)

    def set_device(self, device):
        """Move model and any initializations performed to desired device.

        Parameters:
        ----------
        device: torch.device
            device to move to
        """
        self.to(device)
        self.device = device
