import torch
from torch import nn
from RNNmpc.Forecasters.Training import fit_rnn

class GRUForecaster(nn.Module):
    """GRU model for forecasting given known control inputs.

    GRUForecaster revolves around the use of a pytorch GRU for advancing the
    hidden state, but adds linear output layer (with elu nonlinearities).

    Attributes:
    ----------
    Nr: int
        reservoir dimension
    Nu: int
        dimension of control inputs
    Ns: int
        dimension of sensor inputs
    No: int
        dimension of outputs
    dtype: torch.dtype
        dtype for all layers
    GRU: nn.GRU
        gated recurrent unit of model
    dropout: nn.Dropout
        dropout layer between linear layers
    lin_layer1: nn.Linear
        first layer after GRU
    lin_layer2: nn.Linear
        linear output layer
    nonlin: nn.functional
        nonlinearity for output layers
    device: torch.device
        device to intialize hidden states to, default cpu

    Methods:
    ----------
    forward(input, r_0)
        output final approximated outputs from input sequence and initial hidden state
    output_layers(r_n)
        apply output Linear layers to a hidden state
    fit(U, S, O, lags, num_epochs, lr, patience, train_split)
        train the GRU forecaster model
    forecast(U, U_spin, S_spin, s_k)
        forecast given control inputs and burn in data
    set_device(device)
        move model and initializations to device
    """

    def __init__(
        self,
        Nr: int,
        Nu: int,
        Ns: int,
        No: int,
        dropout_p: float = 0,
        dtype=torch.float64,
    ):
        """
        Parameters:
        ----------
        Nr: int
            reservoir dimension
        Nu: int
            dimension of control inputs
        Ns: int
            dimension of sensor inputs
        No: int
            dimension of outputs
        dropout_p: float
            probability of dropout during training between linear layers
        dtype: torch.dtype
            dtype for all layers
        """
        super(GRUForecaster, self).__init__()
        self.Nr = Nr
        self.No = No
        self.Nu = Nu
        self.Ns = Ns

        self.gru = nn.GRU(self.Ns + self.Nu, self.Nr, dtype=dtype, batch_first=True)
        self.lin_layer1 = nn.Linear(self.Nr, self.Nr, dtype=dtype)
        self.lin_layer2 = nn.Linear(self.Nr, self.No, dtype=dtype)
        self.nonlin = torch.nn.functional.elu
        self.dropout = nn.Dropout(dropout_p)
        self.device = torch.device("cpu")

    def forward(
        self,
        input: torch.DoubleTensor,
        r_0: torch.DoubleTensor,
    ) -> torch.DoubleTensor:
        """Final output approximation given sequence of inputs and initial hidden state.

        Parameters:
        ----------
        input: torch.DoubleTensor
            sequence of stacked control and sensor measurements,
            dims (batch, seq_len, Ns + Nu)
        r_0: torch.DoubleTensor
            initial reservoir state, dims (1, batch, Nr)

        Returns:
        ----------
        output: torch.DoubleTensor
            final approximated outputs, dims (batch, No)
        """

        _, hn = self.gru(input, r_0)
        output = hn[0]
        output = self.lin_layer1(output)
        output = self.dropout(output)
        output = self.nonlin(output)
        output = self.lin_layer2(output)
        return output

    def output_layers(self, r_0: torch.DoubleTensor) -> torch.DoubleTensor:
        """Helper function for computing output given final hidden state

        Parameters:
        ----------
        r_0: torch.DoubleTensor
            hidden state to compute output from, dims (batch, Nr)

        Returns:
        ----------
        output: torch.DoubleTensor
            output from hidden state, dims (batch, Nr)
        """

        output = self.lin_layer1(r_0)
        output = self.nonlin(output)
        output = self.lin_layer2(output)
        return output

    def fit(
        self,
        U: torch.DoubleTensor,
        S: torch.DoubleTensor,
        O: torch.DoubleTensor,
        lags: int,
        num_epochs: int = 500,
        lr: float = 1e-3,
        patience: int = 5,
        train_split: float = 0.8,
    ) -> torch.DoubleTensor:
        """Train GRUforecaster.

        Parameters:
        ----------
        U: torch.DoubleTensor
            sequence of control inputs, dims (Nu, seq_len)
        S: torch.DoubleTensor
            sequence of input measurements, dims (Ns, seq_len)
        O: torch.DoubleTensor
            sequence of output measurements, dims (No, seq_len)
        lags: int
            input sequence length for training samples
        num_epochs: int
            max number of epochs for training
        lr: float
            learning rate for Adam optimizer
        patience: int
            patience for early stopping criteria
        train_split: float
            proportion of data to use in training
        """
        fit_rnn(
            self, U, S, O, lags, num_epochs, lr, patience, train_split, model_type="GRU"
        )
        with torch.no_grad():
            inputs = torch.vstack((U, S))[:, -2 - lags : -2].T.to(self.device)
            inputs = inputs.reshape(1, lags, self.Ns + self.Nu)
            r_0 = torch.zeros((1, 1, self.Nr), dtype=torch.float64).to(self.device)
            _, rn = self.gru(inputs, r_0)
            output = torch.squeeze(rn)
            return output.reshape(-1, 1)

    def forecast(
        self,
        U: torch.DoubleTensor,
        U_spin: torch.DoubleTensor,
        S_spin: torch.DoubleTensor,
        s_k: torch.DoubleTensor,
    ) -> torch.DoubleTensor:
        """Forecast given control input sequence and spinup control and sensor data.

        Parameters:
        ----------
        U: torch.DoubleTensor
            control inputs under which to forecast, dims (Nu, control_len)
        U_spin: torch.DoubleTensor
            spinup control data, dims (Nu, spinup_len)
        S_spin: torch.DoubleTensor
            spinup sensor data, dims (Ns, spinup_len)

        Returns:
        ---------
        fcast: torch.DoubleTensor
            forecast under control input, dims (No, control_len)

        """
        self.eval()
        U = U.to(self.device)
        U_spin = U_spin.to(self.device)
        S_spin = torch.hstack((S_spin, s_k)).to(self.device)
        U_spin = torch.hstack((U_spin, U[:, 0:1]))
        s_k = s_k.to(self.device)
        lags = U_spin.size(1)
        fcast_len = U.size(1)
        O = torch.empty((U.size(1), self.No), requires_grad=False, dtype=torch.float64)
        sens_list = S_spin.T.reshape(1, lags, self.Ns).clone()
        u_list = U_spin.T.reshape(1, lags, self.Nu).clone()

        for step in range(fcast_len - 1):
            r_0 = torch.zeros((1, 1, self.Nr), dtype=torch.float64).to(self.device)
            step_inputs = (
                torch.hstack((u_list[0], sens_list[0]))
                .reshape(1, lags, self.Ns + self.Nu)
                .clone()
            )
            output = self(step_inputs, r_0).clone()
            sens_list[0, :-1, :] = sens_list[0, 1:, :].clone()
            sens_list[0, -1, :] = output[0, : self.Ns].clone()
            u_list[0, :-1, :] = u_list[0, 1:, :].clone()
            u_list[0, -1, :] = U[:, step + 1].clone()
            O[step, :] = sens_list[0, -1].clone()

        r_0 = torch.zeros((1, 1, self.Nr), dtype=torch.float64).to(self.device)
        step_inputs = (
            torch.hstack((u_list[0], sens_list[0]))
            .reshape(1, lags, self.Ns + self.Nu)
            .clone()
        )
        output = self(step_inputs, r_0).clone()
        sens_list[0, :-1, :] = sens_list[0, 1:, :].clone()
        sens_list[0, -1, :] = output[0, : self.Ns].clone()
        O[step + 1, :] = sens_list[0, -1].clone()
        fcast = O.T
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