import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy


def fit_rnn(
    model: nn.Module,
    U: torch.DoubleTensor,
    S: torch.DoubleTensor,
    O: torch.DoubleTensor,
    lags: int,
    num_epochs: int = 1000,
    lr: float = 1e-3,
    patience: int = 5,
    train_split: float = 0.8,
    model_type: str = "GRU",
) -> None:
    """Function for training recurrent forecasting models.

    model: nn.Module
        instantiated forecasting model to train
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
    inputs = np.vstack((U, S)).T
    outputs = O.T
    N_samples = U.size(1)
    N_retained = N_samples - lags
    train_indices = np.random.choice(
        N_retained, size=int(train_split * N_retained), replace=False
    )
    mask = np.ones(N_retained)
    mask[train_indices] = 0
    valid_indices = np.arange(0, N_retained)[np.where(mask != 0)[0]]

    all_in = np.zeros((N_retained, lags, model.Ns + model.Nu))
    all_out = np.zeros((N_retained, model.No))

    for i in range(N_retained):
        all_in[i] = inputs[i : i + lags, :]
        all_out[i] = outputs[i + lags - 1]

    train_data_in = torch.tensor(all_in[train_indices], dtype=torch.float64)
    valid_data_in = torch.tensor(all_in[valid_indices], dtype=torch.float64)
    train_data_out = torch.tensor(all_out[train_indices, :], dtype=torch.float64)
    valid_data_out = torch.tensor(all_out[valid_indices, :], dtype=torch.float64)
    train_dataset = SeriesDataset(train_data_in, train_data_out)
    valid_dataset = SeriesDataset(valid_data_in, valid_data_out)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    patience_counter = 0
    loss_list = []
    best_params = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        for _, data in enumerate(train_loader):

            if model_type == "GRU":
                r_0 = torch.zeros(
                    (1, data[0].size(0), model.Nr), dtype=torch.float64
                ).to(model.device)
                outputs = model(data[0].to(model.device), r_0)
            if model_type == "LSTM":
                r_0 = torch.zeros(
                    (1, data[0].size(0), model.Nr), dtype=torch.float64
                ).to(model.device)
                c_0 = torch.zeros(
                    (1, data[0].size(0), model.Nr), dtype=torch.float64
                ).to(model.device)
                outputs = model(data[0].to(model.device), r_0, c_0)
            optimizer.zero_grad()

            loss = criterion(outputs, data[1].to(model.device))
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                if model_type == "LSTM":
                    r_0 = torch.zeros(
                        (1, valid_dataset.X.size(0), model.Nr), dtype=torch.float64
                    ).to(model.device)
                    c_0 = torch.zeros(
                        (1, valid_dataset.X.size(0), model.Nr), dtype=torch.float64
                    ).to(model.device)
                    outputs = model(valid_dataset.X.to(model.device), r_0, c_0)
                if model_type == "GRU":
                    r_0 = torch.zeros(
                        (1, valid_dataset.X.size(0), model.Nr), dtype=torch.float64
                    ).to(model.device)
                    outputs = model(valid_dataset.X.to(model.device), r_0)
                loss = criterion(outputs, valid_dataset.Y.to(model.device))
                loss_list.append(loss)
                print("Epoch " + str(epoch) + ": " + str(loss.item()))
                patience_counter += 1
                if loss == torch.tensor(loss_list).min():
                    best_params = deepcopy(model.state_dict())
                    patience_counter = 0
                if patience_counter >= patience:
                    model.load_state_dict(best_params)
                    return None
    model.eval()
    model.load_state_dict(best_params)


def fit_fc(
    model,
    U: torch.DoubleTensor,
    S: torch.DoubleTensor,
    O: torch.DoubleTensor,
    num_epochs: int = 1000,
    lr: float = 1e-3,
    patience: int = 5,
    train_split: float = 0.8,
) -> None:
    """Function for training fully connected forecasting models.

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
    outputs = O.T
    N_samples = U.size(1)
    lags = abs(np.min(model.tds))
    N_retained = N_samples + np.min(model.tds)
    train_indices = np.random.choice(
        N_retained, size=int(train_split * N_retained), replace=False
    )
    mask = np.ones(N_retained)
    mask[train_indices] = 0
    valid_indices = np.arange(0, N_retained)[np.where(mask != 0)[0]]

    all_in = np.zeros((N_retained, model.in_features))
    all_out = np.zeros((N_retained, model.No))

    for i in range(N_retained):
        all_in[i] = model.stack_inputs(U[:, i : i + lags], S[:, i : i + lags])
        all_out[i] = outputs[i + lags - 1]

    train_data_in = torch.tensor(all_in[train_indices], dtype=torch.float64)
    valid_data_in = torch.tensor(all_in[valid_indices], dtype=torch.float64)
    train_data_out = torch.tensor(all_out[train_indices, :], dtype=torch.float64)
    valid_data_out = torch.tensor(all_out[valid_indices, :], dtype=torch.float64)
    train_dataset = SeriesDataset(train_data_in, train_data_out)
    valid_dataset = SeriesDataset(valid_data_in, valid_data_out)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    patience_counter = 0
    loss_list = []
    best_params = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        for _, data in enumerate(train_loader):
            outputs = model(data[0].to(model.device))
            optimizer.zero_grad()

            loss = criterion(outputs, data[1].to(model.device))
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(valid_dataset.X.to(model.device))
                loss = criterion(outputs, valid_dataset.Y.to(model.device))
                loss_list.append(loss)
                print("Epoch " + str(epoch) + ": " + str(loss.item()))
                patience_counter += 1
                if loss == torch.tensor(loss_list).min():
                    best_params = deepcopy(model.state_dict())
                    patience_counter = 0
                if patience_counter >= patience:
                    model.load_state_dict(best_params)
                    return None
    model.load_state_dict(best_params)
    model.eval()
    return None


class SeriesDataset(torch.utils.data.Dataset):
    """Dataset for RNN forecaster training."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len