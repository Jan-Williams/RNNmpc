"""Single S5 layer and S5 forecaster classes plus utils functions."""

import numpy as np
import torch


class S5Layer(torch.nn.Module):
    """Single S5 layer.

    Attributes:
    ----------
    n_in: int
        input dimension of S5 layer
    n_hidden: int
        hidden dimension of S5 layer

    Methods:
    ---------
    init_lambda()
        initializes diagonal state matrix, lambda
    init_b()
        initializes input matrix, b_mat
    init_c()
        initializes measurement matric, c_mat
    discretize(lambda_vec, b_mat, delta)
        compute discrete matrices lambda_bar and b_bar via zero order hold
    """

    def __init__(self, n_in: int, n_hidden: int) -> None:
        """Init S5 layer.

        Parameters:
        -----------
        n_in: int
            input/output dimension of layer
        n_hidden: int
            hidden dimension of layer
        """
        super().__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden

        lambda_vec, eig_vecs = self.init_lambda()
        b_mat = self.init_b(eig_vecs)
        c_mat = self.init_c(eig_vecs)
        self.lambda_vec = torch.nn.Parameter(lambda_vec)
        self.eig_vecs = eig_vecs
        self.b_mat = torch.nn.Parameter(b_mat)
        self.c_mat = torch.nn.Parameter(c_mat)

    def init_lambda(self):
        """Initialization of lambda parameter and associated eigenvectors."""
        hippo = torch.zeros((self.n_hidden, self.n_hidden), dtype=torch.complex128)
        for row in range(self.n_hidden):
            for col in range(self.n_hidden):
                if row == col:
                    hippo[row, col] = -1 / 2
                if row > col:
                    hippo[row, col] = -((row + 1 / 2) ** (1 / 2)) * (
                        (col + 1 / 2) ** (1 / 2)
                    )
                if row < col:
                    hippo[row, col] = ((row + 1 / 2) ** (1 / 2)) * (
                        (col + 1 / 2) ** (1 / 2)
                    )

        lambda_vec, eig_vecs = torch.linalg.eig(hippo)
        return lambda_vec, eig_vecs

    def init_b(self, eig_vecs: torch.Tensor):
        """Initalization of continous input matrix, B."""
        b_mat = torch.normal(
            0, (1 / self.n_in) ** (1 / 2), size=(self.n_hidden, self.n_in)
        )
        b_mat = torch.conj(eig_vecs).T @ b_mat.type(torch.complex128)
        return b_mat

    def init_c(self, eig_vecs: torch.Tensor):
        """Initialization of continuous measurement matrix, C."""
        c_mat = torch.normal(
            0, (1 / self.n_in) ** (1 / 2), size=(self.n_hidden, self.n_hidden)
        )
        c_mat = c_mat.type(torch.complex128) @ eig_vecs
        return c_mat

    def discretize(
        self,
        lambda_vec: torch.nn.parameter.Parameter,
        b_mat: torch.nn.parameter.Parameter,
        delta: float,
    ):
        """Compute discrete representation from continuous parameters via ZOH."""
        identity = torch.ones(lambda_vec.shape[0])
        lambda_bar = torch.exp(lambda_vec * delta)
        b_bar = (1 / lambda_vec * (lambda_bar - identity))[..., None] * b_mat
        return lambda_bar, b_bar

    def forward(self, u_input: torch.Tensor, delta: float, x0=None):
        """Compute output of S5 layer (no nonlinearity).

        Parameters:
        -----------
        u_input: torch.cdouble
            measurement history, shape (seq_len, batch_size, n_in)
        delta: float
            discretization of zero order hold
        x0: torch.cdouble
            initial latent state, defaults to zeros, shape (batch_size, n_hidden)
        """
        if x0 is None:
            x = torch.zeros((u_input.shape[1], self.n_hidden), dtype=torch.complex128)
        else:
            x = x0
        if u_input.dtype is not torch.complex128:
            raise TypeError("u_input must be complex double tensor.")
        if len(u_input.shape) != 3:
            raise ValueError("u_input must have shape (seq_len, batch_size, n_in)")
        if x.dtype is not torch.complex128:
            raise TypeError("x0 must be complex double tensor.")
        if len(x.shape) != 2:
            raise ValueError("x0 must have shape (batch_size, n_hidden)")

        lambda_bar, b_bar = self.discretize(self.lambda_vec, self.b_mat, delta)

        output = []
        for idx in range(u_input.shape[0]):
            x = lambda_bar * x + u_input[idx, :, :] @ b_bar.T
            output.append(x)
        output = torch.stack(output, dim=0)
        output = output @ self.c_mat.T
        return output


class S5Forecaster(torch.nn.Module):
    """Sequence of S5 layers used for forecasting.

    Attributes:
    ----------
    n_in: int
        input dimension
    n_hidden: int
        dimension of S5 layers' hidden states
    num_layers: int
        number of S5 layers in the model
    fcast_steps: int
        number of steps over which to train the forecast
    layer_list: torch.nn.ModuleList
        list containing each S5 layer
    output_layer: torch.nn.Linear
        linear output layer mapping from final sequence of hidden states to forecast
    delta: float
        discretization used for zero order hold
    """

    def __init__(self, params: tuple) -> None:
        """Init S5 forecasting model.

        Parameters:
        -----------
        params: tuple
            (n_in: int, n_hidden: int, num_layers: int, fcast_steps: int, delta: float)
        """
        super().__init__()
        self.n_in = params[0]
        self.n_hidden = params[1]
        self.num_layers = params[2]
        self.fcast_steps = params[3]
        self.delta = params[4]

        self.layer_list = torch.nn.ModuleList()
        self.layer_list.append(S5Layer(self.n_in, self.n_hidden))
        for _ in range(self.num_layers - 1):
            self.layer_list.append(S5Layer(self.n_hidden, self.n_hidden))
        self.output_layer = torch.nn.Linear(
            self.n_hidden, self.n_in, dtype=torch.float64
        )

    def forward(self, u_input: torch.Tensor, delta: float, x0=None):
        """Forward call of S5 forecasting model.

        Parameters:
        ----------
        u_input: torch.cdouble
            measurement history, shape (seq_len, batch_size, n_in)
        delta: float
            discretization of zero order hold
        x0: torch.cdouble
            initial latent state, defaults to zeros, shape (num_layers, batch_size, n_hidden)
        """
        if x0 is None:
            x = torch.zeros(
                (len(self.layer_list), u_input.shape[1], self.n_hidden),
                dtype=torch.complex128,
            )
        else:
            x = x0

        if u_input.dtype is not torch.complex128:
            raise TypeError("u_input must be complex double tensor.")
        if len(u_input.shape) != 3:
            raise ValueError("u_input must have shape (seq_len, batch_size, n_in)")
        if x.dtype is not torch.complex128:
            raise TypeError("latent states x must be complex double tensor.")
        if len(x.shape) != 3:
            raise ValueError(
                "latent states musst have shape (num_layers, batch_size, n_in)"
            )

        for layer, _ in enumerate(self.layer_list):
            u_input = self.layer_list[layer].forward(u_input, delta, x[layer])
            u_input = torch.complex(
                torch.nn.functional.gelu(u_input.real),
                torch.nn.functional.gelu(u_input.imag),
            )

        output = self.output_layer(u_input[-self.fcast_steps :].real)

        return output


def train_model(
    model: S5Forecaster,
    ts_data: torch.Tensor,
    lr: float,
    lags: int,
    num_epochs: int = 100,
):
    """Train S5Forecaster model."""
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.MSELoss()
    train_dataset, valid_dataset = train_val_split(ts_data, lags, model.fcast_steps)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=64
    )

    for epoch in range(num_epochs):
        model.train()
        for _, data in enumerate(train_loader):

            data[0] = torch.swapaxes(data[0], 0, 1)
            data[1] = torch.swapaxes(data[1], 0, 1)

            outputs = model(data[0], model.delta)
            optimizer.zero_grad()

            loss = criterion(outputs, data[1].real)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            print("Epoch: " + str(epoch))
            criterion(model(valid_dataset.x_in, model.delta), valid_dataset.y_out.real)
            print(
                "Val. Loss: "
                + str(
                    criterion(
                        model(valid_dataset.x_in, model.delta), valid_dataset.y_out.real
                    ).item()
                )
            )


def train_val_split(ts_data: torch.Tensor, lags: int, fcast_steps: int):
    """Perform training/validation split."""
    n_samples = ts_data.size(1)
    n_retained = n_samples - lags - fcast_steps - 1
    train_indices = np.random.choice(
        n_retained, size=int(0.8 * n_retained), replace=False
    )
    mask = np.ones(n_retained)
    mask[train_indices] = 0
    valid_indices = np.arange(0, n_retained)[np.where(mask != 0)[0]]
    all_in = np.zeros((lags, n_retained, ts_data.size(0)))
    all_out = np.zeros((fcast_steps, n_retained, ts_data.size(0)))

    for idx in range(n_retained):
        all_in[:, idx, :] = ts_data[:, idx : idx + lags].T
        all_out[:, idx, :] = ts_data[:, idx + lags : idx + lags + fcast_steps].T

    train_dataset = SeriesDataset(
        torch.tensor(all_in[:, train_indices, :], dtype=torch.complex128),
        torch.tensor(all_out[:, train_indices, :], dtype=torch.complex128),
    )
    valid_dataset = SeriesDataset(
        torch.tensor(all_in[:, valid_indices, :], dtype=torch.complex128),
        torch.tensor(all_out[:, valid_indices, :], dtype=torch.complex128),
    )
    return train_dataset, valid_dataset


def train_step(
    model: S5Forecaster,
    criterion: torch.nn.MSELoss,
    optimizer: torch.optim.Adam,
    train_loader: torch.utils.data.DataLoader,
):
    """Single training step."""
    model.train()
    for _, data in enumerate(train_loader):

        data[0] = torch.swapaxes(data[0], 0, 1)
        data[1] = torch.swapaxes(data[1], 0, 1)

        outputs = model(data[0], model.delta)
        optimizer.zero_grad()

        loss = criterion(outputs, data[1].real)
        loss.backward()
        optimizer.step()


class SeriesDataset(torch.utils.data.Dataset):
    """Dataset for RNN forecaster training."""

    def __init__(self, x_in: torch.Tensor, y_out: torch.Tensor):
        self.x_in = x_in
        self.y_out = y_out
        self.len = x_in.shape[1]

    def __getitem__(self, index: int):
        return self.x_in[:, index, :], self.y_out[:, index, :]

    def __len__(self):
        return self.len
