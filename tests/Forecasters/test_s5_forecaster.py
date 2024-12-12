"""Testing S5Layer and S5Forecaster."""

import copy
import numpy as np
import pytest
import torch
import RNNmpc.Forecasters.s5_forecaster


def test_lambda_initialization():
    """Test to ensure that the HiPPO initialization is correct."""
    model = RNNmpc.Forecasters.s5_forecaster.S5Layer(3, 3)
    expected = torch.tensor(
        [
            [-0.5, 0.8660254, 1.11803399],
            [-0.8660254, -0.5, 1.93649167],
            [-1.11803399, -1.93649167, -0.5],
        ]
    ).type(torch.complex128)
    obtained = (
        model.eig_vecs
        @ torch.diag(model.lambda_vec)
        @ torch.conj(model.eig_vecs).T
    )
    torch.testing.assert_close(expected, obtained)


def test_forward_exp_decay():
    """Test that under zero input S5 layer yields correct exponential decay."""
    hidden_dim = 10
    input_dim = 3
    model = RNNmpc.Forecasters.s5_forecaster.S5Layer(input_dim, hidden_dim)
    model.c_mat.data = torch.eye(hidden_dim, dtype=torch.complex128)
    batch_size = 1
    seq_len = 200
    x0 = torch.ones((batch_size, hidden_dim), dtype=torch.complex128)
    u_input = torch.zeros(
        (seq_len, batch_size, input_dim), dtype=torch.complex128
    )
    delta = 0.01
    test_output = model.forward(u_input, delta, x0).detach().numpy()
    t = np.arange(1, 201) * delta
    explicit_solns = []
    for idx in range(hidden_dim):
        explicit_solns.append(
            np.exp(model.lambda_vec[idx].detach().numpy() * t)
        )
    np.testing.assert_allclose(
        test_output[:, 0, :], np.array(explicit_solns).T
    )


def test_forward_input():
    """Test to ensure input matrix is applied correctly."""
    hidden_dim = 20
    input_dim = 10
    model = RNNmpc.Forecasters.s5_forecaster.S5Layer(input_dim, hidden_dim)
    model.c_mat.data = torch.eye(hidden_dim, dtype=torch.complex128)
    batch_size = 5
    seq_len = 200
    x0 = torch.zeros((batch_size, hidden_dim), dtype=torch.complex128)
    u_input = torch.ones(
        (seq_len, batch_size, input_dim), dtype=torch.complex128
    )
    output = model.forward(u_input=u_input, delta=0.01, x0=x0)
    lambda_disc, b_disc = model.discretize(
        model.lambda_vec, model.b_mat, delta=0.01
    )
    computed_inputs = output[1:, :, :] - lambda_disc * output[:-1, :, :]
    bu_k = torch.matmul(
        b_disc, torch.ones((input_dim,), dtype=torch.complex128)
    )
    torch.testing.assert_close(
        computed_inputs - bu_k, torch.zeros_like(computed_inputs)
    )


def test_forward_dims():
    """Test to ensure S5 layer produces correct output shape."""
    hidden_dim = 20
    input_dim = 5
    batch_size = 10
    seq_len = 25
    model = RNNmpc.Forecasters.s5_forecaster.S5Layer(input_dim, hidden_dim)
    model.c_mat.data = torch.eye(hidden_dim, dtype=torch.complex128)
    test_input = torch.ones(
        (seq_len, batch_size, input_dim), dtype=torch.complex128
    )
    delta = 0.5
    test_output = model.forward(test_input, delta)
    assert test_output.shape == (seq_len, batch_size, hidden_dim)


def test_layer_input_dims():
    """Test forward of S5Layer fails with incorrect input dims."""
    hidden_dim = 20
    input_dim = 5
    seq_len = 25
    delta = 0.2
    model = RNNmpc.Forecasters.s5_forecaster.S5Layer(input_dim, hidden_dim)
    model.c_mat.data = torch.eye(hidden_dim, dtype=torch.complex128)
    test_input = torch.ones((seq_len, input_dim), dtype=torch.complex128)
    with pytest.raises(ValueError):
        model.forward(test_input, delta)


def test_layer_input_type():
    """Test forward of S5Layer fails with incorrect dtype."""
    hidden_dim = 20
    input_dim = 5
    batch_size = 10
    seq_len = 25
    delta = 0.2
    model = RNNmpc.Forecasters.s5_forecaster.S5Layer(input_dim, hidden_dim)
    model.c_mat.data = torch.eye(hidden_dim, dtype=torch.complex128)
    test_input = torch.ones(
        (seq_len, batch_size, input_dim), dtype=torch.float64
    )
    with pytest.raises(TypeError):
        model.forward(test_input, delta)


def test_smoke_train():
    """Smoke test to ensure that the model trains."""
    n_in = 1
    n_hidden = 54
    num_layers = 4
    fcast_steps = 3
    delta = 0.01
    params = (n_in, n_hidden, num_layers, fcast_steps, delta)
    model = RNNmpc.Forecasters.s5_forecaster.S5Forecaster(params)
    lambda_vec_init = copy.deepcopy(model.layer_list[0].lambda_vec.data)
    ts_data = torch.ones((n_in, 200))
    RNNmpc.Forecasters.s5_forecaster.train_model(
        model, ts_data, lr=0.001, lags=52, num_epochs=2
    )
    lambda_vec_post = copy.deepcopy(model.layer_list[0].lambda_vec.data)
    assert torch.sum(lambda_vec_init - lambda_vec_post) != 0


def test_train_val_split():
    """Ensure dims match for train/val split."""
    n_in = 3
    n_retained = 1000
    lags = 100
    fcast_steps = 10
    n_timeseries = n_retained + fcast_steps + lags + 1
    ts_data = torch.ones((n_in, n_timeseries))
    train_dataset, valid_dataset = (
        RNNmpc.Forecasters.s5_forecaster.train_val_split(
            ts_data, lags, fcast_steps
        )
    )
    assert (
        train_dataset.x_in.shape == (lags, int(n_retained * 0.8), n_in)
        and train_dataset.y_out.shape
        == (fcast_steps, int(n_retained * 0.8), n_in)
        and valid_dataset.x_in.shape == (lags, int(n_retained * 0.2), n_in)
        and valid_dataset.y_out.shape
        == (fcast_steps, int(n_retained * 0.2), n_in)
    )


def test_forecast_dims():
    """Test to ensure forecast output has correct shape."""
    n_in = 1
    n_hidden = 23
    num_layers = 6
    fcast_steps = 2
    delta = 0.01
    params = (n_in, n_hidden, num_layers, fcast_steps, delta)
    model = RNNmpc.Forecasters.s5_forecaster.S5Forecaster(params)
    batch_size = 2
    u_input = torch.ones((200, batch_size, n_in), dtype=torch.complex128)
    output = model.forward(u_input, 0.01)
    assert output.shape == (fcast_steps, batch_size, n_in)


def test_forecast_input_dims():
    """Test to ensure forecast fails with incorrect inputs dims."""
    n_in = 1
    n_hidden = 23
    num_layers = 6
    fcast_steps = 2
    delta = 0.01
    params = (n_in, n_hidden, num_layers, fcast_steps, delta)
    model = RNNmpc.Forecasters.s5_forecaster.S5Forecaster(params)
    u_input = torch.ones((200, n_in), dtype=torch.complex128)
    with pytest.raises(ValueError):
        model.forward(u_input, 2)


def test_forecaster_input_dtype():
    """Test forward of S5Forecaster fails with wrong input dims or dtype."""
    n_in = 34
    n_hidden = 52
    num_layers = 30
    fcast_steps = 2
    batch_size = 7
    delta = 0.3
    params = (n_in, n_hidden, num_layers, fcast_steps, delta)
    model = RNNmpc.Forecasters.s5_forecaster.S5Forecaster(params)
    u_input = torch.ones((200, batch_size, n_in), dtype=torch.float64)
    with pytest.raises(TypeError):
        model.forward(u_input, 2)
