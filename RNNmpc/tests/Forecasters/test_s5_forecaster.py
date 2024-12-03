import pytest
import RNNmpc.Forecasters.s5_forecaster
import torch
import numpy as np


def test_lambda_initialization():
     model = RNNmpc.Forecasters.s5_forecaster.S5Layer(3, 3)
     expected = torch.tensor(
          [
               [-0.5, 0.8660254, 1.11803399],
               [-0.8660254, -0.5, 1.93649167],
               [-1.11803399, -1.93649167, -0.5],
          ]
     ).type(torch.complex128)
     obtained = (
          model.eig_vecs @ torch.diag(model.lambda_vec) @ torch.conj(model.eig_vecs).T
     )
     torch.testing.assert_close(expected, obtained)

def test_forward_exp_decay():
     hidden_dim = 10
     input_dim = 3
     model = RNNmpc.Forecasters.s5_forecaster.S5Layer(input_dim, hidden_dim)
     batch_size = 1
     seq_len = 200
     x0 = torch.ones((batch_size, hidden_dim), dtype=torch.complex128)
     u_input = torch.zeros((seq_len, batch_size, input_dim), dtype=torch.complex128)
     delta = 0.01
     test_output = model.forward(u_input, delta, x0).detach().numpy()
     t = np.arange(1, 201) * delta
     explicit_solns = []
     for idx in range(hidden_dim):
          explicit_solns.append(np.exp(model.lambda_vec[idx].detach().numpy()*t))
     np.testing.assert_allclose(test_output[:,0,:], np.array(explicit_solns).T)
     