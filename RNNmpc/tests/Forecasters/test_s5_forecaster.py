import pytest
import RNNmpc.Forecasters.s5_forecaster
import torch


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
