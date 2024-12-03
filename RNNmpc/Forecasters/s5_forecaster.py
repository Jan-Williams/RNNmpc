"""Single S5 layer and S5 forecaster classes."""

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
    init_A()
        initializes A matrix
    init_B()
        initializes B matrix
    init_C()
        initializes C matrix
    init_D()
        initializes D matrix
    """

    def __init__(self, n_in: int, n_hidden: int) -> None:
        """Init S5 layer.

        Parameters:
        N_in: int
            input/output dimension of layer
        N_hidden: int
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
        for n in range(self.n_hidden):
            for k in range(self.n_hidden):
                if n == k:
                    hippo[n, k] = -1 / 2
                if n > k:
                    hippo[n, k] = -((n + 1 / 2) ** (1 / 2)) * ((k + 1 / 2) ** (1 / 2))
                if n < k:
                    hippo[n, k] = ((n + 1 / 2) ** (1 / 2)) * ((k + 1 / 2) ** (1 / 2))

        lambda_vec, eig_vecs = torch.linalg.eig(hippo)
        return lambda_vec, eig_vecs

    def init_b(self, eig_vecs):
        """Initalization of continous B parameter."""
        b_mat = torch.normal(
            0, (1 / self.n_in) ** (1 / 2), size=(self.n_hidden, self.n_in)
        )
        b_mat = torch.conj(eig_vecs).T @ b_mat.type(torch.complex128)
        return b_mat

    def init_c(self, eig_vecs):
        """Initialization of continuous C parameter"""
        c_mat = torch.normal(
            0, (1 / self.n_in) ** (1 / 2), size=(self.n_hidden, self.n_hidden)
        )
        c_mat = c_mat.type(torch.complex128) @ eig_vecs
        return c_mat

    def discretize(self, lambda_vec, b_mat, delta):
        """Compute discrete representation from continuous parameters via ZOH.
        """
        identity = torch.ones(lambda_vec.shape[0])
        identity = torch.ones(lambda_vec.shape[0])
        lambda_bar = torch.exp(lambda_vec * delta)
        b_bar = (1 / lambda_vec * (lambda_bar - identity))[..., None] * b_mat
        return lambda_bar, b_bar

    def forward(self, u_input, delta):
        lambda_bar, b_bar = self.discretize(self.lambda_vec, self.b_mat, delta)
        x = torch.zeros((u_input.shape[0], self.n_hidden, 1))
        output = []
        for idx in range(u_input.shape[2]):
            x = lambda_bar * x + b_bar @ u_input[:, idx]
            output.append(x)
        output = torch.stack(output, dim=0)
        return output 