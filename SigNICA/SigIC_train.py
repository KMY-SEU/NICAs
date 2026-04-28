import torch
import torch.nn as nn

import SigIC as sigic

'''Invertible Neural Network'''


class CouplingLayer(nn.Module):
    def __init__(self, n_size, n_mask, n_hidden, device):
        super(CouplingLayer, self).__init__()

        self.device = device

        perm = torch.randperm(n_size)
        self.mask = perm[:n_mask]
        self._mask = perm[n_mask:]

        self.fs = nn.Sequential(
            nn.Linear(n_mask, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_size - n_mask),
            nn.Tanh()
        )

        self.ft = nn.Sequential(
            nn.Linear(n_mask, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_size - n_mask)
        )

    def forward(self, y):
        x = torch.zeros(size=y.shape, device=self.device)

        y_1 = y[:, self.mask]
        y_2 = y[:, self._mask]

        x[:, self.mask] = y_1
        x[:, self._mask] = y_2 * torch.exp(self.fs(y_1)) + self.ft(y_1)

        return x


class RealNVP(nn.Module):
    def __init__(self, n_size, n_mask, n_hidden, n_layers, device):
        super(RealNVP, self).__init__()

        self.n_layers = n_layers

        self.nn_list = nn.ModuleList(
            [CouplingLayer(n_size, n_mask, n_hidden, device) for _ in range(n_layers)]
        )

    def forward(self, y):
        for l in range(self.n_layers):
            y = self.nn_list[l](y)

        return y


'''Contrast Function (Signature-Cross-Cumulants)'''


def Phi_SigIC(X_in, mu, batchlen):
    """

    Args:
        X_in: demixed data [n_steps, n_size]
        mu: orders of the signature-cumulants, e.g., [7, 7, 7]
        batchlen: window length

    Returns:

    """

    Phi = sigic.SigCF(X_in, mu, batchlen)

    return Phi
