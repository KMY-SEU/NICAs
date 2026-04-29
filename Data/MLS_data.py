import numpy as np
import os
import time

import torch
import matplotlib.pyplot as plt


class Modulating_Laplacian_Sources:
    def __init__(self, n_steps=400, n_size=4, a=0.05, b=0, rho=0.7, is_tensor=True):
        self.n_steps = n_steps
        self.n_size = n_size
        self.a = a
        self.b = b
        self.rho = rho

        # data: [n, p]
        self.X = self.Laplacian_sampler(self.n_steps, self.n_size, self.a, self.b, self.rho)

        # Mixing
        self.Y = self.apply_nonlinearity_to_source(self.X)

        # to be tensor
        if is_tensor:
            # Convert samples to torch tensor
            self.X = torch.from_numpy(self.X.astype('float32'))
            self.Y = torch.from_numpy(self.Y.astype('float32'))

    def Laplacian_sampler(self, n_steps, n_size, a, b, rho):

        # initial random seed
        # seed = int(time.time()) + os.getpid()
        seed = 12345
        rng = np.random.default_rng(seed)
        z_t0 = rng.normal(0, 0.1, n_size)

        # [T, N]
        sim_sample = np.zeros((n_steps, n_size))

        # initialize x[0]
        sim_sample[0] = z_t0

        # sample
        for i in range(1, n_steps, 1):

            lap_sample = []

            for j in range(n_size):
                lap_sample.append(rng.laplace(loc=rho * sim_sample[i - 1, j], scale=1, size=1)[0])

            lap_sample = np.array(lap_sample)  # 0.1 * np.random.randn(n_size)#

            sim_sample[i, :] = b + a * sim_sample[i - 1, :] + lap_sample

        return sim_sample

    def apply_nonlinearity_to_source(self, X):

        Y = np.zeros_like(X)

        for p in range(self.n_size):
            linear_mix = np.sum(X, axis=1) - X[:, p]
            Y[:, p] = -(1 / (1 + np.exp(- X[:, p]))) + linear_mix

        return Y

    # def plotting(self, X, Y):
    #     T, N = X.shape
    #
    #     # 创建图像
    #     fig, axes = plt.subplots(nrows=N, ncols=2, figsize=(10, 8), sharex=True)
    #     # fig.suptitle("X and Y Signals Over Time", fontsize=16)
    #
    #     for i in range(N):
    #         axes[i, 0].plot(X[:, i], color='tab:blue')
    #         axes[i, 0].set_ylabel(f'#{i+1}')
    #         axes[i, 0].set_xlim(left=0, right=T)
    #         axes[i, 0].set_ylim(bottom=-8, top=8)
    #
    #         if i == N - 1:
    #             axes[i, 0].set_xlabel(r'Time $t$')
    #
    #         axes[i, 1].plot(Y[:, i], color='tab:orange')
    #         axes[i, 1].set_xlim(left=0, right=T)
    #         axes[i, 1].set_ylim(bottom=-18, top=18)
    #
    #         if i == N - 1:
    #             axes[i, 1].set_xlabel(r'Time $t$')
    #
    #     # 加标签
    #     axes[0, 0].set_title(r'Source signals $\mathbf{X}_t$')
    #     axes[0, 1].set_title(r'Mixed signals $\mathbf{Y}_t$')
    #
    #     plt.tight_layout(rect=[0, 0, 1, 0.97])  # 留出 suptitle 空间
    #     plt.show()
