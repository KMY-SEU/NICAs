import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class GaussianLoss(nn.Module):
    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, x, x_recon):

        if isinstance(x_recon, tuple):
            # Learnt a variance on the output
            mu_recon, var_recon = x_recon

        else:
            # No learnt variance on the output
            mu_recon = x_recon
            var_recon = torch.ones_like(x_recon).requires_grad_(False)

        error = x - mu_recon

        B, N = x.size()
        # Assuming diagonalised covariance:
        gauss_log_loss = torch.mul(error.pow(2), 1 / (
                2 * var_recon + 1e-12))  # 2x100 error vector is needed to do normal multiplication
        gauss_log_loss = torch.sum(gauss_log_loss)
        gauss_log_loss += 1 / 2 * torch.sum(torch.log(var_recon + 1e-12))

        # Normalise value
        gauss_log_loss = gauss_log_loss / (B * N)

        return gauss_log_loss


class KL_divergence(nn.Module):
    def __init__(self, std_normal=False):
        super(KL_divergence, self).__init__()

        self.std_normal = std_normal  # A flag to check whether the loss

    def forward(self, mu_0, var_0, mu_1=None, var_1=None):
        if self.std_normal:
            mu_1 = torch.zeros_like(mu_0).requires_grad_(False)
            var_1 = torch.ones_like(var_0).requires_grad_(False)

        # perform everything elementwise and then
        Dkl = var_0 / var_1 + ((mu_1 - mu_0) ** 2) / var_1 - 1 + torch.log(var_1 / var_0)

        Dkl = 0.5 * torch.sum(Dkl, dim=1, keepdim=True)

        return torch.sum(Dkl)  # Unnormalised


class VAE_loss(nn.Module):
    # No ability to learn a variance, variance is controlled by the noise distribution for iVAE
    def __init__(self, loss_name="L2", gamma=1, beta=1, std_normal=False):
        super(VAE_loss, self).__init__()

        self.gamma = gamma
        self.beta = beta

        self.loss_name = loss_name

        if self.loss_name.lower() == "l2":
            self.recon_loss = GaussianLoss()

        elif self.loss_name.lower() == "l1":
            self.recon_loss = nn.L1Loss()

        else:
            print("Unknown loss entered.")
            raise SystemExit

        self.kl_loss = KL_divergence(std_normal)

    def forward(self, x, recon_x, mu_0, var_0, mu_1=None, var_1=None):

        B, N = x.size()

        if isinstance(recon_x,
                      tuple) and self.loss_name.lower() == "l1":  # Check if it is a tuple, will be this by default when it is fed in.
            recon_x = recon_x[0]

        Lrecon = self.recon_loss(x, recon_x)
        Lkl = 1 / (B * N) * self.kl_loss(mu_0, var_0, mu_1,
                                         var_1)  # Normalised with same values reconstruction loss (Pytorch does this automatically unless you specify)

        Ltotal = self.gamma * Lrecon + self.beta * Lkl

        return Ltotal, Lrecon, Lkl


class VAE_trainer(object):
    def __init__(self, VAE_model, VAE_optimiser, VAE_cost, training_iterator, validation_iterator, epochs):
        self.model = VAE_model
        self.optimiser = VAE_optimiser
        self.cost = VAE_cost
        self.train_iterator = iter(training_iterator)
        self.valid_iterator = iter(validation_iterator)
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def optimise(self, real_data, recon_data, mu_z_encoder, var_z_encoder, mu_z_prior, var_z_prior,
                 update=True):  # compute and return loss

        loss, recon, kl = self.cost(real_data, recon_data, mu_z_encoder, var_z_encoder, mu_z_prior, var_z_prior)

        if update:
            loss.backward()
            self.optimiser.step()
            self.model.zero_grad()

        return loss, recon, kl

    def train_model(self):  # train the models

        pbar = tqdm(total=self.epochs, desc="cost at epoch {}: {}".format(0, np.inf))

        cost_train_list = []
        cost_valid_list = []
        max_valid = np.inf

        for i in range(self.epochs):

            cost_train_total = 0
            cost_train_AE = 0
            cost_train_KL = 0
            cnt_train = 0

            cost_valid_total = 0
            cost_valid_AE = 0
            cost_valid_KL = 0
            cnt_valid = 0

            self.model.train()

            if self.train_iterator.random_seed:  # Extracts random samples from the trainer
                print("Random iterator is not implemented.")
                raise SystemExit

            elif not self.train_iterator.random_seed:  # Sequentially loops through data

                for data in self.train_iterator:
                    if isinstance(data, tuple):  # Check to see if the input is a tuple with labels
                        # Separate data
                        Xdata, labels = data

                        # Push to GPU (if necessary)
                        Xdata = Xdata.to(self.device)
                        labels = labels.to(self.device)

                    else:
                        # Push to GPU
                        Xdata = data.to(self.device)
                        labels = None

                    mu_z_prior, var_z_prior = self.model.prior(labels)
                    mu_z_encoder, var_z_encoder = self.model.encoder(Xdata, labels)

                    mu_recon, var_recon = self.model.decoder(mu_z_encoder, var_z_encoder)
                    Xrecon = (mu_recon, var_recon)

                    loss, recon, kl = self.optimise(Xdata, Xrecon, mu_z_encoder, var_z_encoder, mu_z_prior, var_z_prior,
                                                    update=True)

                    cost_train_total += loss.item()
                    cost_train_AE += recon.item()
                    cost_train_KL += kl.item()

                    cnt_train += 1

            # TODO - add in validation iterator component
            with torch.no_grad():
                if self.valid_iterator.random_seed:  # Extracts random samples from the trainer
                    print("Random iterator is not implemented.")
                    raise SystemExit

                elif not self.valid_iterator.random_seed:  # Sequentially loops through data
                    for data in self.valid_iterator:

                        if isinstance(data, tuple):

                            # Separate data
                            Xdata, labels = data

                            # Push to GPU (if necessary)
                            Xdata = Xdata.to(self.device)
                            labels = labels.to(self.device)

                        else:
                            Xdata = data.to(self.device)
                            labels = None

                        mu_z_prior, var_z_prior = self.model.prior(labels)
                        mu_z_encoder, var_z_encoder = self.model.encoder(Xdata, labels)

                        mu_recon, var_recon = self.model.decoder(mu_z_encoder, var_z_encoder)
                        Xrecon = (mu_recon, var_recon)

                        loss, recon, kl = self.optimise(Xdata, Xrecon, mu_z_encoder, var_z_encoder, mu_z_prior,
                                                        var_z_prior, update=False)

                        cost_valid_total += loss.item()
                        cost_valid_AE += recon.item()
                        cost_valid_KL += kl.item()

                        cnt_valid += 1

            cost_train_list.append([np.round(cost_train_total / cnt_train, 5), np.round(cost_train_AE / cnt_train, 5),
                                    np.round(cost_train_KL / cnt_train, 5)])
            cost_valid_list.append([np.round(cost_valid_total / cnt_valid, 5), np.round(cost_valid_AE / cnt_valid, 5),
                                    np.round(cost_valid_KL / cnt_valid, 5)])

            if cost_valid_list[-1][0] < max_valid:
                max_valid = cost_valid_list[-1][0]  # Update to be the new minimum
                self.optimal_state_dict = self.model.state_dict()  # Save the optimal state dict
                self.index_min_valid = i

            pbar.set_description(desc="train cost: {}, valid cost: {}".format(cost_train_list[-1], cost_valid_list[-1]))
            pbar.update(1)

        pbar.close()

        self.train_cost = cost_train_list
        self.valid_cost = cost_valid_list

        self.model.eval()

    def plotter(self):

        v1 = np.array(self.train_cost)
        v2 = np.array(self.valid_cost)

        fig, ax = plt.subplots(1, 2)
        ax = ax.flatten()

        for i in ax:
            i.grid()
            i.set_xlabel("Epochs")
            i.set_ylabel("Cost")

        ax[0].set_title("Training curves")
        ax[0].plot(v1[:, 0], label="Total loss")
        ax[0].plot(v1[:, 1], label="Gaussian loss")
        ax[0].plot(v1[:, 2], label="KL divergence loss")
        ax[0].legend()

        ax[1].set_title("Validation curves")
        ax[1].plot(v2[:, 0], label="Total loss")
        ax[1].plot(v2[:, 1], label="Gaussian loss")
        ax[1].plot(v2[:, 2], label="KL divergence loss")
        ax[1].scatter([self.index_min_valid] * 3, v2[self.index_min_valid, :], marker="x", color="r",
                      label="minimum validation index")
        ax[1].legend()

        fig.tight_layout()
        plt.show()
