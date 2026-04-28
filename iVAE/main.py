import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import configs as cfg
from data import Data

from ivae import Encoder, Decoder, ConditionalPrior
from ivae_train import VAE_trainer, VAE_loss

if __name__ == '__main__':

    def MCC_estimation(real_sources, estimated_sources, PLOT=True, save_fig=False, save_label=None):
        # real_sources 2D ndarray [n_points, n_comp]

        assert (real_sources.shape[0] == estimated_sources.shape[
            0]), "\n*****The number of points in the sources are not the same size. Please check.****\n"

        n_points, n_comp = real_sources.shape
        _, n_source = estimated_sources.shape

        corr_mat = np.zeros((n_comp, n_source))

        for i in range(n_comp):
            s1 = real_sources[:, i]

            for j in range(n_source):
                s2 = estimated_sources[:, j]

                corr_mat[i, j] = np.abs(np.corrcoef(s1, s2))[0, 1]  # The correlation coefficient

        # Perform LSAP to obtain optimal source configuration
        row_ind, col_ind = linear_sum_assignment(corr_mat, maximize=True)  # col_ind is what you want

        MCC = np.mean(corr_mat[row_ind, col_ind])

        if PLOT:

            plt.figure()
            plt.title("Correlation matrix; MCC: {:.2f}%".format(MCC * 100))
            plt.imshow(corr_mat)
            plt.colorbar()
            plt.xticks(range(0, cfg.n_size + 1, 1))
            plt.yticks(range(0, cfg.n_size + 1, 1))
            plt.tight_layout()
            plt.xlabel("Estimated sources")
            plt.ylabel("Recovered sources")

            if save_fig:
                plt.tight_layout()
                plt.savefig("./tmp_figures/correlation_matrix_" + save_label + ".png")
            plt.show()

            print("\n")
            col_size = min(4, n_comp)  # Assume that we have at least 4, this is sub-optimal but will work generally.
            row_size = n_comp // col_size + 1

            fig, ax = plt.subplots(n_comp, 1, figsize=(12, 8))
            ax = ax.flatten()
            for i, j in zip(range(n_comp), col_ind):
                ax[i].set_title("Signal {}; correlation: {:.2f}%".format(i, corr_mat[i, j] * 100))
                ax[i].plot(real_sources[:, i])
                ax[i].plot(estimated_sources[:, j])

            plt.tight_layout()

            if save_fig:
                plt.tight_layout()
                plt.savefig("./tmp_figures/correlated_signals_" + save_label + ".png")
            plt.show()

            print("\n")
            fig, ax = plt.subplots(n_comp, 1, figsize=(12, 8))
            ax = ax.flatten()

            for i, j in zip(range(n_comp), col_ind):
                ax[i].set_title("Signal {}".format(i))
                ax[i].scatter(real_sources[:, i], estimated_sources[:, j])

            plt.tight_layout()

            if save_fig:
                plt.tight_layout()
                plt.savefig("./tmp_figures/correlated_signals_scatter_" + save_label + ".png")
            plt.show()

        return MCC, corr_mat[row_ind, col_ind]


    # cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    data_sampler = Data(cfg.n_steps, cfg.n_size, cfg.data_type, cfg.U_size, cfg.batch_size)
    data_iterator = iter(data_sampler)

    # model
    encoder_model = Encoder(cfg.latent_size, cfg.U_size, cfg.n_size, cfg.encode_dict)
    decoder_model = Decoder(cfg.latent_size, cfg.U_size, cfg.n_size, cfg.decode_dict, var_flag=False)
    prior_model = ConditionalPrior(cfg.latent_size, cfg.U_size, cfg.n_size, cfg.prior_dict, continuous_prior=False)

    # Push models to device
    encoder_model.to(device)
    decoder_model.to(device)
    prior_model.to(device)

    print(encoder_model)
    print(decoder_model)
    print(prior_model)
    # print(dir(prior_model))

    # train
    loss = VAE_loss(loss_name=cfg.Loss_type, gamma=1 / cfg.noise_var_decoder, beta=cfg.Beta_VAE, std_normal=False)

    ls = list(encoder_model.parameters()) + list(decoder_model.parameters()) + list(prior_model.parameters())
    optim = torch.optim.Adam(ls, lr=cfg.Lr_init)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=cfg.gamma_scheduler)

    total_cost_list = []
    recon_cost_list = []
    kl_cost_list = []
    MCC = []

    for i in range(cfg.epochs):

        total_cost = 0
        recon_cost = 0
        kl_cost = 0
        N = 0

        for data in data_iterator:
            ##
            Xdata, labels = data

            Xdata = Xdata.to(device)
            labels = labels.to(device)

            ##
            mu_z_encoder, var_z_encoder = encoder_model(Xdata, labels)

            mu_z_prior, var_z_prior = prior_model(labels)  # labels).to(device)) #

            Xrecon = decoder_model(mu_z_encoder, var_z_encoder)

            loss_val, recon_loss, kl_loss = loss(Xdata, Xrecon, mu_z_encoder, var_z_encoder, mu_z_prior, var_z_prior)

            loss_val.backward()
            optim.step()
            optim.zero_grad()

            total_cost += loss_val.item()
            recon_cost += recon_loss.item()
            kl_cost += kl_loss.item()
            N += 1

        # Update scheduler
        scheduler.step()

        total_cost_list.append(total_cost / N)
        recon_cost_list.append(recon_cost / N)
        kl_cost_list.append(kl_cost / N)

        # checkpoint
        if i % 100 == 0:
            ###############
            # MCC for inputs, sub-optimal implementation, should rather loop through the elements in batches
            ###############

            with torch.no_grad():
                z, var = encoder_model(data_sampler.mixed_data.to(device), data_sampler.sample_labels.to(device))

                MCC_iter, _ = MCC_estimation(data_sampler.data.cpu().numpy(), z.cpu().numpy(), PLOT=False)
                MCC.append(MCC_iter)

            # show the cost
            print("cost[MSE, KL]/MCC at epoch {}: [{:.4f}, {:.4f}]/{:.2f}%".format(i, recon_cost_list[-1],
                                                                                   kl_cost_list[-1],
                                                                                   MCC_iter * 100))

            # save model
            torch.save(encoder_model.state_dict(), cfg.train_dir + 'model.pth')
