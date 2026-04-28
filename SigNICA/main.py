import configs as cfg
from data import Data

from SigIC_train import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    data = Data(cfg.n_steps, cfg.n_size, cfg.data_type)
    X = data.data.X
    Y = data.data.Y.to(device)

    # model
    invertible_f = RealNVP(cfg.n_size, cfg.n_mask, cfg.n_hidden, cfg.n_layers, device)
    invertible_f = invertible_f.to(device)

    # optimizer
    opt = torch.optim.Adam(invertible_f.parameters(), lr=cfg.learning_rate)

    # train
    for i in range(cfg.n_epoch):
        _X = invertible_f(Y)

        phi_loss = Phi_SigIC(_X, cfg.order, cfg.batchlen)

        phi_loss.backward()
        opt.step()

        if i % 100 == 0:
            # show the cost
            print('Epoch: {}, Phi_SigIC in Eq. (96): {:.4f}'.format(i, phi_loss.item()))

            # save model
            torch.save(invertible_f.state_dict(), cfg.train_dir + 'model.pth')
