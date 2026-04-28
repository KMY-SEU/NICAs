import configs as cfg
from data import Data
from pcl_train import train

if __name__ == '__main__':
    # load data
    data = Data(cfg.n_steps, cfg.n_size, cfg.data_type)

    # model
    train(n_size=cfg.n_size,
          n_layers=cfg.n_layers,
          data=data,
          batch_size=cfg.batch_size,
          train_dir=cfg.train_dir,
          maxout_flag=cfg.maxout_flag)
