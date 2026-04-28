import pickle
import configs as cfg
from data import Data

# ..tcl
from tcl_train import train

if __name__ == '__main__':
    # load data
    data = Data(cfg.n_steps, cfg.n_size, cfg.data_type)
    X = data.data.X
    Y = data.data.Y
    labels = data.labels
    num_class = len(labels)

    # train
    train(data=X.T,
          label=labels,
          num_class=num_class,
          list_hidden_nodes=cfg.list_hidden_nodes,
          initial_learning_rate=cfg.initial_learning_rate,
          momentum=cfg.momentum,
          max_steps=cfg.max_steps,
          decay_steps=cfg.decay_steps,
          decay_factor=cfg.decay_factor,
          batch_size=cfg.batch_size,
          train_dir=cfg.train_dir,
          checkpoint_steps=cfg.checkpoint_steps,
          moving_average_decay=cfg.moving_average_decay)
