"""
    Mingyu Kang
    kangmingyu@ustc.edu.cn
    University of Science and Technology of China
"""

import sys

sys.path.append('../../NICAs')

import os
import shutil

# data
n_size = 4
n_steps = 2000
data_type = 'MLS'

# NNs
list_hidden_nodes = [40, 40, 40, 40, n_size]

# training
initial_learning_rate = 0.01  # initial learning rate
momentum = 0.9  # momentum parameter of SGD
max_steps = 100000  # number of iterations (mini-batches)
decay_steps = int(5e5)  # decay steps (tf.train.exponential_decay)
decay_factor = 0.1  # decay factor (tf.train.exponential_decay)
batch_size = 512  # mini-batch size
moving_average_decay = 0.999  # moving average decay of variables to be saved
checkpoint_steps = 1e5  # interval to save checkpoint

# save & load
train_dir = './storage/model/'  # save directory (Caution!! this folder will be removed at first)

# Prepare save folder -----------------------------------------
if train_dir.find('./storage/') > -1:
    if os.path.exists(train_dir):
        print('delete savefolder: {0:s}...'.format(train_dir))
        shutil.rmtree(train_dir)  # Remove folder
    print('make savefolder: {0:s}...'.format(train_dir))
    os.makedirs(train_dir)  # Make folder
else:
    assert False, 'savefolder looks wrong'
