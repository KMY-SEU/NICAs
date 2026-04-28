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
n_steps = 2 ** 14
data_type = 'MLS'

# NNs
n_layers = 3
maxout_flag = False  # no using

# train
batch_size = 2 ** 8

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
