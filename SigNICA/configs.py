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

# model
n_mask = n_size // 2
n_hidden = 256
n_layers = 7

# train
n_epoch = 200000
batchlen = 20
learning_rate = 1e-6
order = [4] * n_size  ## the maximum order of the cumulants to be used for each dimension

# save & load
train_dir = './storage/model/'  # save directory (Caution!! this folder will be removed at first)

# Prepare save folder -----------------------------------------
if train_dir.find("./storage/") > -1:
    if os.path.exists(train_dir):
        print("delete savefolder: {0:s}...".format(train_dir))
        shutil.rmtree(train_dir)  # Remove folder
    print("make savefolder: {0:s}...".format(train_dir))
    os.makedirs(train_dir)  # Make folder
else:
    assert False, "savefolder looks wrong"
