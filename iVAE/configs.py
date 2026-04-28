"""
    Mingyu Kang
    kangmingyu@ustc.edu.cn
    University of Science and Technology of China
"""

import sys

sys.path.append('../../NICAs')

import os
import shutil
import numpy as np

# data
n_size = 4
n_steps = 2000
data_type = 'MLS'

# NNs
U_size = 5
latent_size = n_size   ## keep the dimension of z equals to that of the mixed signals
k = 2
L = 512
mixL = 5

encode_dict = {"ff_layers": [n_size + U_size, 128, 128, 128, latent_size],
               "conv_flag": False}

decode_dict = {"ff_layers": [latent_size, 128, 128, 128, n_size],
               "conv_flag": False}

prior_dict = {"ff_layers": [U_size, 128, 128, 128, latent_size],
              "conv_flag": False}

# train
epochs = 5000
batch_size = 64
Lr_init = 1e-3
Lr_end = 1e-3

Loss_type = "L2"
noise_var_decoder = 1 / 100
Beta_VAE = 1

gamma_scheduler = 10 ** (np.log10(Lr_end / Lr_init) / epochs)

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
