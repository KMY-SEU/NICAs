import os

## models: {'TCL', ‘PCL’, 'iVAE', 'SigNICA'}
name_model = 'iVAE'

if name_model == 'TCL':
    from TCL import configs as cfg
    from TCL.data import Data

    import TCL.tcl_eval as tcl_eval
    import TCL.tcl as tcl

    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    ##
    eval_dir = './TCL/storage/model/'
    parmpath = os.path.join(eval_dir, 'parm.pkl')

    # Load trained file -------------------------------------------
    ckpt = tf.compat.v1.train.get_checkpoint_state(eval_dir)
    modelpath = ckpt.model_checkpoint_path

elif name_model == 'PCL':
    from PCL import configs as cfg
    from PCL.data import Data

    from PCL.pcl import feature_MLP

    import torch

    # Load trained file -------------------------------------------
    f_model = feature_MLP(cfg.n_size, cfg.n_layers, maxout_flag=False)
    f_model.load_state_dict(torch.load('./PCL/storage/model/model.pth'))
    f_model.eval()

    print(f_model)

elif name_model == 'iVAE':
    from iVAE import configs as cfg
    from iVAE.data import Data

    from iVAE.ivae import Encoder

    import torch

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained file -------------------------------------------
    encoder_model = Encoder(cfg.latent_size, cfg.U_size, cfg.n_size, cfg.encode_dict)
    encoder_model.to(device)

    encoder_model.load_state_dict(torch.load('./iVAE/storage/model/model.pth'))
    encoder_model.eval()

    print(encoder_model)

elif name_model == 'SigNICA':
    import torch

    from SigNICA import configs as cfg
    from SigNICA.data import Data
    from SigNICA.SigIC_train import RealNVP

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    invertible_f = RealNVP(cfg.n_size, cfg.n_mask, cfg.n_hidden, cfg.n_layers, device)
    invertible_f = invertible_f.to(device)

    invertible_f.load_state_dict(torch.load('./SigNICA/storage/model/model.pth'))
    invertible_f.eval()


else:
    raise ValueError

# evaluate
# Evaluate model ----------------------------------------------
if name_model == 'TCL':
    # load data
    data = Data(cfg.n_steps, cfg.n_size, cfg.data_type)

    Y = data.data.Y
    sensor = Y.T  ## mixed signals

    X = data.data.X  ## source signals, [n_steps, n_size]

    num_segment = data.num_segment

    # demixing
    with tf.Graph().as_default() as g:
        data_holder = tf.compat.v1.placeholder(tf.float32, shape=[None, sensor.shape[0]], name='data')
        label_holder = tf.compat.v1.placeholder(tf.int32, shape=[None], name='label')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, feats = tcl.inference(data_holder, cfg.list_hidden_nodes, num_class=num_segment)

        # Calculate predictions.
        top_value, preds = tf.nn.top_k(logits, k=1, name='preds')

        # Restore the moving averaged version of the learned variables for eval.
        variable_averages = tf.compat.v1.train.ExponentialMovingAverage(cfg.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.compat.v1.train.Saver(variables_to_restore)

        with tf.compat.v1.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)

            tensor_val = tcl_eval.get_tensor(sensor, [preds, feats], sess, data_holder, batch=256)
            pred_val = tensor_val[0].reshape(-1)
            feat_val = tensor_val[1]

    _X = feat_val
    print('TCL tensor ==', _X)

elif name_model == 'PCL':

    # load data
    data = Data(cfg.n_steps, cfg.n_size, cfg.data_type)

    Y = data.mixed_samples  ## mixed signals, [n_steps, n_size]
    X = data.samples  ## source signals, [n_steps, n_size]

    with torch.no_grad():

        _X = f_model(Y)

        print('tensor', _X.shape)

elif name_model == 'iVAE':

    # load data
    data_sampler = Data(cfg.n_steps, cfg.n_size, cfg.data_type, cfg.U_size, cfg.batch_size)
    data_iterator = iter(data_sampler)

    Y = data_sampler.mixed_data

    # evaluate
    _X = []
    for data in data_iterator:
        ##
        Xdata, labels = data

        Xdata = Xdata.to(device)
        labels = labels.to(device)

        ##
        mu_z_encoder, var_z_encoder = encoder_model(Xdata, labels)
        _X += [mu_z_encoder]

    _X = torch.vstack(_X)

    print('iVAE tensor', _X.shape)

elif name_model == 'SigNICA':

    # load data
    data = Data(cfg.n_steps, cfg.n_size, cfg.data_type)
    X = data.data.X
    Y = data.data.Y.to(device)

    with torch.no_grad():

        _X = invertible_f(Y)

        print('SigNICA tensor', _X.shape)

else:
    raise ValueError

print('tensor', Y)
