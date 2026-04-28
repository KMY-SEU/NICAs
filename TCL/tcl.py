"""TCL"""

import tensorflow as tf

# Preserve TF1-style graph/session behavior under TF2
tf.compat.v1.disable_eager_execution()

FLAGS = tf.compat.v1.app.flags.FLAGS
tf.compat.v1.app.flags.DEFINE_string('FILTER_COLLECTION', 'filter',
                                     """filter collection.""")


# =============================================================
# =============================================================
def _variable_init(name, shape, wd, initializer=None, trainable=True,
                   collections=None):
    """Helper to create an initialized Variable with weight decay.

    Args:
        name: name of the variable
        shape: list of ints
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """

    if collections is None:
        collections = [tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]
    else:
        collections = [tf.compat.v1.GraphKeys.GLOBAL_VARIABLES] + collections

    if initializer is None:
        initializer = tf.keras.initializers.VarianceScaling()

    with tf.device('/cpu:0'):
        var = tf.compat.v1.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable,
                                        collections=collections)

    # Weight decay
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.compat.v1.add_to_collection('losses', weight_decay)

    return var


# =============================================================
# =============================================================
def inference(x, list_hidden_nodes, num_class, wd=1e-4, maxout_k=2, MLP_trainable=True, feature_nonlinearity='abs'):
    """Build the model.
        MLP with maxout activation units
    Args:
        x: data holder.
        list_hidden_nodes: number of nodes for each layer. 1D array [num_layer]
        num_class: number of classes of MLR
        wd: (option) parameter of weight decay (not for bias)
        maxout_k: (option) number of affine feature maps
        MLP_trainable: (option) If false, fix MLP layers
        feature_nonlinearity: (option) Nonlinearity of the last hidden layer (feature value)
    Returns:
        logits: logits tensor:
        feat: feature tensor
    """
    print("Building model...")

    # Maxout --------------------------------------------------
    def maxout(y, k):
        input_shape = y.get_shape().as_list()
        ndim = len(input_shape)
        ch = input_shape[-1]

        assert ndim == 4 or ndim == 2
        assert ch is not None and ch % k == 0

        if ndim == 4:
            y = tf.reshape(y, [-1, input_shape[1], input_shape[2], int(ch // k), k])
        else:
            y = tf.reshape(y, [-1, int(ch // k), k])

        y = tf.reduce_max(y, axis=ndim)

        return y

    num_layer = len(list_hidden_nodes)

    # Hidden layers -------------------------------------------
    for ln in range(num_layer):
        with tf.compat.v1.variable_scope('layer' + str(ln + 1)) as scope:
            in_dim = list_hidden_nodes[ln - 1] if ln > 0 else x.get_shape().as_list()[1]
            out_dim = list_hidden_nodes[ln]

            if ln < num_layer - 1:  # Increase number of nodes for maxout
                out_dim = maxout_k * out_dim

            # Inner product
            W = _variable_init('W', [in_dim, out_dim], wd, trainable=MLP_trainable,
                               collections=[FLAGS.FILTER_COLLECTION])
            b = _variable_init('b', [out_dim], 0, tf.compat.v1.constant_initializer(0.0), trainable=MLP_trainable,
                               collections=[FLAGS.FILTER_COLLECTION])
            x = tf.matmul(x, W) + b

            # Nonlinearity
            if ln < num_layer - 1:
                x = maxout(x, maxout_k)
            else:  # The last layer (feature value)
                if feature_nonlinearity == 'abs':
                    x = tf.abs(x)
                else:
                    raise ValueError

            # Add summary
            tf.compat.v1.summary.histogram('layer' + str(ln + 1) + '/activations', x)

    feats = x

    # MLR -----------------------------------------------------
    with tf.compat.v1.variable_scope('MLR') as scope:
        in_dim = list_hidden_nodes[-1]
        out_dim = num_class

        # Inner product
        W = _variable_init('W', [in_dim, out_dim], wd, collections=[FLAGS.FILTER_COLLECTION])
        b = _variable_init('b', [out_dim], 0, tf.compat.v1.constant_initializer(0.0),
                           collections=[FLAGS.FILTER_COLLECTION])
        logits = tf.matmul(x, W) + b

    return logits, feats


# =============================================================
# =============================================================
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    tf.compat.v1.add_to_collection('losses', cross_entropy_mean)

    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acurracy')

    return tf.add_n(tf.compat.v1.get_collection('losses'), name='total_loss'), accuracy


# =============================================================
# =============================================================
def _add_loss_summaries(total_loss):
    loss_averages = tf.compat.v1.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.compat.v1.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.compat.v1.summary.scalar(l.op.name + ' (raw)', l)
        tf.compat.v1.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


# =============================================================
# =============================================================
def train(total_loss,
          accuracy,
          global_step,
          initial_learning_rate,
          momentum,
          decay_steps,
          decay_factor,
          moving_average_decay=0.9999,
          moving_average_collections=tf.compat.v1.trainable_variables()):
    lr = tf.compat.v1.train.exponential_decay(initial_learning_rate,
                                              global_step,
                                              decay_steps,
                                              decay_factor,
                                              staircase=True)

    tf.compat.v1.summary.scalar('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    accu_averages = tf.compat.v1.train.ExponentialMovingAverage(0.9, name='avg_accu')
    accu_averages_op = accu_averages.apply([accuracy])
    tf.compat.v1.summary.scalar(accuracy.op.name + ' (raw)', accuracy)
    tf.compat.v1.summary.scalar(accuracy.op.name, accu_averages.average(accuracy))

    with tf.control_dependencies([loss_averages_op, accu_averages_op]):
        opt = tf.compat.v1.train.MomentumOptimizer(lr, momentum)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.compat.v1.trainable_variables():
        tf.compat.v1.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.compat.v1.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.compat.v1.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.compat.v1.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.compat.v1.no_op(name='train')

    return train_op, lr
