import locale
import functools
import tensorflow as tf
import lib.ops.Conv1D, lib.ops.Linear, lib.ops.BatchNorm, lib.ops.LSTM


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True, pooling_size=2):
    output = lib.ops.Conv1D.conv1d(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.nn.pool(output, [pooling_size], 'AVG', 'SAME', strides=[pooling_size])
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True, pooling_size=2):
    output = inputs
    output = tf.nn.pool(output, [pooling_size], 'AVG', 'SAME', strides=[pooling_size])
    output = lib.ops.Conv1D.conv1d(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def normalize(name, inputs, is_training_ph, use_bn=True):
    with tf.variable_scope(name):
        if use_bn:
            return tf.contrib.layers.batch_norm(inputs, fused=True, decay=0.9, is_training=is_training_ph,
                                                scope='BN', reuse=tf.get_variable_scope().reuse,
                                                updates_collections=None)
        else:
            return inputs


def resblock(name, input_dim, output_dim, filter_size, inputs, resample, is_training_ph, r=1.0,
             use_bn=True, stride=2):
    if resample == 'down':
        conv1 = functools.partial(lib.ops.Conv1D.conv1d, input_dim=input_dim, output_dim=input_dim)
        conv2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim, pooling_size=stride)
        shortcut_func = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim, pooling_size=stride)
    elif resample is None:
        conv1 = functools.partial(lib.ops.Conv1D.conv1d, input_dim=input_dim, output_dim=output_dim)
        conv2 = functools.partial(lib.ops.Conv1D.conv1d, input_dim=output_dim, output_dim=output_dim)
        shortcut_func = functools.partial(lib.ops.Conv1D.conv1d, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('Choose between up-sampling and down-sampling!')
    with tf.variable_scope(name):

        if output_dim == input_dim and resample is None:
            shortcut = inputs
        else:
            shortcut = shortcut_func(name='Shortcut', filter_size=1, he_init=False, biases=True, inputs=inputs)

        output = inputs
        output = normalize(name='Norm1', is_training_ph=is_training_ph, inputs=output, use_bn=use_bn)
        output = tf.nn.relu(output)
        output = conv1(name='Conv1', filter_size=filter_size, inputs=output)
        output = normalize(name='Norm2', is_training_ph=is_training_ph, inputs=output, use_bn=use_bn)
        output = tf.nn.relu(output)
        output = conv2(name='Conv2', filter_size=filter_size, inputs=output)

        return r * output + shortcut


def OptimizedResBlockDisc1(inputs, input_dim, output_dim, resample='down', filter_size=3):
    conv1 = functools.partial(lib.ops.Conv1D.conv1d, input_dim=input_dim, output_dim=output_dim)
    if resample == 'down':
        conv2 = functools.partial(ConvMeanPool, input_dim=output_dim, output_dim=output_dim)
        conv_shortcut = MeanPoolConv
    else:
        conv2 = functools.partial(lib.ops.Conv1D.conv1d, input_dim=output_dim, output_dim=output_dim)
        conv_shortcut = lib.ops.Conv1D.conv1d
    shortcut = conv_shortcut('Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False,
                             biases=True, inputs=inputs)

    output = inputs
    output = conv1('Conv1', filter_size=filter_size, inputs=output)
    output = tf.nn.relu(output)
    output = conv2('Conv2', filter_size=filter_size, inputs=output)
    return shortcut + output
