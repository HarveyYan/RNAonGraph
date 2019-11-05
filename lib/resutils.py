import functools
import tensorflow as tf
import lib.ops.Conv1D, lib.ops.Linear, lib.ops.LSTM


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.Conv1D.conv1d(name, input_dim, output_dim, filter_size, inputs,
                                   he_init=he_init, biases=biases)
    output = tf.nn.pool(output, [2], 'AVG', 'SAME', strides=[2])
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.nn.pool(output, [2], 'AVG', 'SAME', strides=[2])
    output = lib.ops.Conv1D.conv1d(name, input_dim, output_dim, filter_size, output,
                                   he_init=he_init, biases=biases)
    return output


def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True, stride=2,
                 use_nearest_neighbor=True):
    output = inputs
    length = output.get_shape().as_list()[1]
    if use_nearest_neighbor:
        output = tf.image.resize_nearest_neighbor(output[:, :, None, :], [stride * length, 1])[:, :, 0, :]
        output = lib.ops.Conv1D.conv1d(name, input_dim, output_dim, filter_size, output,
                                       he_init=he_init, biases=biases)

    else:
        output = lib.ops.Conv1D.transposd_conv1d(name, input_dim, output_dim, filter_size, output,
                                                 he_init=he_init, biases=biases, stride=stride)
    return tf.reshape(output, [-1, length * stride, output_dim])


def normalize(name, inputs, is_training_ph, use_bn=True):
    with tf.variable_scope(name):
        if use_bn:
            return tf.contrib.layers.batch_norm(inputs, fused=True, scale=True, decay=0.9,
                                                is_training=is_training_ph,
                                                epsilon=1e-5, scope='BN', updates_collections=None)
        else:
            return inputs


def resblock(name, input_dim, output_dim, filter_size, inputs, resample, is_training_ph, r=1.0,
             use_bn=True, stride=2):
    if resample == 'down':
        conv1 = functools.partial(lib.ops.Conv1D.conv1d, input_dim=input_dim, output_dim=input_dim)
        conv2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        shortcut_func = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample == 'up':
        conv1 = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim, stride=stride)
        conv2 = functools.partial(lib.ops.Conv1D.conv1d, input_dim=output_dim, output_dim=output_dim)
        shortcut_func = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim, stride=stride)
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
