import tensorflow as tf
import numpy as np


def conv1d(name, input_dim, output_dim, filter_size, inputs, stride=1, he_init=True, biases=True, dilation=1, pad_mode='SAME', variables_on_cpu=True):
    with tf.variable_scope(name):
        # ignoring mask_type from original code

        def uniform_init(stdev):
            return tf.initializers.random_uniform(
                minval=-stdev * np.sqrt(3),
                maxval=stdev * np.sqrt(3)
            )

        '''
        Cited notes from stackoverflow:

        fan_in = n_feature_maps_in * receptive_field_height * receptive_field_width
        fan_out = n_feature_maps_out * receptive_field_height * receptive_field_width / max_pool_area
        where receptive_field_height and receptive_field_width correspond to those of the conv layer under consideration,
        and max_pool_area is the product of the height and width of the max pooling (or any downsampling) that follows the convolution layer.
        '''
        fan_in = input_dim * filter_size
        fan_out = output_dim * filter_size / stride

        if he_init:
            filters_stdev = np.sqrt(4 / (fan_in + fan_out))
        else:  # Normalized init (Glorot)
            filters_stdev = np.sqrt(2 / (fan_in + fan_out))

        if variables_on_cpu:
            with tf.device('/cpu:0'):
                filters = tf.get_variable('filters', shape=(filter_size, input_dim, output_dim),
                                          initializer=uniform_init(filters_stdev))
        else:
            filters = tf.get_variable('filters', shape=(filter_size, input_dim, output_dim),
                                      initializer=uniform_init(filters_stdev))

        '''
        use mirror-padding on the spatial axis, rank of inputs would be 3, SAME padding mode
        1 2 3 4 5 6 7 # # #
        1 1 1 1
            1 1 1 1
                1 1 1 1
                    1 1 1 1
        output_length = (length - F + P) // stride + 1
        '''
        if pad_mode == 'SAME':
            length = tf.shape(inputs)[1]
            output_length = (length + stride - 1) // stride
            num_paddings = (output_length - 1)*stride + filter_size - length
            left_paddings = num_paddings
            right_paddings = 0
            inputs = tf.pad(inputs, [[0, 0], [left_paddings, right_paddings], [0, 0]], mode="CONSTANT")

        result = tf.nn.conv1d(
            value=inputs,
            filters=filters,
            stride=stride,
            padding='VALID',
            # dilations=dilation
        )

        if biases:
            if variables_on_cpu:
                with tf.device('/cpu:0'):
                    bias = tf.get_variable('bias', shape=(output_dim,), initializer=tf.initializers.zeros())
            else:
                bias = tf.get_variable('bias', shape=(output_dim,), initializer=tf.initializers.zeros())
            result = tf.nn.bias_add(result, bias)

        return result


def transposd_conv1d(name, input_dim, output_dim, filter_size, inputs, stride=2, he_init=True, biases=True):
    # From [batch, width, input_dim] to [batch, width*2, output_dim]
    with tf.variable_scope(name):
        # ignoring mask_type from original code

        def uniform_init(stdev):
            return tf.initializers.random_uniform(
                minval=-stdev * np.sqrt(3),
                maxval=stdev * np.sqrt(3)
            )

        fan_in = input_dim * filter_size / stride
        fan_out = output_dim * filter_size

        if he_init:
            filters_stdev = np.sqrt(4 / (fan_in + fan_out))
        else:  # Normalized init (Glorot)
            filters_stdev = np.sqrt(2 / (fan_in + fan_out))

        with tf.device('/cpu:0'):
            # requires [filter_width, output_channels, in_channels] of a kernel
            filters = tf.get_variable('filters', shape=(filter_size, output_dim, input_dim),
                                      initializer=uniform_init(filters_stdev))

        # infer output shape
        input_shape = tf.shape(inputs)
        output_shape = tf.stack([input_shape[0], stride * input_shape[1], output_dim])

        result = tf.contrib.nn.conv1d_transpose(
            value=inputs,
            filter=filters,
            output_shape=output_shape,
            stride=stride,
            padding='SAME'
        )

        if biases:
            with tf.device('/cpu:0'):
                bias = tf.get_variable('bias', shape=(output_dim,), initializer=tf.initializers.zeros())
            result = tf.nn.bias_add(result, bias)

        return result