import tensorflow as tf
import numpy as np


def conv1d(name, input_dim, output_dim, filter_size, inputs, stride=1, he_init=True, biases=True):
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

        filters = tf.get_variable('filters', shape=(filter_size, input_dim, output_dim),
                                  initializer=uniform_init(filters_stdev))

        result = tf.nn.conv1d(
            value=inputs,
            filters=filters,
            stride=stride,
            padding='SAME'
        )

        if biases:
            bias = tf.get_variable('bias', shape=(output_dim,), initializer=tf.initializers.zeros())
            result = tf.nn.bias_add(result, bias)

        return result


def transposd_conv1d(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    # From [batch, width, input_dim] to [batch, width*2, output_dim]
    with tf.variable_scope(name):
        # ignoring mask_type from original code

        def uniform_init(stdev):
            return tf.initializers.random_uniform(
                minval=-stdev * np.sqrt(3),
                maxval=stdev * np.sqrt(3)
            )

        stride = 2
        fan_in = input_dim * filter_size / stride
        fan_out = output_dim * filter_size

        if he_init:
            filters_stdev = np.sqrt(4 / (fan_in + fan_out))
        else:  # Normalized init (Glorot)
            filters_stdev = np.sqrt(2 / (fan_in + fan_out))

        # requires [filter_width, output_channels, in_channels] of a kernel
        filters = tf.get_variable('filters', shape=(filter_size, output_dim, input_dim),
                                  initializer=uniform_init(filters_stdev))

        # infer output shape
        input_shape = tf.shape(inputs)
        output_shape = tf.stack([input_shape[0], 2*input_shape[1], output_dim])

        result = tf.contrib.nn.conv1d_transpose(
            value=inputs,
            filter=filters,
            output_shape=output_shape,
            stride=stride,
            padding='SAME'
        )

        if biases:
            bias = tf.get_variable('bias', shape=(output_dim,), initializer=tf.initializers.zeros())
            result = tf.nn.bias_add(result, bias)

        return result