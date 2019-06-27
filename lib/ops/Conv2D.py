import tensorflow as tf
import numpy as np


def conv2d(name, input_dim, output_dim, filter_size, inputs, stride=1, he_init=True, biases=True):

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
        fan_in = input_dim * filter_size ** 2
        fan_out = output_dim * filter_size ** 2 / stride ** 2

        if he_init:
            filters_stdev = np.sqrt(4 / (fan_in + fan_out))
        else:  # Normalized init (Glorot)
            filters_stdev = np.sqrt(2 / (fan_in + fan_out))

        # [filter_height, filter_width, in_channels, out_channels]
        filters = tf.get_variable('filters', shape=(filter_size, filter_size, input_dim, output_dim),
                                  initializer=uniform_init(filters_stdev))

        result = tf.nn.conv2d(
            input=inputs,
            filter=filters,
            strides=(1, stride, stride, 1),
            padding='SAME'
        )

        if biases:
            bias = tf.get_variable('bias', shape=(output_dim,), initializer=tf.initializers.zeros())
            result = tf.nn.bias_add(result, bias)

        return result