import tensorflow as tf
import numpy as np

def linear(name, input_dim, output_dim, inputs, initialization = None, biases=True):
    with tf.variable_scope(name):

        def uniform_init(stdev):
            return tf.initializers.random_uniform(
                minval=-stdev * np.sqrt(3),
                maxval=stdev * np.sqrt(3)
            )

        if initialization == 'lecun':  # and input_dim != output_dim):
            # disabling orth. init for now because it's too slow
            init = uniform_init(
                np.sqrt(1. / input_dim)
            )
        elif initialization == 'glorot' or (initialization == None):
            init = uniform_init(
                np.sqrt(2. / (input_dim + output_dim))
            )
        elif initialization == 'he':
            init = uniform_init(
                np.sqrt(2. / input_dim)
            )
        elif initialization == 'glorot_he':
            init = uniform_init(
                np.sqrt(4. / (input_dim + output_dim))
            )
        elif initialization == 'orthogonal' or \
                (initialization == None and input_dim == output_dim):

            # From lasagne
            def sample(shape):
                if len(shape) < 2:
                    raise RuntimeError("Only shapes of length 2 or more are "
                                       "supported.")
                flat_shape = (shape[0], np.prod(shape[1:]))
                # TODO: why normal and not uniform?
                a = np.random.normal(0.0, 1.0, flat_shape)
                u, _, v = np.linalg.svd(a, full_matrices=False)
                # pick the one with the correct shape
                q = u if u.shape == flat_shape else v
                q = q.reshape(shape)
                return q.astype('float32')

            init = tf.constant_initializer(sample((input_dim, output_dim)))
        elif initialization[0] == 'uniform':
            init = tf.initializers.random_uniform(
                minval=-initialization[1],
                maxval=initialization[1],
            )
        else:
            raise Exception('Unknown initialization!')

        weight = tf.get_variable('W', shape=(input_dim, output_dim), initializer=init)

        if inputs.get_shape().ndims == 2:
            result = tf.matmul(inputs, weight)
        else:
            # presumably NWHC
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            result = tf.matmul(reshaped_inputs, weight)
            result = tf.reshape(result, tf.stack(tf.unstack(tf.shape(inputs))[:-1] + [output_dim]))

        if biases:
            result = tf.nn.bias_add(
                result,
                bias=tf.get_variable('b', shape=(output_dim,), initializer=tf.zeros_initializer())
            )

        return result