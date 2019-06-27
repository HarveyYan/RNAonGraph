import tensorflow as tf

def add_noise(name, inputs, stddev, bAdd, bMul=True):
    with tf.variable_scope(name):

        if bAdd:
            inputs += tf.truncated_normal(tf.shape(inputs), 0, stddev)

        if bMul:
            inputs *= tf.truncated_normal(tf.shape(inputs), 1, stddev)

        return inputs