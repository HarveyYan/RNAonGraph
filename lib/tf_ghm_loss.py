"""
MIT License

Copyright (c) 2019 Kuan-Yu Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

TF version for gradient harmonized weights

Modified by Z. Yan
"""

import numpy as np
import tensorflow as tf


def get_ghm_weight(predict, target, valid_mask, bins=10, alpha=0.75,
                   dtype=tf.float32, name='GHM_weight'):
    """ Get gradient Harmonized Weights.
    This is an implementation of the GHM ghm_weights described
    in https://arxiv.org/abs/1811.05181.

    This module will only work on the gradients, i.e. training part

    Args:
        predict:
            The prediction of categories branch, [0, 1].
            -shape [batch_size, length, 2].
        target:
            The target of categories branch, {0, 1}.
            -shape [batch_size, length, ].
        valid_mask:
            The valid mask, is 0 when the sample is ignored, {0, 1}.
            -shape [batch_size, category_num].
        bins:
            The number of bins for region approximation.
        alpha:
            The moving average parameter.
            Default to 0.75 as suggested in the original literature.
        dtype:
            The dtype for all operations.
    
    Returns:
        weights:
            The beta value of each sample described in paper.
    """
    with tf.variable_scope(name):
        _edges = [x / bins for x in range(bins + 1)]
        _edges[-1] += 1e-6
        edges = tf.constant(_edges, dtype=dtype)

        # _shape = predict.get_shape().as_list()
        _shape = [128, 200]

        _init_statistics = (np.prod(_shape)) / bins
        statistics = tf.get_variable(
            name='statistics', shape=[bins], dtype=dtype, trainable=False,
            initializer=tf.constant_initializer(_init_statistics, dtype=dtype))

        _b_valid = valid_mask > 0
        total = tf.reduce_sum(valid_mask)

        # [batch_size, length]
        gradients = tf.reduce_sum((1. - tf.one_hot(target, depth=2, dtype=tf.float32)) * predict, axis=-1)

        # Calculate new statics and new weights
        w_list = []
        s_list = []
        for i in range(bins):
            inds = (gradients >= edges[i]) & (gradients < edges[i + 1]) & _b_valid
            # number of examples lying in bin, same as R in paper.
            num_in_bin = tf.reduce_sum(tf.cast(inds, dtype=dtype))
            statistics_i = alpha * statistics[i] + (1 - alpha) * num_in_bin
            gradient_density = statistics_i * bins
            update_weights = total / gradient_density
            weights_i = tf.where(
                inds,
                x=tf.ones_like(gradients) * update_weights,
                y=tf.zeros_like(gradients)
            )
            w_list.append(weights_i)
            s_list.append(statistics_i)

        weights = tf.add_n(w_list)
        new_statistics = tf.stack(s_list)

        # Avoid the tiny value in statistics
        new_statistics = tf.maximum(new_statistics, _init_statistics)
        # Update statistics
        statistics_updated_op = tf.assign(statistics, new_statistics)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, statistics_updated_op)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        return weights


def main():
    ghm_weights = get_ghm_weight(predict=tf.constant([[1., 0., 0.5, 0.]]),
                                 target=tf.constant([[1., 0., 0., 1.]]),
                                 valid_mask=tf.constant([[1., 1., 1., 1.]]))

    # update method same as the batch norm update with optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = tf.constant(0, name='your_optimizer')

    with tf.Session() as sess:
        init_ops = tf.global_variables_initializer()
        sess.run(init_ops)

        _, _ghm_weights = sess.run([opt, ghm_weights])
        print('update 1 times: ', _ghm_weights)

        for _ in range(100):
            sess.run([opt])
        print('update 100 times: ', sess.run([ghm_weights]))


if __name__ == '__main__':
    main()
