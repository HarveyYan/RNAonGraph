import tensorflow as tf
from lib.ops.Linear import linear
from lib.ops.LSTM import set2set_pooling, naive_attention


def graph_convolution_layers(name, inputs, units):
    with tf.variable_scope(name):
        adj_tensor, hidden_tensor, node_tensor = inputs

        # no bond excluded from computation, [batch_size, nb_bonds, length, length]
        adj = tf.transpose(adj_tensor[:, :, :, 1:], (0, 3, 1, 2))

        # binarize --> dropping relations: [batch_size, length, length]
        # adj = tf.cast(tf.greater(tf.argmax(adj_tensor, axis=-1), 0), tf.float32)

        # first layer or not [batch_size, length, input_dim]
        annotations = hidden_tensor if hidden_tensor is not None else node_tensor

        input_dim = annotations.get_shape().as_list()[-1]
        nb_bonds = adj.get_shape().as_list()[1]

        # [batch_size, nb_bonds, length, units]
        output = tf.stack([linear('lt_bond_%d' % (i + 1), input_dim, units, annotations)
                           for i in range(nb_bonds)], axis=1)
        # [batch_size, length, units]
        # output = linear('lt_bond', input_dim, units, annotations)

        output = tf.reduce_mean(tf.matmul(adj, output), axis=1)
        # output = tf.matmul(adj, output)  # [batch_size, length, units]

        # where is the normalization?
        output = output + linear('self-connect', input_dim, units, annotations)
        # output = output / tf.reduce_sum(adj, [-1])[:, :, None] + \
        #          linear('self-connect', input_dim, units, annotations)
        return output


def relational_gcn(inputs, units, is_training_ph, dropout_rate=0.):
    adj_tensor, hidden_tensor, node_tensor = inputs

    for i, u in enumerate(units):
        hidden_tensor = graph_convolution_layers('graph_convolution_%d' % (i + 1),
                                                 (adj_tensor, hidden_tensor, node_tensor), u)
        hidden_tensor = normalize('Norm_%d' % (i + 1), hidden_tensor, True, is_training_ph)
        hidden_tensor = tf.nn.leaky_relu(hidden_tensor)
        hidden_tensor = tf.layers.dropout(hidden_tensor, dropout_rate, training=is_training_ph)
        # [batch_size, length, u]

    # node_tensor = lib.ops.Conv1D.conv1d('channel_expanding', 4, units[0], 1, node_tensor)
    # for i, u in enumerate(units):
    #     if i == 0:
    #         hidden_tensor = residual_rgcn_block('rgcn_resblock_optim', units[0], units[0],
    #                                             (adj_tensor, hidden_tensor, node_tensor),
    #                                             is_training_ph, True)
    #     else:
    #         hidden_tensor = residual_rgcn_block('rgcn_resblock_%d' % (i + 1), units[i-1], u,
    #                                             (adj_tensor, hidden_tensor, node_tensor),
    #                                             is_training_ph, use_bn=True)

    # return hidden_tensor
    with tf.variable_scope('graph_aggregation'):
        # annotations = tf.concat(
        #     [hidden_tensor, inputs[1], node_tensor] if inputs[1] is not None else [hidden_tensor, node_tensor],
        #     axis=-1
        # )
        # set2set_pooling('set2set_pooling', hidden_tensor, 10)
        return naive_attention('naive_attention', 50, hidden_tensor)


def normalize(name, inputs, use_bn, is_training_ph):
    with tf.variable_scope(name):
        if use_bn:
            return tf.contrib.layers.batch_norm(inputs, fused=True, decay=0.9, is_training=is_training_ph,
                                                scope='BN', reuse=tf.get_variable_scope().reuse,
                                                updates_collections=None)
        else:
            return tf.contrib.layers.layer_norm(inputs, scope='LN', reuse=tf.get_variable_scope().reuse)


def residual_rgcn_block(name, input_dim, output_dim, inputs, is_training_ph, optimized=False, use_bn=True,
                        r=1.0):
    adj_tensor, hidden_tensor, node_tensor = inputs
    with tf.variable_scope(name):

        if output_dim == input_dim:
            shortcut = 0. if hidden_tensor is None else hidden_tensor
        else:
            shortcut = graph_convolution_layers('graph_convolution_shortcut', inputs, output_dim)

        if not optimized:
            hidden_tensor = normalize(name='Norm1', is_training_ph=is_training_ph, inputs=hidden_tensor, use_bn=use_bn)
            hidden_tensor = tf.nn.leaky_relu(hidden_tensor)
            hidden_tensor = graph_convolution_layers('graph_convolution_1', (adj_tensor, hidden_tensor, node_tensor),
                                                     output_dim)
            hidden_tensor = normalize(name='Norm2', is_training_ph=is_training_ph, inputs=hidden_tensor, use_bn=use_bn)
            hidden_tensor = tf.nn.leaky_relu(hidden_tensor)
            hidden_tensor = graph_convolution_layers('graph_convolution_2', (adj_tensor, hidden_tensor, node_tensor),
                                                     output_dim)
            return r * hidden_tensor + shortcut
        else:
            hidden_tensor = graph_convolution_layers('graph_convolution_1', (adj_tensor, hidden_tensor, node_tensor),
                                                     output_dim)
            hidden_tensor = tf.nn.leaky_relu(hidden_tensor)
            hidden_tensor = graph_convolution_layers('graph_convolution_2', (adj_tensor, hidden_tensor, node_tensor),
                                                     output_dim)
            return hidden_tensor + shortcut
