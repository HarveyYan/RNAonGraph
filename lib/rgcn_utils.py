import tensorflow as tf
from lib.ops.Linear import linear
from lib.ops.LSTM import set2set_pooling, naive_attention


def graph_convolution_layers(name, inputs, units, reuse=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE if reuse else False):
        # adj_tensor: [batch_size, length, length, nb_bonds]
        adj_tensor, hidden_tensor, node_tensor = inputs

        # no bond excluded from computation, [batch_size, nb_bonds, length, length]
        adj = tf.transpose(adj_tensor[:, :, :, 1:], (0, 3, 1, 2))

        # binarize --> dropping relations: [batch_size, length, length]
        # adj = tf.cast(tf.greater(tf.argmax(adj_tensor, axis=-1), 0), tf.float32)

        # first layer or not [batch_size, length, input_dim]
        # annotations = tf.concat([hidden_tensor, node_tensor], axis=-1)\
        #     if hidden_tensor is not None else node_tensor
        annotations = hidden_tensor if hidden_tensor is not None else node_tensor

        input_dim = annotations.get_shape().as_list()[-1]
        nb_bonds = adj.get_shape().as_list()[1]

        # [batch_size, nb_bonds, length, units]
        output = tf.stack([linear('lt_bond_%d' % (i + 1), input_dim, units, annotations)
                           for i in range(nb_bonds)], axis=1)

        '''normalization doesn't really help ==> most nucleotides only have two incident edges'''
        # # [batch_size, length, units], message passing through all adjacent nodes and relations
        # output = tf.reduce_sum(tf.matmul(adj, output), axis=1)
        # # normalization factor, equals to the number of adjacent nodes (not relation specific)
        # normalization = tf.expand_dims(tf.reduce_sum(
        #     tf.cast(tf.greater(tf.argmax(adj_tensor, axis=-1, output_type=tf.int32), 0), tf.float32)
        #     , axis=-1), axis=-1)
        # output = output / normalization + \
        #          linear('self-connect', input_dim, units, annotations)

        output = tf.reduce_mean(tf.matmul(adj, output), axis=1)
        # self-connection \approx residual connection
        output = output + linear('self-connect', input_dim, units, annotations)
        return output  # messages


def att_gcl(name, inputs, units, reuse=True, expr_simplified_att=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE if reuse else False):
        # [batch_size, length, length, nb_bonds]
        adj_tensor, hidden_tensor, node_tensor = inputs

        # first layer or not [batch_size, length, input_dim]
        annotations = hidden_tensor if hidden_tensor is not None else node_tensor

        input_dim, length = annotations.get_shape().as_list()[-1], annotations.get_shape().as_list()[-2]
        nb_bonds = adj_tensor.get_shape().as_list()[-1] - 1

        # dropping no bond type and adding self-connection
        # adj_tensor = tf.concat([
        #     tf.tile(tf.eye(length)[None, :, :], [tf.shape(annotations)[0], 1, 1])[:, :, :, None],
        #     # tf.expand_dims(tf.eye(length, [tf.shape(annotations)[0]]), axis=-1),
        #     adj_tensor[:, :, :, 1:]
        # ], axis=-1)
        adj_tensor = adj_tensor[:, :, :, 1:]
        # [batch_size, length, length, nb_bonds]

        # nb_bonds equals the number of relations. The first relation is self-connection.
        output = linear('lt', input_dim, units * nb_bonds, annotations, biases=False)

        '''first step: pre-allocating for massage passing'''
        # [batch_size, length, units, nb_bonds]
        # (outcoming) messages for each node by relation
        output_by_rel = tf.reshape(output, [-1, length, units, nb_bonds])

        # selecting the message by relation for each pair of nodes => a pair of nodes would only have at most one type of relation
        # reduce_mean/normalization with pre_sum would lead to the ordinary rgcn
        # NOTE: messages from the entire graph are different for each node, based on the relation types with its neighbors
        # this is a major difference to the previous graph attention network
        # [batch_size, length, length, units]
        pre_sum = tf.matmul(adj_tensor, output_by_rel, transpose_b=True)

        # [batch_size, length, length], determine if interaction between nodes is present
        bias_mat = (tf.reduce_max(adj_tensor, axis=-1) - 1.) * 1e9

        if expr_simplified_att:
            # simply comparing the node features,
            # disregarding the connection (and types) between the nodes
            att_weights = tf.nn.softmax(tf.matmul(
                linear('message_attention_simplified', units, units, annotations, biases=False),
                annotations,
                transpose_b=True
            ) + bias_mat)

            # att_weights = tf.nn.softmax(
            #     tf.nn.leaky_relu(
            #         linear('message_attention_simplified_l', units, 1, annotations, biases=False) + \
            #         tf.transpose(linear('message_attention_simplified_r', units, 1, annotations, biases=False),
            #                      [0, 2, 1])
            #     ) + bias_mat)

        else:
            '''note, messages haven't been passed yet'''
            '''second step: compute scores by comparing the messages within each relation'''
            '''multiplicative style'''
            # [batch_size, nb_bonds, length, units]
            output_rel_first = tf.transpose(output_by_rel, (0, 3, 1, 2))
            # [batch_size, length, length, nb_bonds]
            scores_by_rel = tf.transpose(
                tf.matmul(
                    linear('message_attention', units, units, output_rel_first, biases=False),
                    output_rel_first,
                    transpose_b=True),
                (0, 2, 3, 1)
            )
            # select scores by relation
            sum_rel = tf.reduce_sum(tf.multiply(scores_by_rel, adj_tensor), axis=-1)

            # node level attention
            att_weights = tf.nn.softmax(sum_rel + bias_mat)

        hidden_tensor = tf.reduce_sum(pre_sum * att_weights[:, :, :, None], axis=2)

        # if input_dim == units:
        #     return hidden_tensor + annotations
        # else:
        return hidden_tensor + linear('OutputMapping', input_dim, units, annotations, biases=False)


def relational_gcn(inputs, units, is_training_ph, dropout_rate=0., use_att=False):
    adj_tensor, hidden_tensor, node_tensor = inputs

    for i, u in enumerate(units):
        if use_att:
            hidden_tensor = att_gcl('graph_convolution_%d' % (i + 1),
                                    (adj_tensor, hidden_tensor, node_tensor), u)
        else:
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
        annotations = tf.concat(
            [hidden_tensor, inputs[1], node_tensor] if inputs[1] is not None else [hidden_tensor, node_tensor],
            axis=-1
        )
        return set2set_pooling('set2set_pooling', hidden_tensor, 10)
        # return naive_attention('naive_attention', 50, annotations)


def normalize(name, inputs, use_bn, is_training_ph):
    with tf.variable_scope(name):
        if use_bn:
            return tf.contrib.layers.batch_norm(inputs, scale=True, fused=True, decay=0.9, is_training=is_training_ph,
                                                scope='BN', reuse=tf.get_variable_scope().reuse,
                                                updates_collections=None)
        else:
            return inputs


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
