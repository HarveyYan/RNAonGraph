import tensorflow as tf
from lib.ops.Linear import linear
from lib.ops.Conv1D import conv1d


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
        output = tf.stack([linear('lt_bond_%d' % (i + 1), input_dim, units, annotations, biases=False)
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
        output = output + linear('self-connect', input_dim, units, annotations, biases=False)
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


def sparse_graph_convolution_layers(name, inputs, units, reuse=True):
    """
    This one is used by the Joint_SMRGCN model;
    A crude prototypical operation
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE if reuse else False):
        # adj_tensor: list (size nb_bonds) of [length, length] matrices
        adj_tensor, hidden_tensor, node_tensor = inputs
        annotations = hidden_tensor if hidden_tensor is not None else node_tensor
        input_dim = annotations.get_shape().as_list()[-1]
        nb_bonds = len(adj_tensor)
        output = []
        for i in range(nb_bonds):
            msg_bond = linear('lt_bond_%d' % (i + 1), input_dim, units,
                              annotations, biases=False, variables_on_cpu=False)
            output.append(tf.sparse_tensor_dense_matmul(adj_tensor[i], msg_bond))
        output = tf.add_n(output) / nb_bonds
        # self-connection \approx residual connection
        output = output + linear('self-connect', input_dim, units, annotations, variables_on_cpu=False)
        return output  # messages


def sparse_att_gcl(name, inputs, units, reuse=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE if reuse else False):
        # adj_tensor: list (size nb_bonds) of [length, length] matrices
        adj_tensor, hidden_tensor, node_tensor = inputs
        annotations = hidden_tensor if hidden_tensor is not None else node_tensor
        input_dim = annotations.get_shape().as_list()[-1]
        nb_bonds = len(adj_tensor)
        output = []
        for i in range(nb_bonds):
            # the first two are covalent bonds, doesn't make sense to compute attention
            msg_bond = linear('lt_bond_%d' % (i + 1), input_dim, units,
                              annotations, biases=False, variables_on_cpu=False)
            if i in [0, 1]:
                output.append(tf.sparse_tensor_dense_matmul(adj_tensor[i], msg_bond))
            else:
                # not feasible given current implementation,
                # unless an additional batch_size dimension is maintained to reduce the space complexity
                f_1 = tf.layers.conv1d(annotations[None, :, :], 1, 1, name='att_linear_left_%d' % (i))[0]
                f_2 = tf.layers.conv1d(annotations[None, :, :], 1, 1, name='att_linear_right_%d' % (i))[0]
                logits = f_1 + tf.transpose(f_2, [1, 0])
                coefs = adj_tensor[i] * tf.nn.softmax(tf.nn.leaky_relu(logits))
                output.append(tf.sparse_tensor_dense_matmul(coefs, msg_bond))
        output = tf.add_n(output) / nb_bonds
        # self-connection \approx residual connection
        output = output + linear('self-connect', input_dim, units, annotations, variables_on_cpu=False)
        return output  # messages


def sparse_dense_matmult_batch(sp_a, b):
    def map_function(x):
        i, dense_slice = x[0], x[1]
        sparse_slice = tf.sparse.reshape(tf.sparse.slice(
            sp_a, [i, 0, 0], [1, sp_a.dense_shape[1], sp_a.dense_shape[2]]),
            [sp_a.dense_shape[1], sp_a.dense_shape[2]])
        mult_slice = tf.sparse.sparse_dense_matmul(sparse_slice, dense_slice)
        return mult_slice

    elems = (tf.range(0, sp_a.dense_shape[0], delta=1, dtype=tf.int64), b)
    return tf.map_fn(map_function, elems, dtype=tf.float32, back_prop=True)


def joint_layer(name, inputs, units, reuse=True, batch_axis=False,
                use_attention=False):
    '''
    1. viewing the RNA secondary structure as undirected graph
       (in fact, the bidrectional lstm after the GNN layers
       would tell us something about the direction later)
    2. the undirected adjacency matrix would contain the equilibrium base pairing probabilities,
       or the actual base pairs sampled from that probabilities
    3. attention is only enabled when the adjacency tensor has a batch_size dimension,
       that comes with a shape: (batch_size, nb_nodes, nb_nodes); therefore the adjacency tensor
       in this case must follow a dense implementation, since tensorflow doesn't have a
       very good support for sparse matrix multiplication over rank 2.
        3.1 when there isn't a batch axis, for a adjacency matrix that has shape:
            (batch_size * nb_nodes, batch_size * nb_nodes), a sparse implementation must
            then be followed, otherwise the it may not be fit into the
            memory.
        3.2 Attention is only possible in the dense setting, for in the sparse settings, attention
            would require too much memory
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE if reuse else False):
        # adj_tensor: list (size nb_bonds) of [length, length] matrices
        adj_tensor, hidden_tensor, node_tensor = inputs
        annotations = hidden_tensor if hidden_tensor is not None else node_tensor
        input_dim = annotations.get_shape().as_list()[-1]

        output = []
        msg_bond = linear('hydro_bond', input_dim, units,
                          annotations, biases=False, variables_on_cpu=False)
        if batch_axis:
            # tensorflow doesn't have a very well support for higher ranks (>=3) sparse tensor multiplication
            # therefore, if we have a batch axis, we'd better use the vanilla matmul for dense matrices
            if use_attention:
                bias_mat = (1. - tf.cast(adj_tensor > 0, tf.float32)) * -10000. + adj_tensor
                f_1 = tf.layers.conv1d(annotations, 1, 1, name='att_linear_left', use_bias=False)
                f_2 = tf.layers.conv1d(annotations, 1, 1, name='att_linear_right', use_bias=False)
                logits = f_1 + tf.transpose(f_2, [0, 2, 1])
                coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) * bias_mat)
                bp_msg = tf.matmul(coefs, msg_bond)
            else:
                bp_msg = tf.matmul(adj_tensor, msg_bond) # / (tf.reduce_sum(adj_tensor, axis=-1, keepdims=True) + 1e-6)
            output.append(bp_msg)
            output.append(conv1d('conv1', input_dim, units, 10, annotations, biases=False,
                                 pad_mode='SAME_EVEN', pad_val='CONSTANT', variables_on_cpu=False))
        else:
            # boundary nucleotides from other sequences are used as padding, does not seem to matter too much
            # padding are all placed on the left
            output.append(tf.sparse_tensor_dense_matmul(adj_tensor, msg_bond))
            output.append(conv1d('conv1', input_dim, units, 10, annotations[None, :, :], biases=False,
                                 pad_mode='SAME', pad_val='CONSTANT', variables_on_cpu=False)[0, :, :])

        output = tf.add_n(output) / 2
        # self-connection \approx residual connection
        output = output + linear('self-connect', input_dim, units, annotations, variables_on_cpu=False)
        return output  # messages


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
