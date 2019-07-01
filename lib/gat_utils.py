import tensorflow as tf
import lib.ops.LSTM


def normalize(name, inputs, use_bn, is_training_ph):
    with tf.variable_scope(name):
        if use_bn:
            return tf.contrib.layers.batch_norm(inputs, fused=True, decay=0.9, is_training=is_training_ph,
                                                scope='BN', reuse=tf.get_variable_scope().reuse,
                                                updates_collections=None)
        else:
            return tf.contrib.layers.layer_norm(inputs, scope='LN', reuse=tf.get_variable_scope().reuse)


def attn_head(name, node_tensor, bias_mat, hidden_units, activation, is_training_ph,
              in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope(name):
        inputs = node_tensor

        node_tensor = tf.layers.dropout(node_tensor, in_drop, is_training_ph)
        node_tensor = tf.layers.conv1d(node_tensor, hidden_units, 1, use_bias=False)
        # [bs, len, hidden_dim]

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(node_tensor, 1, 1)  # [bs, len, 1]
        f_2 = tf.layers.conv1d(node_tensor, 1, 1)  # [bs, len, 1]
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])  # [bs, len, len]
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)  # oops, touché...

        coefs = tf.layers.dropout(coefs, coef_drop, is_training_ph)
        node_tensor = tf.layers.dropout(node_tensor, in_drop, is_training_ph)

        # message passing
        vals = tf.matmul(coefs, node_tensor)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if inputs.shape[-1] != ret.shape[-1]:
                ret = ret + tf.layers.conv1d(inputs, ret.shape[-1], 1)  # activation
            else:
                ret = ret + inputs

        return activation(ret)  # activation


def gat_model(inputs, bias_mat, hid_units, n_heads, attn_drop, ffd_drop, is_training_ph,
              activation=tf.nn.elu, residual=False):
    output = inputs
    for i, heads in enumerate(n_heads):
        attns = []
        with tf.variable_scope('att_layer_%d' % (i)):
            for j in range(heads):
                attns.append(attn_head('head_%d' % (j), output, bias_mat, hid_units[i],
                                       activation, is_training_ph,
                                       in_drop=ffd_drop, coef_drop=attn_drop,
                                       residual=False if i == 0 or len(n_heads) - 1 else residual))
            if i == len(n_heads) - 1:
                output = tf.add_n(attns) / n_heads[-1]
            else:
                output = tf.concat(attns, axis=-1)

            output = normalize('bn%d'%(i), output, True, is_training_ph)


    with tf.variable_scope('aggregation_stage'):
        output = lib.ops.LSTM.naive_attention('naive_attention', 50, output)

    return output
