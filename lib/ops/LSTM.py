import tensorflow as tf
import lib.ops.Linear, lib.ops.BatchNorm
import functools


def bilstm(name, hidden_units, inputs, length, dropout_rate, is_training_ph):
    with tf.variable_scope(name):
        cell_forward = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(hidden_units, name='forward_cell'),
            output_keep_prob=tf.cond(is_training_ph, lambda: 1 - dropout_rate, lambda: 1.)  # keep prob
        )
        cell_backward = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(hidden_units, name='backward_cell'),
            output_keep_prob=tf.cond(is_training_ph, lambda: 1 - dropout_rate, lambda: 1.)
        )

        state_forward = cell_forward.zero_state(tf.shape(inputs)[0], tf.float32)
        state_backward = cell_backward.zero_state(tf.shape(inputs)[0], tf.float32)

        input_forward = inputs
        input_backward = tf.reverse(inputs, [1])

        output_forward = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)
        output_backward = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)

        # unroll
        i = tf.constant(0)
        while_condition = lambda i, _1, _2, _3, _4: tf.less(i, length)

        def body(i, output_forward, output_backward, state_forward, state_backward):
            cell_output_forward, state_forward = cell_forward(input_forward[:, i, :], state_forward)
            output_forward = output_forward.write(i, cell_output_forward)
            cell_output_backward, state_backward = cell_backward(input_backward[:, i, :], state_backward)
            output_backward = output_backward.write(i, cell_output_backward)
            return [tf.add(i, 1), output_forward, output_backward, state_forward, state_backward]

        _, output_forward, output_backward, state_forward, state_backward = tf.while_loop(while_condition, body,
                                                                                          [i, output_forward,
                                                                                           output_backward,
                                                                                           state_forward,
                                                                                           state_backward])
        output_forward = tf.transpose(output_forward.stack(), [1, 0, 2])
        output_backward = tf.reverse(tf.transpose(output_backward.stack(), [1, 0, 2]), [1])
        output = tf.concat([output_forward, output_backward], axis=2)

        return output


def naive_attention(name, attention_size, inputs):
    batch_size, nb_steps, nb_features = inputs.shape.as_list()
    with tf.variable_scope(name):
        context_vec = tf.nn.relu(
            lib.ops.Linear.linear('Context_Vector', nb_features, attention_size, tf.reshape(inputs, [-1, nb_features])))
        pre_weights_exp = tf.exp(
            tf.reshape(lib.ops.Linear.linear('Attention_weights', attention_size, 1, context_vec, biases=False),
                       [tf.shape(inputs)[0], -1]))
        weights = pre_weights_exp / tf.reduce_sum(pre_weights_exp, 1)[:, None]
        output = tf.reduce_sum(inputs * weights[:, :, None], 1)
        return output


def self_attention(name, attention_size, inputs, use_conv=False):
    batch_size, nb_steps, nb_features = inputs.shape.as_list()
    with tf.variable_scope(name):
        if use_conv:
            func = functools.partial(lib.ops.Conv1D.conv1d, filter_size=1)
        else:
            func = lib.ops.Linear.linear
        cv_f = func(name='Context_Vector_f', input_dim=nb_features, output_dim=attention_size, inputs=inputs)
        cv_g = func(name='Context_Vector_g', input_dim=nb_features, output_dim=attention_size, inputs=inputs)
        cv_h = func(name='Context_Vector_h', input_dim=nb_features, output_dim=nb_features, inputs=inputs)

        sa_scores = tf.matmul(cv_f, cv_g, transpose_b=True)  # [batch_size, nb_steps, nb_steps]
        sa_weights = tf.nn.softmax(sa_scores, axis=-1)[:, :, :, None]  # [batch_size, nb_steps, nb_steps, 1]

        # tf.transpose(tf.reshape(cv_h, [batch_size, nb_steps, nb_features]), perm=[1, 2, 0])  #[nb_steps, nb_features, batch_size]
        # return tf.reduce_sum(sa_weights * tf.stack([cv_h] * nb_steps, axis=1),
        #                      axis=2)  # [batch_size, nb_steps, nb_features]
        stacked = tf.reshape(tf.tile(cv_h, [1, tf.shape(inputs)[1], 1]),
                             [-1, tf.shape(inputs)[1], tf.shape(inputs)[1], nb_features])
        return tf.reduce_sum(sa_weights * stacked, axis=2)  # [batch_size, nb_steps, nb_features]


#########################################
#
# set2set pooling operations
#
#########################################

def BiLSTMEncoder(name, hidden_units, inputs, length, dropout_rate, is_training_ph):
    with tf.variable_scope(name):
        cell_forward = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(hidden_units, name='forward_cell'),
            output_keep_prob=tf.cond(is_training_ph, lambda: 1 - dropout_rate, lambda: 1.)  # keep prob
        )
        cell_backward = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(hidden_units, name='backward_cell'),
            output_keep_prob=tf.cond(is_training_ph, lambda: 1 - dropout_rate, lambda: 1.)
        )

        state_forward = cell_forward.zero_state(tf.shape(inputs)[0], tf.float32)
        state_backward = cell_backward.zero_state(tf.shape(inputs)[0], tf.float32)

        input_forward = inputs
        input_backward = tf.reverse(inputs, [1])

        output_forward = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)
        output_backward = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)

        # unroll
        i = tf.constant(0)
        while_condition = lambda i, _1, _2, _3, _4: tf.less(i, length)

        def body(i, output_forward, output_backward, state_forward, state_backward):
            cell_output_forward, state_forward = cell_forward(input_forward[:, i, :], state_forward)
            output_forward = output_forward.write(i, cell_output_forward)
            cell_output_backward, state_backward = cell_backward(input_backward[:, i, :], state_backward)
            output_backward = output_backward.write(i, cell_output_backward)
            return [tf.add(i, 1), output_forward, output_backward, state_forward, state_backward]

        _, output_forward, output_backward, state_forward, state_backward = tf.while_loop(while_condition, body,
                                                                                          [i, output_forward,
                                                                                           output_backward,
                                                                                           state_forward,
                                                                                           state_backward])
        output_forward = tf.transpose(output_forward.stack(), [1, 0, 2])
        output_backward = tf.reverse(tf.transpose(output_backward.stack(), [1, 0, 2]), [1])
        output = tf.concat([output_forward, output_backward], axis=2)

        encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=tf.concat([state_forward[0], state_backward[0]], axis=-1),
                                                      h=tf.concat([state_forward[1], state_backward[1]], axis=-1))

        return output, encoder_state


def set2set_attention(name, encoder_outputs, cell_output):
    '''
    Luong's multiplicative attention
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        scores = tf.matmul(encoder_outputs, cell_output[:, None, :], transpose_b=True)[:, :, 0]
        attention_weights = tf.nn.softmax(scores, axis=-1)
        context_vector = tf.reduce_sum(encoder_outputs * attention_weights[:, :, None], axis=1)
        return tf.concat([context_vector, cell_output], axis=-1)


def set2set_pooling(name, inputs, T, dropout_rate, is_training_ph, lstm_encoder=False):
    with tf.variable_scope(name):
        nb_features = inputs.get_shape().as_list()[-1]
        if lstm_encoder:
            inputs, state = BiLSTMEncoder('BiLSTMEncoder', nb_features, inputs,
                                          tf.shape(inputs)[1], dropout_rate, is_training_ph)
            nb_features = state[0].get_shape().as_list()[-1]
            cell = tf.nn.rnn_cell.LSTMCell(nb_features, name='decoder_lstm_cell')
        else:
            cell = tf.nn.rnn_cell.LSTMCell(nb_features, name='decoder_lstm_cell')
            state = cell.zero_state(tf.shape(inputs)[0], tf.float32)
        start_token = tf.zeros((tf.shape(inputs)[0], nb_features * 2))

        i = tf.constant(0)
        while_condition = lambda i, *args: tf.less(i, T)

        def body(i, state, token):
            cell_output, state = cell(token, state)
            attention_vector = set2set_attention('DecoderATT', inputs, cell_output)
            return [tf.add(i, 1), state, attention_vector]

        _, state, final_token = tf.while_loop(while_condition, body, [i, state, start_token])
        return final_token
