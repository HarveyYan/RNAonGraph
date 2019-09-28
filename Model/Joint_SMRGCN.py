# hierarchical model
import os
import sys
import time
import numpy as np
import tensorflow as tf

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Model import _stats
from lib.rgcn_utils import sparse_graph_convolution_layers, normalize
import lib.plot, lib.logger, lib.clr
import lib.ops.LSTM, lib.ops.Linear, lib.ops.Conv1D
from lib.tf_ghm_loss import get_ghm_weights
from lib.AMSGrad import AMSGrad


class JSMRGCN:

    def __init__(self, node_dim, edge_dim, embedding_vec, gpu_device, **kwargs):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.embedding_vec = embedding_vec
        self.vocab_size = embedding_vec.shape[0]
        self.gpu_device = gpu_device
        assert (self.edge_dim == 4)
        # hyperparams
        self.units = kwargs.get('units', 32)
        self.layers = kwargs.get('layers', 20)
        self.pool_steps = kwargs.get('pool_steps', 10)
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.learning_rate = kwargs.get('learning_rate', 2e-4)
        self.use_clr = kwargs.get('use_clr', False)
        self.use_momentum = kwargs.get('use_momentum', False)
        self.use_bn = kwargs.get('use_bn', False)

        self.reuse_weights = kwargs.get('reuse_weights', False)
        self.lstm_ggnn = kwargs.get('lstm_ggnn', False)
        self.probabilistic = kwargs.get('probabilistic', True)

        self.mixing_ratio = kwargs.get('mixing_ratio', 0.)
        self.use_ghm = kwargs.get('use_ghm', False)

        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            if self.use_momentum:
                self.optimizer = tf.contrib.opt.MomentumWOptimizer(
                    1e-4, self.learning_rate * self.lr_multiplier,
                    0.9, use_nesterov=True
                )
            else:
                # self.optimizer = tf.contrib.opt.AdamWOptimizer(
                #     1e-4,
                #     learning_rate=self.learning_rate * self.lr_multiplier
                # )
                self.optimizer = AMSGrad(
                    learning_rate=self.learning_rate * self.lr_multiplier,
                    beta2=0.999
                )

            with tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE):
                self._build_ggnn()
                self._loss()
                self._train()
                self._merge()
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    self.train_op = self.optimizer.apply_gradients(self.gv)
                _stats('Joint_SMRGCN', self.gv)
                self.saver = tf.train.Saver(max_to_keep=10)
                self.init = tf.global_variables_initializer()
                self.local_init = tf.local_variables_initializer()
        self._init_session()

    def _placeholders(self):
        self.node_input_ph = tf.placeholder(tf.int32, shape=[None, ])  # nb_nodes
        self.adj_mat_ph = [tf.sparse_placeholder(tf.float32, shape=[None, None]) for _ in range(self.edge_dim)]
        # nb_nodes x nb_nodes

        self.labels = tf.placeholder(tf.int32, shape=[None, None, ])
        self.max_len = tf.placeholder(tf.int32, shape=())
        self.segment_length = tf.placeholder(tf.int32, shape=[None, ])  # always batch_size

        self.is_training_ph = tf.placeholder(tf.bool, ())
        self.global_step = tf.placeholder(tf.int32, ())
        self.hf_iters_per_epoch = tf.placeholder(tf.int32, ())
        if self.use_clr:
            self.lr_multiplier = lib.clr. \
                cyclic_learning_rate(self.global_step, 0.5, 5.,
                                     self.hf_iters_per_epoch, mode='exp_range')
        else:
            self.lr_multiplier = 1.
        # self.mixing_ratio_var = tf.placeholder_with_default(self.mixing_ratio, ())

    def _build_ggnn(self):
        embedding = tf.get_variable('embedding_layer', shape=(self.vocab_size, self.node_dim),
                                    initializer=tf.constant_initializer(self.embedding_vec), trainable=False)
        node_tensor = tf.nn.embedding_lookup(embedding, self.node_input_ph)

        if self.reuse_weights:
            if self.node_dim < self.units:
                node_tensor = tf.pad(node_tensor,
                                     [[0, 0], [0, self.units - self.node_dim]])
            elif self.node_dim > self.units:
                print('Changing \'self.units\' to %d!' % (self.node_dim))
                self.units = self.node_dim

        hidden_tensor = None
        with tf.variable_scope('gated-rgcn', reuse=tf.AUTO_REUSE):
            if self.lstm_ggnn:
                cell = tf.nn.rnn_cell.LSTMCell(self.units, name='lstm_cell')
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell,
                    output_keep_prob=tf.cond(self.is_training_ph, lambda: 1 - self.dropout_rate, lambda: 1.)
                    # keep prob
                )
                memory = None
            else:
                cell = tf.contrib.rnn.GRUCell(self.units)

            for i in range(self.layers):
                name = 'graph_convolution' if self.reuse_weights else 'graph_convolution_%d' % (i + 1)
                # variables for sparse implementation default placement to gpu
                msg_tensor = sparse_graph_convolution_layers(name, (self.adj_mat_ph, hidden_tensor, node_tensor),
                                                             self.units, reuse=self.reuse_weights)
                msg_tensor = normalize('Norm' if self.reuse_weights else 'Norm%d' % (i + 1),
                                       msg_tensor, self.use_bn, self.is_training_ph)
                msg_tensor = tf.nn.leaky_relu(msg_tensor)
                msg_tensor = tf.layers.dropout(msg_tensor, self.dropout_rate, training=self.is_training_ph)

                if hidden_tensor is None:  # hidden_state
                    state = node_tensor
                else:
                    state = hidden_tensor

                if self.lstm_ggnn:
                    if i == 0:
                        memory = tf.zeros(tf.shape(state), tf.float32)
                    hidden_tensor, (memory, _) = cell(msg_tensor, tf.nn.rnn_cell.LSTMStateTuple(memory, state))
                else:
                    hidden_tensor, _ = cell(msg_tensor, state)
                # [batch_size, length, u]
        # the original nucleotide embeddings learnt by GNN
        self.hidden_tensor = hidden_tensor
        # [nb_nodes, units] no dummies
        output = hidden_tensor

        # while loop to recover batch size
        batch_output = tf.TensorArray(tf.float32, size=tf.shape(self.segment_length)[0], infer_shape=True,
                                      dynamic_size=True)
        mask_offset = tf.TensorArray(tf.int32, size=tf.shape(self.segment_length)[0], infer_shape=True,
                                     dynamic_size=True)
        i = tf.constant(0)
        start_idx = tf.constant(0)
        while_condition = lambda i, _1, _2, _3: tf.less(i, tf.shape(self.segment_length)[0])

        def body(i, start_idx, batch_output, mask_offset):
            end_idx = start_idx + self.segment_length[i]
            segment = output[start_idx:end_idx]
            # pad segment to max len
            segment = tf.pad(segment, [[self.max_len - self.segment_length[i], 0], [0, 0]])
            batch_output = batch_output.write(i, segment)
            mask_offset = mask_offset.write(i, self.max_len - self.segment_length[i])
            return [tf.add(i, 1), end_idx, batch_output, mask_offset]

        _, _, batch_output, mask_offset = tf.while_loop(while_condition, body,
                                                        [i, start_idx, batch_output, mask_offset])
        output = batch_output.stack()
        mask_offset = mask_offset.stack()
        self.mask_offset = mask_offset
        self.gnn_nuc_embedding = output

        # we have dummies padded to the front
        with tf.variable_scope('set2set_pooling'):
            output = lib.ops.LSTM.set2set_pooling('set2set_pooling', output, self.pool_steps, self.dropout_rate,
                                                  self.is_training_ph, True, mask_offset,
                                                  variables_on_cpu=False)

        self.bilstm_nuc_embedding = tf.get_collection('bilstm_nuc_emb')[0]
        # [batch_size, max_len, 2]
        self.gnn_nuc_output = lib.ops.Linear.linear('gnn_nuc_output', self.units, 2, self.gnn_nuc_embedding)
        self.bilstm_nuc_output = lib.ops.Linear.linear('bilstm_nuc_output', self.units * 2, 2,
                                                       self.bilstm_nuc_embedding)
        self.output = lib.ops.Linear.linear('OutputMapping', output.get_shape().as_list()[-1],
                                            2, output, variables_on_cpu=False)  # categorical logits

    def _loss(self):
        self.prediction = tf.nn.softmax(self.output)
        self.gnn_nuc_prediction = tf.nn.softmax(self.gnn_nuc_output)
        self.bilstm_nuc_prediction = tf.nn.softmax(self.bilstm_nuc_output)

        # graph level loss
        self.graph_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.output,  # reduce along the RNA sequence to a graph label
                labels=tf.one_hot(tf.reduce_max(self.labels, axis=1), depth=2),
            ))
        # nucleotide level loss
        # dummies are padded to the front...
        self.mask = 1.0 - tf.sequence_mask(self.mask_offset, maxlen=self.max_len, dtype=tf.float32)
        if self.use_ghm:
            self.gnn_nuc_cost = tf.reduce_sum(
                get_ghm_weights(self.gnn_nuc_prediction, self.labels, self.mask,
                                bins=10, alpha=0.75, name='GHM_GNN') * \
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.gnn_nuc_output,
                    labels=tf.one_hot(self.labels, depth=2),
                ) / tf.cast(tf.reduce_sum(self.segment_length), tf.float32)
            )
            self.bilstm_nuc_cost = tf.reduce_sum(
                get_ghm_weights(self.bilstm_nuc_prediction, self.labels, self.mask,
                                bins=10, alpha=0.75, name='GHM_BILSTM') * \
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.bilstm_nuc_output,
                    labels=tf.one_hot(self.labels, depth=2),
                ) / tf.cast(tf.reduce_sum(self.segment_length), tf.float32)
            )
        else:
            self.gnn_nuc_cost = tf.reduce_sum(
                self.mask * \
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.gnn_nuc_output,
                    labels=tf.one_hot(self.labels, depth=2),
                )) / tf.cast(tf.reduce_sum(self.segment_length), tf.float32)
            self.bilstm_nuc_cost = tf.reduce_sum(
                self.mask * \
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.bilstm_nuc_output,
                    labels=tf.one_hot(self.labels, depth=2),
                )) / tf.cast(tf.reduce_sum(self.segment_length), tf.float32)

        self.cost = self.mixing_ratio * self.graph_cost + (1. - self.mixing_ratio) * self.bilstm_nuc_cost

    def _train(self):
        self.gv = self.optimizer.compute_gradients(self.cost,
                                                   var_list=[var for var in tf.trainable_variables()],
                                                   colocate_gradients_with_ops=True)

    def _merge(self):
        # graph level accuracy
        self.seq_acc_val, self.seq_acc_update_op = tf.metrics.accuracy(
            labels=tf.reduce_max(self.labels, axis=-1),
            predictions=tf.to_int32(tf.argmax(self.prediction, axis=-1)),
        )

        # nucleotide level accuracy of precise matching on each location
        self.gnn_pos_acc_val, self.gnn_pos_acc_update_op = tf.metrics.accuracy(
            labels=self.segment_length,
            predictions=tf.reduce_sum(
                self.mask *
                tf.cast(
                    tf.equal(
                        tf.to_int32(tf.argmax(self.gnn_nuc_output, axis=-1)),
                        self.labels
                    ), tf.float32), axis=-1)  # along the RNA sequence
        )
        self.bilstm_pos_acc_val, self.bilstm_pos_acc_update_op = tf.metrics.accuracy(
            labels=self.segment_length,
            predictions=tf.reduce_sum(
                self.mask *
                tf.cast(
                    tf.equal(
                        tf.to_int32(tf.argmax(self.bilstm_nuc_output, axis=-1)),
                        self.labels
                    ), tf.float32), axis=-1)  # along the RNA sequence
        )

        # nucleotide level accuracy of containing a binding site
        self.gnn_nuc_acc_val, self.gnn_nuc_acc_update_op = tf.metrics.accuracy(
            labels=tf.reduce_max(self.labels, axis=-1),
            predictions=tf.to_int32(tf.reduce_max(
                tf.argmax(self.gnn_nuc_prediction, axis=-1), axis=-1)),
        )
        self.bilstm_nuc_acc_val, self.bilstm_nuc_acc_update_op = tf.metrics.accuracy(
            labels=tf.reduce_max(self.labels, axis=-1),
            predictions=tf.to_int32(tf.reduce_max(
                tf.argmax(self.bilstm_nuc_prediction, axis=-1), axis=-1)),
        )

        self.acc_val = [self.seq_acc_val, self.gnn_pos_acc_val, self.bilstm_pos_acc_val,
                        self.gnn_nuc_acc_val, self.bilstm_nuc_acc_val]
        self.acc_update_op = [self.seq_acc_update_op, self.gnn_pos_acc_update_op, self.bilstm_pos_acc_update_op,
                              self.gnn_nuc_acc_update_op, self.bilstm_nuc_acc_update_op]

        # graph level ROC AUC
        self.auc_val, self.auc_update_op = tf.metrics.auc(
            labels=tf.reduce_max(self.labels, axis=-1),
            predictions=self.prediction[:, 1],
        )

    def _init_session(self):
        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        if type(self.gpu_device) is list:
            gpu_options.visible_device_list = ','.join([device[-1] for device in self.gpu_device])
        else:
            gpu_options.visible_device_list = self.gpu_device[-1]
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init)
        self.sess.run(self.local_init)

    def reset_session(self):
        del self.saver
        with self.g.as_default():
            self.saver = tf.train.Saver(max_to_keep=10)
        self.sess.run(self.init)
        self.sess.run(self.local_init)
        lib.plot.reset()

    @classmethod
    def _merge_sparse_submatrices(cls, data, row_col, segments):
        '''
        merge sparse submatrices
        '''
        all_tensors = []
        for i in [0, 2]:  # forward_covalent, forward_hydrogen
            all_data, all_row_col = [], []
            size = 0
            for _data, _row_col, _segment in zip(data, row_col, segments):
                all_data.append(_data[i])
                all_row_col.append(np.array(_row_col[i]) + size)
                size += _segment
            all_tensors.append(
                tf.compat.v1.SparseTensorValue(
                    np.concatenate(all_row_col),
                    np.concatenate(all_data),
                    (size, size)
                )
            )
            # trick, transpose
            all_tensors.append(
                tf.compat.v1.SparseTensorValue(
                    np.concatenate(all_row_col)[:, [1, 0]],
                    np.concatenate(all_data),
                    (size, size)
                )
            )

        # return 4 matrices, one for each relation, max_len and segment_length
        return all_tensors

    @classmethod
    def indexing_iterable(cls, iterable, idx):
        return [item[idx] for item in iterable]

    def fit(self, X, y, epochs, batch_size, output_dir, logging=False, epoch_to_start=0):
        checkpoints_dir = os.path.join(output_dir, 'checkpoints/')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # split validation set
        row_sum = np.array(list(map(lambda label: np.sum(label), y)))
        pos_idx, neg_idx = np.where(row_sum > 0)[0], np.where(row_sum == 0)[0]

        dev_idx = np.array(list(np.random.choice(pos_idx, int(len(pos_idx) * 0.1), False)) + \
                           list(np.random.choice(neg_idx, int(len(neg_idx) * 0.1), False)))
        train_idx = np.delete(np.arange(len(y)), dev_idx)

        dev_data = self.indexing_iterable(X, dev_idx)
        dev_targets = y[dev_idx]

        X = self.indexing_iterable(X, train_idx)
        train_targets = y[train_idx]

        size_train = train_targets.shape[0]
        iters_per_epoch = size_train // batch_size + (0 if size_train % batch_size == 0 else 1)
        best_dev_cost = np.inf
        # best_dev_auc = 0.
        lib.plot.set_output_dir(output_dir)
        if logging:
            logger = lib.logger.CSVLogger('run.csv', output_dir,
                                          ['epoch', 'cost', 'graph_cost', 'gnn_nuc_cost', 'bilstm_nuc_cost',
                                           'seq_acc', 'gnn_pos_acc', 'bilstm_pos_acc', 'gnn_nuc_acc', 'bilstm_nuc_acc',
                                           'auc',
                                           'dev_cost', 'dev_graph_cost', 'dev_gnn_nuc_cost', 'dev_bilstm_nuc_cost',
                                           'dev_seq_acc',
                                           'dev_gnn_pos_acc', 'dev_bilstm_pos_acc', 'dev_gnn_nuc_acc',
                                           'dev_bilstm_nuc_acc', 'dev_auc'])

        for epoch in range(epoch_to_start, epochs):

            permute = np.random.permutation(size_train)
            node_tensor, all_rel_data, all_row_col, segment_length = self.indexing_iterable(X, permute)
            y = train_targets[permute]
            prepro_time = 0.
            training_time = 0.
            for i in range(iters_per_epoch):
                prepro_start = time.time()
                _node_tensor, _rel_data, _row_col, _segment, _labels \
                    = node_tensor[i * batch_size: (i + 1) * batch_size], \
                      all_rel_data[i * batch_size: (i + 1) * batch_size], \
                      all_row_col[i * batch_size: (i + 1) * batch_size], \
                      segment_length[i * batch_size: (i + 1) * batch_size], \
                      y[i * batch_size: (i + 1) * batch_size]

                _max_len = max(_segment)
                _labels = np.array([np.pad(label, [_max_len - len(label), 0], mode='constant') for label in _labels])
                all_adj_mat = self._merge_sparse_submatrices(_rel_data, _row_col, _segment)

                feed_dict = {
                    self.node_input_ph: np.concatenate(_node_tensor, axis=0),
                    **{self.adj_mat_ph[i]: all_adj_mat[i] for i in range(4)},
                    self.labels: _labels,
                    self.max_len: _max_len,
                    self.segment_length: _segment,
                    self.global_step: i + epoch * iters_per_epoch,
                    self.hf_iters_per_epoch: iters_per_epoch // 2,
                    self.is_training_ph: True,
                    # self.mixing_ratio_var: self.mixing_ratio if epoch < max(100, epochs // 2) else 0.2
                }
                prepro_end = time.time()
                prepro_time += (prepro_end - prepro_start)
                self.sess.run(self.train_op, feed_dict)
                training_time += (time.time() - prepro_end)
            print('preprocessing time: %.4f, training time: %.4f' % (prepro_time / (i + 1), training_time / (i + 1)))
            train_cost, train_acc, train_auc = self.evaluate(X, train_targets, batch_size)
            lib.plot.plot('train_cost', train_cost[0])
            lib.plot.plot('train_graph_cost', train_cost[1])
            lib.plot.plot('train_gnn_nuc_cost', train_cost[2])
            lib.plot.plot('train_bilstm_nuc_cost', train_cost[3])
            lib.plot.plot('train_seq_acc', train_acc[0])
            lib.plot.plot('train_gnn_pos_acc', train_acc[1])
            lib.plot.plot('train_bilstm_pos_acc', train_acc[2])
            lib.plot.plot('train_gnn_nuc_acc', train_acc[3])
            lib.plot.plot('train_bilstm_nuc_acc', train_acc[4])
            lib.plot.plot('train_auc', train_auc)

            dev_cost, dev_acc, dev_auc = self.evaluate(dev_data, dev_targets, batch_size)
            lib.plot.plot('dev_cost', dev_cost[0])
            lib.plot.plot('dev_graph_cost', dev_cost[1])
            lib.plot.plot('dev_gnn_nuc_cost', dev_cost[2])
            lib.plot.plot('dev_bilstm_nuc_cost', dev_cost[3])
            lib.plot.plot('dev_seq_acc', dev_acc[0])
            lib.plot.plot('dev_gnn_pos_acc', dev_acc[1])
            lib.plot.plot('dev_bilstm_pos_acc', dev_acc[2])
            lib.plot.plot('dev_gnn_nuc_acc', dev_acc[3])
            lib.plot.plot('dev_bilstm_nuc_acc', dev_acc[4])
            lib.plot.plot('dev_auc', dev_auc)

            logger.update_with_dict({
                'epoch': epoch, 'cost': train_cost[0], 'graph_cost': train_cost[1], 'gnn_nuc_cost': train_cost[2],
                'bilstm_nuc_cost': train_cost[3], 'seq_acc': train_acc[0], 'gnn_pos_acc': train_acc[1],
                'bilstm_pos_acc': train_acc[2], 'gnn_nuc_acc': train_acc[3], 'bilstm_nuc_acc': train_acc[4],
                'auc': train_auc,

                'dev_cost': dev_cost[0], 'dev_graph_cost': dev_cost[1], 'dev_gnn_nuc_cost': dev_cost[2],
                'dev_bilstm_nuc_cost': dev_cost[3], 'dev_seq_acc': dev_acc[0], 'dev_gnn_pos_acc': dev_acc[1],
                'dev_bilstm_pos_acc': dev_acc[2], 'dev_gnn_nuc_acc': dev_acc[3], 'dev_bilstm_nuc_acc': dev_acc[4],
                'dev_auc': dev_auc,
            })

            lib.plot.flush()
            lib.plot.tick()

            if dev_cost[0] < best_dev_cost and epoch - epoch_to_start >= 10:  # unstable loss in the beginning
                best_dev_cost = dev_cost[0]
                save_path = self.saver.save(self.sess, checkpoints_dir, global_step=epoch)
                print('Validation sample cost improved. Saved to path %s\n' % (save_path), flush=True)
            else:
                print('\n', flush=True)

        print('Loading best weights %s' % (save_path), flush=True)
        self.saver.restore(self.sess, save_path)
        if logging:
            logger.close()

    def evaluate(self, X, y, batch_size):
        node_tensor, all_rel_data, all_row_col, segment_length = X
        all_cost, all_graph_cost, all_gnn_nuc_cost, all_bilstm_nuc_cost = 0., 0., 0., 0.
        iters_per_epoch = len(node_tensor) // batch_size + (0 if len(node_tensor) % batch_size == 0 else 1)
        for i in range(iters_per_epoch):
            _node_tensor, _rel_data, _row_col, _segment, _labels \
                = node_tensor[i * batch_size: (i + 1) * batch_size], \
                  all_rel_data[i * batch_size: (i + 1) * batch_size], \
                  all_row_col[i * batch_size: (i + 1) * batch_size], \
                  segment_length[i * batch_size: (i + 1) * batch_size], \
                  y[i * batch_size: (i + 1) * batch_size]

            _max_len = max(_segment)
            _labels = np.array([np.pad(label, [_max_len - len(label), 0], mode='constant') for label in _labels])
            all_adj_mat = self._merge_sparse_submatrices(_rel_data, _row_col, _segment)

            feed_dict = {
                self.node_input_ph: np.concatenate(_node_tensor, axis=0),
                **{self.adj_mat_ph[i]: all_adj_mat[i] for i in range(4)},
                self.labels: _labels,
                self.max_len: _max_len,
                self.segment_length: _segment,
                self.is_training_ph: False
            }

            cost, graph_cost, gnn_nuc_cost, bilstm_nuc_cost, _, _ = self.sess.run(
                [self.cost, self.graph_cost, self.gnn_nuc_cost, self.bilstm_nuc_cost, self.acc_update_op,
                 self.auc_update_op], feed_dict)
            all_cost += cost * len(_node_tensor)
            all_graph_cost += graph_cost * len(_node_tensor)
            all_gnn_nuc_cost += gnn_nuc_cost * len(_node_tensor)
            all_bilstm_nuc_cost += bilstm_nuc_cost * len(_node_tensor)
        acc, auc = self.sess.run([self.acc_val, self.auc_val])
        self.sess.run(self.local_init)
        return (all_cost / len(node_tensor), all_graph_cost / len(node_tensor),
                all_gnn_nuc_cost / len(node_tensor), all_bilstm_nuc_cost / len(node_tensor)), acc, auc

    def predict(self, X, y=None):
        # predict one at a time without masking
        node_tensor, all_rel_data, all_row_col, segment_length = X
        all_predicton = []
        for i in range(len(node_tensor)):
            _node_tensor, _rel_data, _row_col, _segment \
                = node_tensor[i], \
                  all_rel_data[i], \
                  all_row_col[i], \
                  segment_length[i]

            all_adj_mat = self._merge_sparse_submatrices([_rel_data], [_row_col], [_segment])

            feed_dict = {
                self.node_input_ph: _node_tensor,
                **{self.adj_mat_ph[i]: all_adj_mat[i] for i in range(4)},
                self.max_len: _segment,
                self.segment_length: [_segment],
                self.is_training_ph: False
            }

            feed_tensor = [self.prediction]

            if y is not None:
                feed_dict[self.labels] = [y[i]]
                feed_tensor += [self.acc_update_op, self.auc_update_op]

            all_predicton.append(self.sess.run(feed_tensor, feed_dict)[0])
        all_predicton = np.concatenate(all_predicton, axis=0)

        if y is not None:
            acc, auc = self.sess.run([self.acc_val, self.auc_val])
            self.sess.run(self.local_init)
            return all_predicton, acc, auc
        else:
            return all_predicton

    def delete(self):
        tf.reset_default_graph()
        self.sess.close()

    def load(self, chkp_path):
        self.saver.restore(self.sess, chkp_path)
