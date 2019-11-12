# hierarchical model
import os
import sys
import time
import numpy as np
import tensorflow as tf
import threading
from queue import Queue

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Model import _stats
from lib.rgcn_utils import joint_layer, normalize
import lib.plot, lib.logger, lib.clr
import lib.ops.LSTM, lib.ops.Linear, lib.ops.Conv1D
from lib.tf_ghm_loss import get_ghm_weights
from lib.AMSGrad import AMSGrad


class BackgroundGenerator(threading.Thread):
    def __init__(self, X, y, batch_size, **kwargs):
        threading.Thread.__init__(self)
        self.X = X
        self.y = y
        self.size_train = y.shape[0]
        self.batch_size = batch_size
        self.iters_per_epoch = self.size_train // batch_size + (0 if self.size_train % batch_size == 0 else 1)

        self.queue = Queue(self.iters_per_epoch)
        self.daemon = True
        self.kill = threading.Event()
        self.start()

    def run(self):
        while True:
            permute = np.random.permutation(self.size_train)
            node_tensor, all_rel_data, all_row_col, segment_length, raw_seq = JointAdaModel.indexing_iterable(
                self.X, permute)
            y = self.y[permute]
            for i in range(self.iters_per_epoch):
                _node_tensor, _rel_data, _row_col, _segment, _labels \
                    = node_tensor[i * self.batch_size: (i + 1) * self.batch_size], \
                      all_rel_data[i * self.batch_size: (i + 1) * self.batch_size], \
                      all_row_col[i * self.batch_size: (i + 1) * self.batch_size], \
                      segment_length[i * self.batch_size: (i + 1) * self.batch_size], \
                      y[i * self.batch_size: (i + 1) * self.batch_size]

                _max_len = max(_segment)
                _mask_offset = np.array([_max_len - _seg for _seg in _segment])
                _node_tensor = np.array(
                    [np.pad(seq, [_max_len - len(seq), 0], mode='constant') for seq in _node_tensor])
                _labels = np.array([np.pad(label, [_max_len - len(label), 0], mode='constant') for label in _labels])
                all_adj_mat = JointAdaModel._merge_sparse_submatrices(_rel_data, _row_col, _segment)
                self.queue.put([_node_tensor, _mask_offset, all_adj_mat, _labels])
                if self.kill.is_set():
                    return

    def next(self):
        next_item = self.queue.get()
        return next_item


class JointAdaModel:

    def __init__(self, node_dim, embedding_vec, gpu_device, **kwargs):
        self.node_dim = node_dim
        self.embedding_vec = embedding_vec
        self.vocab_size = embedding_vec.shape[0]
        self.gpu_device = gpu_device
        # hyperparams
        self.units = kwargs.get('units', 32)
        self.layers = kwargs.get('layers', 10)
        self.pool_steps = kwargs.get('pool_steps', 10)
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.learning_rate = kwargs.get('learning_rate', 2e-4)
        self.use_clr = kwargs.get('use_clr', False)
        self.use_momentum = kwargs.get('use_momentum', False)
        self.use_bn = kwargs.get('use_bn', False)

        self.reuse_weights = kwargs.get('reuse_weights', False)
        self.lstm_ggnn = kwargs.get('lstm_ggnn', False)
        self.probabilistic = kwargs.get('probabilistic', True)
        self.use_attention = kwargs.get('use_attention', False)

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
                self.saver = tf.train.Saver(max_to_keep=5)
                self.init = tf.global_variables_initializer()
                self.local_init = tf.local_variables_initializer()
        self._init_session()

    def _placeholders(self):
        self.node_input_ph = tf.placeholder(tf.int32, shape=[None, None, ])  # batch_size x nb_nodes
        self.adj_mat_ph = tf.sparse_placeholder(tf.float32, shape=[None, None, None])
        # batch_size x nb_nodes x nb_nodes

        self.labels = tf.placeholder(tf.int32, shape=[None, None, ])  # batch_size x nb_nodes
        self.mask_offset = tf.placeholder(tf.int32, shape=[None, ])  # batch_size

        self.is_training_ph = tf.placeholder(tf.bool, ())
        self.global_step = tf.placeholder(tf.int32, ())
        self.hf_iters_per_epoch = tf.placeholder(tf.int32, ())
        if self.use_clr:
            print('using cyclic learning rate')
            self.lr_multiplier = lib.clr. \
                cyclic_learning_rate(self.global_step, 0.5, 5.,
                                     self.hf_iters_per_epoch, mode='exp_range')
        else:
            print('using constant learning rate')
            self.lr_multiplier = 1.

    def _build_ggnn(self):
        embedding = tf.get_variable('embedding_layer', shape=(self.vocab_size, self.node_dim),
                                    initializer=tf.constant_initializer(self.embedding_vec), trainable=False)
        node_tensor = tf.nn.embedding_lookup(embedding, self.node_input_ph)
        self.node_tensor = node_tensor
        if self.reuse_weights:
            if self.node_dim < self.units:
                node_tensor = tf.pad(node_tensor,
                                     [[0, 0], [0, 0], [0, self.units - self.node_dim]])
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

            adj_mat = tf.sparse.to_dense(self.adj_mat_ph, validate_indices=False)

            for i in range(self.layers):
                name = 'joint_convolutional' if self.reuse_weights else 'joint_convolutional_%d' % (i + 1)
                # variables for sparse implementation default placement to gpu
                msg_tensor = joint_layer(name, (adj_mat, hidden_tensor, node_tensor),
                                         self.units, reuse=self.reuse_weights, batch_axis=True,
                                         use_attention=self.use_attention)
                msg_tensor = normalize('Norm' if self.reuse_weights else 'Norm%d' % (i + 1),
                                       msg_tensor, self.use_bn, self.is_training_ph)
                msg_tensor = tf.nn.leaky_relu(msg_tensor)
                msg_tensor = tf.layers.dropout(msg_tensor, self.dropout_rate, training=self.is_training_ph)

                # reshaping msg_tensor to two dimensions
                original_shape = tf.shape(msg_tensor)  # batch_size, nb_nodes, units
                msg_tensor = tf.reshape(msg_tensor, (-1, self.units))

                if hidden_tensor is None:  # hidden_state
                    state = node_tensor
                else:
                    state = hidden_tensor

                state = tf.reshape(state, (-1, self.units))

                if self.lstm_ggnn:
                    if i == 0:
                        memory = tf.zeros(tf.shape(state), tf.float32)
                    # state becomes the hidden tensor
                    hidden_tensor, (memory, _) = cell(msg_tensor, tf.nn.rnn_cell.LSTMStateTuple(memory, state))
                else:
                    hidden_tensor, _ = cell(msg_tensor, state)
                # [batch_size, length, u]
                hidden_tensor = tf.reshape(hidden_tensor, original_shape)
                hidden_tensor *= (1. - tf.sequence_mask(self.mask_offset, maxlen=original_shape[1], dtype=tf.float32))[
                                 :, :, None]
        # the original nucleotide embeddings learnt by GNN
        output = hidden_tensor
        self.gnn_embedding = output

        # we have dummies padded to the front
        with tf.variable_scope('set2set_pooling'):
            output = lib.ops.LSTM.set2set_pooling('set2set_pooling', output, self.pool_steps, self.dropout_rate,
                                                  self.is_training_ph, True, self.mask_offset,
                                                  variables_on_cpu=False)

        self.bilstm_embedding = tf.get_collection('nuc_emb')[0]
        # [batch_size, max_len, 2]
        self.gnn_output = lib.ops.Linear.linear('gnn_nuc_output', self.units, 2, self.gnn_embedding)
        self.bilstm_output = lib.ops.Linear.linear('bilstm_nuc_output', self.units * 2, 2,
                                                   self.bilstm_embedding)
        self.output = lib.ops.Linear.linear('OutputMapping', output.get_shape().as_list()[-1],
                                            2, output, variables_on_cpu=False)  # categorical logits

    def _loss(self):
        self.prediction = tf.nn.softmax(self.output)
        self.gnn_prediction = tf.nn.softmax(self.gnn_output)
        self.bilstm_prediction = tf.nn.softmax(self.bilstm_output)

        # graph level loss
        self.graph_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.output,  # reduce along the RNA sequence to a graph label
                labels=tf.one_hot(tf.reduce_max(self.labels, axis=1), depth=2),
            ))
        # nucleotide level loss
        # dummies are padded to the front...
        self.mask = 1.0 - tf.sequence_mask(self.mask_offset, maxlen=tf.shape(self.labels)[1], dtype=tf.float32)
        if self.use_ghm:
            self.gnn_cost = tf.reduce_sum(
                get_ghm_weights(self.gnn_prediction, self.labels, self.mask,
                                bins=10, alpha=0.75, name='GHM_GNN') * \
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.gnn_output,
                    labels=tf.one_hot(self.labels, depth=2),
                ) / tf.cast(tf.reduce_sum(self.mask), tf.float32)
            )
            self.bilstm_cost = tf.reduce_sum(
                get_ghm_weights(self.bilstm_prediction, self.labels, self.mask,
                                bins=10, alpha=0.75, name='GHM_BILSTM') * \
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.bilstm_output,
                    labels=tf.one_hot(self.labels, depth=2),
                ) / tf.cast(tf.reduce_sum(self.mask), tf.float32)
            )
        else:
            self.gnn_cost = tf.reduce_sum(
                self.mask * \
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.gnn_output,
                    labels=tf.one_hot(self.labels, depth=2),
                )) / tf.cast(tf.reduce_sum(self.mask), tf.float32)
            self.bilstm_cost = tf.reduce_sum(
                self.mask * \
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.bilstm_output,
                    labels=tf.one_hot(self.labels, depth=2),
                )) / tf.cast(tf.reduce_sum(self.mask), tf.float32)

        self.cost = self.mixing_ratio * self.graph_cost + (1. - self.mixing_ratio) * self.bilstm_cost

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

        # nucleotide level accuracy of containing a binding site
        self.gnn_acc_val, self.gnn_acc_update_op = tf.metrics.accuracy(
            labels=tf.reduce_max(self.labels, axis=-1),
            predictions=tf.to_int32(tf.reduce_max(
                tf.argmax(self.gnn_prediction, axis=-1), axis=-1)),
        )
        self.bilstm_acc_val, self.bilstm_acc_update_op = tf.metrics.accuracy(
            labels=tf.reduce_max(self.labels, axis=-1),
            predictions=tf.to_int32(tf.reduce_max(
                tf.argmax(self.bilstm_prediction, axis=-1), axis=-1)),
        )

        self.acc_val = [self.seq_acc_val, self.gnn_acc_val, self.bilstm_acc_val]
        self.acc_update_op = [self.seq_acc_update_op, self.gnn_acc_update_op, self.bilstm_acc_update_op]

        # graph level ROC AUC
        self.auc_val, self.auc_update_op = tf.metrics.auc(
            labels=tf.reduce_max(self.labels, axis=-1),
            predictions=self.prediction[:, 1],
        )

        self.g_nodes = tf.gradients(self.prediction[:, 1], self.node_tensor)[0]

    def _init_session(self):
        gpu_options = tf.GPUOptions()
        gpu_options.per_process_gpu_memory_fraction = 0.3
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
            self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(self.init)
        self.sess.run(self.local_init)
        lib.plot.reset()

    @classmethod
    def _merge_sparse_submatrices(cls, data, row_col, segments):
        '''
        merge sparse submatrices to 3 dimensional sparse tensor
        take note that padding has to be made in the beginning of each submatrix
        '''

        all_data, all_row_col = [], []
        max_size = np.max(segments)
        for i, (_data, _row_col, _segment) in enumerate(zip(data, row_col, segments)):
            all_data.append(_data[2])
            all_data.append(_data[3])
            all_row_col.append(np.concatenate([(np.ones((len(_row_col[2]), 1)) * i).astype(np.int32),
                                               (np.array(_row_col[2]) + max_size - _segment).reshape(-1, 2)], axis=-1))
            all_row_col.append(np.concatenate([(np.ones((len(_row_col[3]), 1)) * i).astype(np.int32),
                                               (np.array(_row_col[3]) + max_size - _segment).reshape(-1, 2)], axis=-1))

        return tf.compat.v1.SparseTensorValue(
            np.concatenate(all_row_col),
            np.concatenate(all_data),
            (len(segments), max_size, max_size)
        )

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

        best_dev_cost = np.inf
        # best_dev_auc = 0.
        lib.plot.set_output_dir(output_dir)
        if logging:
            logger = lib.logger.CSVLogger('run.csv', output_dir,
                                          ['epoch', 'cost', 'graph_cost', 'gnn_cost', 'bilstm_cost',
                                           'seq_acc', 'gnn_acc', 'bilstm_acc', 'auc',
                                           'dev_cost', 'dev_graph_cost', 'dev_gnn_cost', 'dev_bilstm_cost',
                                           'dev_seq_acc', 'dev_gnn_acc', 'dev_bilstm_acc', 'dev_auc'])

        train_generator = BackgroundGenerator(X, train_targets, batch_size, random_crop=False)
        val_generator = BackgroundGenerator(dev_data, dev_targets, batch_size)
        iters_per_epoch = train_generator.iters_per_epoch

        for epoch in range(epoch_to_start, epochs):

            prepro_time = 0.
            training_time = 0.
            for i in range(iters_per_epoch):
                prepro_start = time.time()
                _node_tensor, _mask_offset, all_adj_mat, _labels = train_generator.next()
                feed_dict = {
                    self.node_input_ph: _node_tensor,
                    self.adj_mat_ph: all_adj_mat,
                    self.labels: _labels,
                    self.mask_offset: _mask_offset,
                    self.global_step: i + epoch * iters_per_epoch,
                    self.hf_iters_per_epoch: iters_per_epoch // 2,
                    self.is_training_ph: True,
                }
                prepro_end = time.time()
                prepro_time += (prepro_end - prepro_start)
                self.sess.run(self.train_op, feed_dict)
                training_time += (time.time() - prepro_end)
            print('preprocessing time: %.4f, training time: %.4f' % (prepro_time / (i + 1), training_time / (i + 1)))
            train_cost, train_acc, train_auc = self.evaluate_with_generator(train_generator)
            lib.plot.plot('train_cost', train_cost[0])
            lib.plot.plot('train_graph_cost', train_cost[1])
            lib.plot.plot('train_gnn_cost', train_cost[2])
            lib.plot.plot('train_bilstm_cost', train_cost[3])
            lib.plot.plot('train_seq_acc', train_acc[0])
            lib.plot.plot('train_gnn_acc', train_acc[1])
            lib.plot.plot('train_bilstm_acc', train_acc[2])
            lib.plot.plot('train_auc', train_auc)

            dev_cost, dev_acc, dev_auc = self.evaluate_with_generator(val_generator)
            lib.plot.plot('dev_cost', dev_cost[0])
            lib.plot.plot('dev_graph_cost', dev_cost[1])
            lib.plot.plot('dev_gnn_cost', dev_cost[2])
            lib.plot.plot('dev_bilstm_cost', dev_cost[3])
            lib.plot.plot('dev_seq_acc', dev_acc[0])
            lib.plot.plot('dev_gnn_acc', dev_acc[1])
            lib.plot.plot('dev_bilstm_acc', dev_acc[2])
            lib.plot.plot('dev_auc', dev_auc)

            logger.update_with_dict({
                'epoch': epoch, 'cost': train_cost[0], 'graph_cost': train_cost[1], 'gnn_cost': train_cost[2],
                'bilstm_cost': train_cost[3], 'seq_acc': train_acc[0], 'gnn_acc': train_acc[1],
                'bilstm_acc': train_acc[2], 'auc': train_auc,

                'dev_cost': dev_cost[0], 'dev_graph_cost': dev_cost[1], 'dev_gnn_cost': dev_cost[2],
                'dev_bilstm_cost': dev_cost[3], 'dev_seq_acc': dev_acc[0], 'dev_gnn_acc': dev_acc[1],
                'dev_bilstm_acc': dev_acc[2], 'dev_auc': dev_auc,
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
        train_generator.kill.set()
        val_generator.kill.set()
        train_generator.next()
        val_generator.next()
        train_generator.join()
        val_generator.join()

    def evaluate_with_generator(self, generator):
        all_cost, all_graph_cost, all_gnn_nuc_cost, all_bilstm_nuc_cost = 0., 0., 0., 0.
        for i in range(generator.iters_per_epoch):
            _node_tensor, _mask_offset, all_adj_mat, _labels = generator.next()

            feed_dict = {
                self.node_input_ph: _node_tensor,
                self.adj_mat_ph: all_adj_mat,
                self.labels: _labels,
                self.mask_offset: _mask_offset,
                self.is_training_ph: False
            }
            cost, graph_cost, gnn_nuc_cost, bilstm_nuc_cost, _, _ = self.sess.run(
                [self.cost, self.graph_cost, self.gnn_cost, self.bilstm_cost, self.acc_update_op,
                 self.auc_update_op], feed_dict)
            all_cost += cost * _labels.shape[0]
            all_graph_cost += graph_cost * _labels.shape[0]
            all_gnn_nuc_cost += gnn_nuc_cost * _labels.shape[0]
            all_bilstm_nuc_cost += bilstm_nuc_cost * _labels.shape[0]
        acc, auc = self.sess.run([self.acc_val, self.auc_val])
        self.sess.run(self.local_init)
        return (all_cost / generator.size_train, all_graph_cost / generator.size_train,
                all_gnn_nuc_cost / generator.size_train, all_bilstm_nuc_cost / generator.size_train), acc, auc

    def evaluate(self, X, y, batch_size):
        node_tensor, all_rel_data, all_row_col, segment_length, raw_seq = X
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
            _mask_offset = np.array([_max_len - _seg for _seg in _segment])
            _node_tensor = np.array([np.pad(seq, [_max_len - len(seq), 0], mode='constant') for seq in _node_tensor])
            _labels = np.array([np.pad(label, [_max_len - len(label), 0], mode='constant') for label in _labels])
            all_adj_mat = self._merge_sparse_submatrices(_rel_data, _row_col, _segment)

            feed_dict = {
                self.node_input_ph: _node_tensor,
                self.adj_mat_ph: all_adj_mat,
                self.labels: _labels,
                self.mask_offset: _mask_offset,
                self.is_training_ph: False
            }
            cost, graph_cost, gnn_nuc_cost, bilstm_nuc_cost, _, _ = self.sess.run(
                [self.cost, self.graph_cost, self.gnn_cost, self.bilstm_cost, self.acc_update_op,
                 self.auc_update_op], feed_dict)
            all_cost += cost * len(_node_tensor)
            all_graph_cost += graph_cost * len(_node_tensor)
            all_gnn_nuc_cost += gnn_nuc_cost * len(_node_tensor)
            all_bilstm_nuc_cost += bilstm_nuc_cost * len(_node_tensor)
        acc, auc = self.sess.run([self.acc_val, self.auc_val])
        self.sess.run(self.local_init)
        return (all_cost / len(node_tensor), all_graph_cost / len(node_tensor),
                all_gnn_nuc_cost / len(node_tensor), all_bilstm_nuc_cost / len(node_tensor)), acc, auc

    def predict(self, X, batch_size):
        node_tensor, all_rel_data, all_row_col, segment_length, raw_seq = X
        preds = []
        iters_per_epoch = len(node_tensor) // batch_size + (0 if len(node_tensor) % batch_size == 0 else 1)
        for i in range(iters_per_epoch):
            _node_tensor, _rel_data, _row_col, _segment \
                = node_tensor[i * batch_size: (i + 1) * batch_size], \
                  all_rel_data[i * batch_size: (i + 1) * batch_size], \
                  all_row_col[i * batch_size: (i + 1) * batch_size], \
                  segment_length[i * batch_size: (i + 1) * batch_size]

            _max_len = max(_segment)
            _mask_offset = np.array([_max_len - _seg for _seg in _segment])
            _node_tensor = np.array([np.pad(seq, [_max_len - len(seq), 0], mode='constant') for seq in _node_tensor])
            all_adj_mat = self._merge_sparse_submatrices(_rel_data, _row_col, _segment)

            feed_dict = {
                self.node_input_ph: _node_tensor,
                self.adj_mat_ph: all_adj_mat,
                self.mask_offset: _mask_offset,
                self.is_training_ph: False
            }
            preds.append(self.sess.run(self.prediction, feed_dict))

        return np.concatenate(np.array(preds), axis=0)

    def integrated_gradients(self, X, y, ids, interp_steps=100, save_path=None, max_plots=np.inf):
        counter = 0
        for _node_tensor, _rel_data, _row_col, _segment, _, _label, _id in zip(*X, y, ids):
            if np.max(_label) == 0:
                continue
            if counter >= max_plots:
                break
            _meshed_node_tensor = np.array([self.embedding_vec[idx] for idx in _node_tensor])
            _meshed_reference_input = np.zeros_like(_meshed_node_tensor)
            new_node_tensor = []
            for i in range(0, interp_steps + 1):
                new_node_tensor.append(
                    _meshed_reference_input + i / interp_steps * (_meshed_node_tensor - _meshed_reference_input))
            all_adj_mat = self._merge_sparse_submatrices([_rel_data] * (interp_steps + 1),
                                                         [_row_col] * (interp_steps + 1),
                                                         [_segment] * (interp_steps + 1))

            feed_dict = {
                self.node_tensor: np.array(new_node_tensor),
                self.adj_mat_ph: all_adj_mat,
                self.mask_offset: np.zeros((interp_steps + 1,), np.int32),
                self.is_training_ph: False
            }

            grads = self.sess.run(self.g_nodes, feed_dict).reshape((interp_steps + 1, _segment, 4))
            grads = (grads[:-1] + grads[1:]) / 2.0
            node_scores = np.average(grads, axis=0) * (_meshed_node_tensor - _meshed_reference_input)

            pos_idx = np.where(_label == 1)[0]
            extended_start = max(pos_idx[0] - 50, 0)
            extended_end = min(pos_idx[-1] + 50, _segment)
            extended_region = [extended_start, extended_end]
            viewpoint_region = [pos_idx[0] - extended_start, pos_idx[-1] - extended_start + 1]

            if save_path is not None:
                saveto = os.path.join(save_path, '%s.jpg' % (_id))
            else:
                saveto = None
            lib.plot.plot_weights(node_scores[range(*extended_region)],
                                  subticks_frequency=10, highlight={'r': [viewpoint_region]},
                                  save_path=saveto)
            counter += 1

    def delete(self):
        tf.reset_default_graph()
        self.sess.close()

    def load(self, chkp_path):
        self.saver.restore(self.sess, chkp_path)


if __name__ == "__main__":
    data = [[None, None, [1, 2], [3]], [None, None, [4, 5], []], [None, None, [6], []]]
    row_col = [[None, None, [[0, 0], [1, 1]], [[2, 2]]], [None, None, [[0, 0], [5, 5]], []], [None, None, [[6, 6]], []]]
    segments = [3, 6, 7]
    sess = tf.Session()
    sp_ph = tf.sparse_placeholder(tf.float32, shape=[None, None, None])
    print(sess.run(sp_ph, {sp_ph: JointAdaModel._merge_sparse_submatrices(data, row_col, segments)}))
