import os
import sys
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from functools import partial
from sklearn.metrics import roc_auc_score
from . import _average_gradients, _stats

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.rgcn_utils import graph_convolution_layers
import lib.plot, lib.logger, lib.clr, lib.rna_utils
import lib.ops.LSTM, lib.ops.Linear, lib.ops.Conv1D


class RGCN:

    def __init__(self, max_len, node_dim, edge_dim, embedding_vec, gpu_device_list=['/gpu:0'], return_label=True,
                 **kwargs):
        self.max_len = max_len
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.embedding_vec = embedding_vec
        self.vocab_size = embedding_vec.shape[0]
        self.gpu_device_list = gpu_device_list
        self.return_label = return_label

        # hyperparams
        self.units = kwargs.get('units', 32)
        self.layers = kwargs.get('layers', 20)
        self.pool_steps = kwargs.get('pool_steps', 10)
        self.lstm_encoder = kwargs.get('lstm_encoder', True)
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.learning_rate = kwargs.get('learning_rate', 2e-4)
        self.use_clr = kwargs.get('use_clr', False)
        self.use_momentum = kwargs.get('use_momentum', False)

        self.reuse_weights = kwargs.get('reuse_weights', False)
        self.test_gated_nn = kwargs.get('test_gated_nn', False)

        self.use_conv = kwargs.get('use_conv', True)
        self.probabilistic = kwargs.get('probabilistic', True)

        self.g = tf.Graph()
        with self.g.as_default():
            with tf.device('/cpu:0'):
                self._placeholders()
                if self.use_momentum:
                    self.optimizer = tf.contrib.opt.MomentumWOptimizer(
                        1e-4, self.learning_rate * self.lr_multiplier,
                        0.9, use_nesterov=True
                    )
                else:
                    self.optimizer = tf.contrib.opt.AdamWOptimizer(
                        1e-4,
                        learning_rate=self.learning_rate * self.lr_multiplier
                    )

            for i, device in enumerate(self.gpu_device_list):
                # if we don't use name_scope, the operation id will automatically increment
                with tf.device('/gpu:%d' % (i)), tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE):
                    if self.test_gated_nn:
                        self._build_ggnn(i, mode='training')
                    else:
                        raise ValueError('non gated model has been excluded')
                        # self._build_rgcn(i, mode='training')
                    self._loss(i)
                    self._train(i)

            with tf.device('/cpu:0'):
                self._merge()
                self.train_op = self.optimizer.apply_gradients(self.gv)
                _stats('RGCN', self.gv)
                self.saver = tf.train.Saver(max_to_keep=10)
                self.init = tf.global_variables_initializer()
                self.local_init = tf.local_variables_initializer()
        self._init_session()

    def _placeholders(self):
        self.node_input_ph = tf.placeholder(tf.int32, shape=[None, self.max_len])
        self.node_input_splits = tf.split(self.node_input_ph, len(self.gpu_device_list))

        self.adj_mat_ph = tf.placeholder(tf.int32, shape=[None, self.max_len, self.max_len])
        self.adj_mat_splits = tf.split(tf.one_hot(self.adj_mat_ph, self.edge_dim), len(self.gpu_device_list))

        if self.return_label:
            self.labels = tf.placeholder(tf.int32, shape=[None])  # binary
        else:
            self.labels = tf.placeholder(tf.int32, shape=[None, self.max_len])
        self.labels_split = tf.split(self.labels, len(self.gpu_device_list))

        self.is_training_ph = tf.placeholder(tf.bool, ())
        self.global_step = tf.placeholder(tf.int32, ())
        self.hf_iters_per_epoch = tf.placeholder(tf.int32, ())
        if self.use_clr:
            self.lr_multiplier = lib.clr.cyclic_learning_rate(self.global_step, 0.5, 5.,
                                                              self.hf_iters_per_epoch, mode='exp_range')
        else:
            self.lr_multiplier = 1.

    def _build_ggnn(self, split_idx, mode):
        if mode == 'training':
            node_tensor = self.node_input_splits[split_idx]
            adj_tensor = self.adj_mat_splits[split_idx]
        else:
            raise ValueError('unknown mode')

        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding_layer', shape=(self.vocab_size, self.node_dim),
                                        initializer=tf.constant_initializer(self.embedding_vec), trainable=False)
        node_tensor = tf.nn.embedding_lookup(embedding, node_tensor)

        input_dim = self.node_dim

        if self.reuse_weights:
            if input_dim < self.units:
                node_tensor = tf.pad(node_tensor,
                                     [[0, 0], [0, 0], [0, self.units - self.node_dim]])
            elif input_dim > self.units:
                print('Changing \'self.units\' to %d!' % (input_dim))
                self.units = input_dim

        hidden_tensor = None
        with tf.variable_scope('gated-rgcn', reuse=tf.AUTO_REUSE):

            with tf.device('/cpu:0'):
                cell = tf.nn.rnn_cell.LSTMCell(self.units, name='lstm_cell')
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell,
                output_keep_prob=tf.cond(self.is_training_ph, lambda: 1 - self.dropout_rate, lambda: 1.)
                # keep prob
            )
            memory = None

            for i in range(self.layers):
                name = 'graph_convolution' if self.reuse_weights else 'graph_convolution_%d' % (i + 1)

                msg_tensor = graph_convolution_layers(name, (adj_tensor, hidden_tensor, node_tensor), self.units,
                                                      reuse=self.reuse_weights)
                msg_tensor = tf.nn.leaky_relu(msg_tensor)
                msg_tensor = tf.layers.dropout(msg_tensor, self.dropout_rate, training=self.is_training_ph)

                if hidden_tensor is None:  # hidden_state
                    state = tf.reshape(node_tensor, [-1, self.units])
                else:
                    state = tf.reshape(hidden_tensor, [-1, self.units])

                input = tf.reshape(msg_tensor, [-1, self.units])

                if i == 0:
                    memory = tf.zeros(tf.shape(state), tf.float32)
                new_state, (memory, _) = cell(input, tf.nn.rnn_cell.LSTMStateTuple(memory, state))

                hidden_tensor = tf.reshape(new_state, [-1, self.max_len, self.units])
                # [batch_size, length, u]

        output = tf.concat([hidden_tensor, node_tensor], axis=-1)

        if self.return_label:
            # globally pooling along the spatial axis of data,
            # arriving at a single feature vector for the whole graph
            if self.use_conv:
                with tf.variable_scope('seq_scan'):
                    output = lib.ops.Conv1D.conv1d('conv1', self.units * 2, self.units, 10, output, biases=False)
                    output = tf.nn.relu(output)
                    output = tf.layers.dropout(output, self.dropout_rate, training=self.is_training_ph)

                    output = lib.ops.Conv1D.conv1d('conv2', self.units, self.units, 10, output, biases=False)
                    output = tf.nn.relu(output)
                    output = tf.layers.dropout(output, self.dropout_rate, training=self.is_training_ph)

            with tf.variable_scope('set2set_pooling'):
                output = lib.ops.LSTM.set2set_pooling('set2set_pooling', output, self.pool_steps, self.dropout_rate,
                                                      self.is_training_ph, self.lstm_encoder)

        output = lib.ops.Linear.linear('OutputMapping', output.get_shape().as_list()[-1], 2,
                                       output)  # categorical logits

        if mode == 'training':
            if not hasattr(self, 'output'):
                self.output = [output]
            else:
                self.output += [output]

    def _loss(self, split_idx):
        prediction = tf.nn.softmax(self.output[split_idx])

        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.output[split_idx],
                labels=tf.one_hot(self.labels_split[split_idx], depth=2),
            ))

        if not hasattr(self, 'cost'):
            self.cost, self.prediction = [cost], [prediction]
        else:
            self.cost += [cost]
            self.prediction += [prediction]

    def _train(self, split_idx):
        gv = self.optimizer.compute_gradients(self.cost[split_idx],
                                              var_list=[var for var in tf.trainable_variables()],
                                              colocate_gradients_with_ops=True)

        if not hasattr(self, 'gv'):
            self.gv = [gv]
        else:
            self.gv += [gv]

    def _merge(self):
        # output, prediction, cost, acc, pears, gv
        self.output = tf.concat(self.output, axis=0)
        self.prediction = tf.concat(self.prediction, axis=0)
        self.cost = tf.add_n(self.cost) / len(self.gpu_device_list)
        self.gv = _average_gradients(self.gv)

        if self.return_label:
            self.acc_val, self.acc_update_op = tf.metrics.accuracy(
                labels=self.labels,
                predictions=tf.argmax(self.prediction, axis=-1),
            )
        else:
            self.acc_val, self.acc_update_op = tf.metrics.accuracy(
                labels=tf.ones(tf.shape(self.prediction)[0]),
                predictions=tf.reduce_prod(
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(self.prediction, axis=-1)),
                            self.labels
                        ), tf.float32), axis=-1)
            )

        self.auc_val, self.auc_update_op = tf.metrics.auc(
            labels=self.labels if self.return_label else tf.reshape(self.labels, [-1]),
            predictions=self.prediction[:, 1] if self.return_label else tf.reshape(self.prediction[:, :, 1], [-1]),
        )

    def _init_session(self):
        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        if type(self.gpu_device_list) is list:
            gpu_options.visible_device_list = ','.join([device[-1] for device in self.gpu_device_list])
        else:
            gpu_options.visible_device_list = self.gpu_device_list[-1]
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

    def pretrain(self, X, y, epochs, batch_size, output_dir, logging=False):
        checkpoints_dir = os.path.join(output_dir, 'checkpoints/')
        os.makedirs(checkpoints_dir)
        node_tensor, adj_mat = X
        train_rmd = node_tensor.shape[0] % len(self.gpu_device_list)
        if train_rmd != 0:
            node_tensor = node_tensor[:-train_rmd]
            adj_mat = adj_mat[:-train_rmd]
            y = y[:-train_rmd]
        train_data = [node_tensor, adj_mat]
        train_targets = y
        size_train = node_tensor.shape[0]
        iters_per_epoch = size_train // batch_size + (0 if size_train % batch_size == 0 else 1)
        lib.plot.set_output_dir(output_dir)
        if logging:
            logger = lib.logger.CSVLogger('pretrain.csv', output_dir, ['epoch', 'cost', 'acc', 'auc'])
        for epoch in range(epochs):
            permute = np.random.permutation(size_train)
            node_tensor = node_tensor[permute]
            adj_mat = adj_mat[permute]
            y = y[permute]
            start_time = time.time()
            for i in range(iters_per_epoch):
                _node_tensor, _adj_mat, _labels \
                    = node_tensor[i * batch_size: (i + 1) * batch_size], \
                      adj_mat[i * batch_size: (i + 1) * batch_size], \
                      y[i * batch_size: (i + 1) * batch_size]
                feed_dict = {
                    self.node_input_ph: _node_tensor,
                    self.adj_mat_ph: _adj_mat,
                    self.labels: _labels,
                    self.global_step: i,
                    self.hf_iters_per_epoch: iters_per_epoch // 2,
                    self.is_training_ph: True
                }
                self.sess.run(self.train_op, feed_dict)
            lib.plot.plot('training_time_per_iter', (time.time() - start_time) / iters_per_epoch)
            train_cost, train_acc, train_auc = self.evaluate(train_data, train_targets, batch_size)
            lib.plot.plot('train_cost', train_cost)
            lib.plot.plot('train_acc', train_acc)
            lib.plot.plot('train_auc', train_auc)
            if logging:
                logger.update_with_dict({
                    'epoch': epoch,
                    'cost': train_cost,
                    'acc': train_acc,
                    'auc': train_auc,
                })
            lib.plot.flush()
            lib.plot.tick()
        self.saver.save(self.sess, checkpoints_dir, global_step=epochs)
        if logging:
            logger.close()

    def evaluate(self, X, y, batch_size):
        '''non stochastic'''
        node_tensor, adj_mat = X
        all_cost = 0.
        iters_per_epoch = len(node_tensor) // batch_size + (0 if len(node_tensor) % batch_size == 0 else 1)
        for i in range(iters_per_epoch):
            _node_tensor, _adj_mat, _labels \
                = node_tensor[i * batch_size: (i + 1) * batch_size], \
                  adj_mat[i * batch_size: (i + 1) * batch_size], \
                  y[i * batch_size: (i + 1) * batch_size]
            feed_dict = {self.node_input_ph: _node_tensor,
                         self.adj_mat_ph: _adj_mat,
                         self.labels: _labels,
                         self.is_training_ph: False}

            cost, _, _ = self.sess.run([self.cost, self.acc_update_op, self.auc_update_op], feed_dict)
            all_cost += cost * len(_node_tensor)
        acc, auc = self.sess.run([self.acc_val, self.auc_val])
        self.sess.run(self.local_init)
        return all_cost / len(node_tensor), acc, auc

    def train_sampled_structures(self, node_tensor, y, epochs, batch_size, output_dir, pool, graph_passes=10,
                                 forward_passes=100):

        # split validation set
        if self.return_label:
            pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
        else:
            pos_idx, neg_idx = np.where(np.count_nonzero(y, axis=-1) > 0)[0], \
                               np.where(np.count_nonzero(y, axis=-1) == 0)[0]

        dev_idx = np.array(list(np.random.choice(pos_idx, int(len(pos_idx) * 0.1), False)) + \
                           list(np.random.choice(neg_idx, int(len(neg_idx) * 0.1), False)))
        train_idx = np.delete(np.arange(len(y)), dev_idx)

        dev_node_tensor = node_tensor[dev_idx]
        dev_targets = y[dev_idx]
        node_tensor = node_tensor[train_idx]
        y = y[train_idx]
        np.save('dev_targets.npy', dev_targets)

        train_rmd = node_tensor.shape[0] % len(self.gpu_device_list)
        if train_rmd != 0:
            node_tensor = node_tensor[:-train_rmd]
            y = y[:-train_rmd]

        dev_rmd = dev_node_tensor.shape[0] % len(self.gpu_device_list)
        if dev_rmd != 0:
            dev_node_tensor = dev_node_tensor[:-dev_rmd]
            dev_targets = dev_targets[:-train_rmd]

        seqs = [''.join(['PACGTN'[c] for c in seq]) for seq in node_tensor]
        dev_seqs = [''.join(['PACGTN'[c] for c in seq]) for seq in dev_node_tensor]

        size_train = node_tensor.shape[0]
        iters_per_epoch = size_train // batch_size + (0 if size_train % batch_size == 0 else 1)
        lib.plot.set_output_dir(output_dir)

        logger = lib.logger.CSVLogger('run.csv', output_dir, ['epoch', 'train_acc', 'train_auc', 'dev_acc', 'dev_auc'])

        print('Begin stochastic training')
        for epoch in range(epochs):
            print('stochastic training, epoch: %d' % (epoch))

            print('sampling graphs')
            sample_one_seq = partial(lib.rna_utils.sample_one_seq, passes=graph_passes)
            all_train_adj_mat = np.stack(list(tqdm(pool.imap(sample_one_seq, seqs))), axis=1)
            all_dev_adj_mat = np.stack(list(tqdm(pool.imap(sample_one_seq, dev_seqs))), axis=1)

            for graph_iter in tqdm(range(graph_passes)):  # epochs * graph_passes
                adj_mat = all_train_adj_mat[graph_iter]

                permute = np.random.permutation(size_train)
                shuffled_node_tensor = node_tensor[permute]
                shuffled_adj_mat = adj_mat[permute]
                shuffled_y = y[permute]

                for i in range(iters_per_epoch):
                    _node_tensor, _adj_mat, _labels \
                        = shuffled_node_tensor[i * batch_size: (i + 1) * batch_size], \
                          shuffled_adj_mat[i * batch_size: (i + 1) * batch_size], \
                          shuffled_y[i * batch_size: (i + 1) * batch_size]
                    feed_dict = {
                        self.node_input_ph: _node_tensor,
                        self.adj_mat_ph: _adj_mat,
                        self.labels: _labels,
                        self.global_step: i,
                        self.hf_iters_per_epoch: iters_per_epoch // 2,
                        self.is_training_ph: True
                    }
                    self.sess.run(self.train_op, feed_dict)

            # at the end of each epoch, evaluate
            _, _, train_auc, train_acc = \
                self.evaluate_stochastic(node_tensor, y,
                                         batch_size * 5 * len(self.gpu_device_list),
                                         pool, graph_passes, forward_passes,
                                         adj_mat_list=all_train_adj_mat)
            dev_preds_mean, dev_preds_variance, dev_auc, dev_acc = \
                self.evaluate_stochastic(dev_node_tensor, dev_targets,
                                         batch_size * 5 * len(self.gpu_device_list),
                                         pool, graph_passes, forward_passes,
                                         adj_mat_list=all_dev_adj_mat)
            np.save('dev_preds_mean_%d.npy' % (epoch), dev_preds_mean)
            np.save('dev_preds_variance_%d.npy' % (epoch), dev_preds_variance)
            logger.update_with_dict({
                'epoch': epoch,
                'train_auc': train_auc,
                'train_acc': train_acc,
                'dev_auc': dev_auc,
                'dev_acc': dev_acc,
            })
        logger.close()

    def evaluate_stochastic(self, node_tensor, y, batch_size, pool, graph_passes, forward_passes, adj_mat_list=None):
        seqs = [''.join(['PACGTN'[c] for c in seq]) for seq in node_tensor]
        if adj_mat_list is None:
            sample_one_seq = partial(lib.rna_utils.sample_one_seq, passes=graph_passes)
            adj_mat_list = np.stack(list(pool.imap(sample_one_seq, seqs)), axis=0)
        all_preds_mean, all_preds_sq_mean = [], []
        for i in range(graph_passes):
            adj_mat = adj_mat_list[i]
            preds_mean, preds_sq_mean = self.MC_dropout((node_tensor, adj_mat), batch_size, forward_passes)
            all_preds_mean.append(preds_mean)
            all_preds_sq_mean.append(preds_sq_mean)
        averaged_preds = np.stack(all_preds_mean, -1).mean(-1)
        averaged_variance = np.stack(all_preds_sq_mean, -1).mean(-1) - averaged_preds ** 2
        auc = roc_auc_score(y, averaged_preds[:, 1])
        acc = np.mean(averaged_preds.argmax(axis=-1) == y)
        return averaged_preds, averaged_variance, auc, acc

    def MC_dropout(self, X, batch_size, T=100):
        print('Monte Carlo dropout')
        node_tensor, adj_mat = X
        preds_all_passes = []
        iters_per_epoch = len(node_tensor) // batch_size + (0 if len(node_tensor) % batch_size == 0 else 1)
        for t in tqdm(range(T)):
            all_preds = []
            for i in range(iters_per_epoch):
                _node_tensor, _adj_mat \
                    = node_tensor[i * batch_size: (i + 1) * batch_size], \
                      adj_mat[i * batch_size: (i + 1) * batch_size]
                feed_dict = {
                    self.node_input_ph: _node_tensor,
                    self.adj_mat_ph: _adj_mat,
                    self.is_training_ph: True
                }
                all_preds.append(self.sess.run(self.prediction, feed_dict))
            preds_all_passes.append(np.concatenate(all_preds, axis=0))
        # all passes w.r.t one graph
        preds_all_passes = np.stack(preds_all_passes, axis=-1)
        return np.mean(preds_all_passes, axis=-1), np.mean(preds_all_passes ** 2, axis=-1)

    def delete(self):
        tf.reset_default_graph()
        self.sess.close()

    def load(self, chkp_path):
        self.saver.restore(self.sess, chkp_path)
