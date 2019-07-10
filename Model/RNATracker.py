import os
import sys
import numpy as np
import tensorflow as tf
from . import _average_gradients, _stats

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import lib.plot, lib.logger, lib.clr
import lib.ops.LSTM, lib.ops.Linear
from lib.resutils import normalize


class RNATracker:

    def __init__(self, max_len, node_dim, edge_dim, gpu_device_list=['/gpu:0'], **kwargs):
        self.max_len = max_len
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.gpu_device_list = gpu_device_list

        # hyperparams
        self.units = kwargs.get('units', 128)
        self.pool_steps = kwargs.get('pool_steps', 10)
        self.lstm_encoder = kwargs.get('lstm_encoder', True)
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.learning_rate = kwargs.get('learning_rate', 2e-4)
        self.use_clr = kwargs.get('use_clr', False)
        self.use_momentum = kwargs.get('use_momentum', False)
        self.use_bn = kwargs.get('use_bn', False)

        self.g = tf.get_default_graph()
        with self.g.as_default():
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
                with tf.device(device), tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE):
                    self._build_rnatracker(i, mode='training')
                    self._loss(i)
                    self._train(i)

            with tf.device(self.gpu_device_list[0]), tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE):
                self._build_rnatracker(None, mode='inference')

            self._merge()
            self.train_op = self.optimizer.apply_gradients(self.gv)
            _stats('RNATracker', self.gv)
            self.saver = tf.train.Saver(max_to_keep=1000)
            self.init = tf.global_variables_initializer()
            self.local_init = tf.local_variables_initializer()
        self._init_session()

    def _placeholders(self):
        self.node_input_ph = tf.placeholder(tf.int32, shape=[None, self.max_len])
        self.node_input_splits = tf.split(tf.one_hot(self.node_input_ph, self.node_dim), len(self.gpu_device_list))

        self.inference_node_ph = tf.placeholder(tf.int32, shape=[None, self.max_len])
        self.inference_node = tf.one_hot(self.inference_node_ph, self.node_dim)

        self.labels = tf.placeholder(tf.int32, shape=[None])  # binary
        self.labels_split = tf.split(self.labels, len(self.gpu_device_list))

        self.is_training_ph = tf.placeholder(tf.bool, ())
        self.global_step = tf.placeholder(tf.int32, ())
        self.hf_iters_per_epoch = tf.placeholder(tf.int32, ())
        if self.use_clr:
            self.lr_multiplier = lib.clr.cyclic_learning_rate(self.global_step, 0.5, 5.,
                                                              self.hf_iters_per_epoch, mode='exp_range')
        else:
            self.lr_multiplier = 1.

    def _build_rnatracker(self, split_idx, mode):
        if mode == 'training':
            node_tensor = self.node_input_splits[split_idx]
        elif mode == 'inference':
            node_tensor = self.inference_node
        else:
            raise ValueError('unknown mode')

        with tf.variable_scope('seq_scan'):
            output = tf.layers.conv1d(node_tensor, self.units, 10, padding='same', use_bias=False, name='conv1')
            output = tf.nn.relu(output)
            output = normalize('bn1', output, self.is_training_ph, self.use_bn)
            output = tf.layers.dropout(output, self.dropout_rate, training=self.is_training_ph)

            output = tf.layers.conv1d(output, self.units, 10, padding='same', use_bias=False, name='conv2')
            output = tf.nn.relu(output)
            output = normalize('bn2', output, self.is_training_ph, self.use_bn)
            output = tf.layers.dropout(output, self.dropout_rate, training=self.is_training_ph)

        with tf.variable_scope('set2set_pooling'):
            output = lib.ops.LSTM.set2set_pooling('set2set_pooling', output, self.pool_steps, 0.,
                                                  self.is_training_ph,
                                                  self.lstm_encoder)
            output = lib.ops.Linear.linear('OutputMapping', output.get_shape().as_list()[-1], 2,
                                           output)  # categorical logits

        if mode == 'training':
            if not hasattr(self, 'output'):
                self.output = [output]
            else:
                self.output += [output]
        else:
            self.inference_output = output

    def _loss(self, split_idx):
        prediction = tf.nn.softmax(self.output[split_idx])

        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.output[split_idx],
                labels=self.labels_split[split_idx],
            ))

        if not hasattr(self, 'cost'):
            self.cost, self.prediction = [cost], [prediction]
        else:
            self.cost += [cost]
            self.prediction += [prediction]

    def _train(self, split_idx):
        gv = self.optimizer.compute_gradients(self.cost[split_idx],
                                              var_list=[var for var in tf.trainable_variables()])

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

        self.acc_val, self.acc_update_op = tf.metrics.accuracy(
            labels=self.labels,
            predictions=tf.argmax(self.prediction, axis=-1),
        )

        self.auc_val, self.auc_update_op = tf.metrics.auc(
            labels=self.labels,
            predictions=self.prediction[:, 1],
        )

        self.inference_prediction = tf.nn.softmax(self.inference_output)
        self.inference_acc_val, self.inference_acc_update_op = tf.metrics.accuracy(
            labels=self.labels,
            predictions=tf.argmax(self.inference_prediction, axis=-1),
        )

        self.inference_auc_val, self.inference_auc_update_op = tf.metrics.auc(
            labels=self.labels,
            predictions=self.inference_prediction[:, 1],
        )

    def _init_session(self):
        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init)
        self.sess.run(self.local_init)

    def reset_session(self):
        del self.saver
        with self.g.as_default():
            self.saver = tf.train.Saver(max_to_keep=100)
        self.sess.run(self.init)
        self.sess.run(self.local_init)
        lib.plot.reset()

    def fit(self, X, y, epochs, batch_size, output_dir, dev_data=None, dev_targets=None, logging=False):

        checkpoints_dir = os.path.join(output_dir, 'checkpoints/')
        os.makedirs(checkpoints_dir)

        node_tensor = X
        if dev_data is None or dev_targets is None:
            # split validation set
            pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
            dev_idx = np.array(list(np.random.choice(pos_idx, int(len(pos_idx) * 0.1), False)) + \
                               list(np.random.choice(neg_idx, int(len(neg_idx) * 0.1), False)))
            train_idx = np.delete(np.arange(len(y)), dev_idx)
            dev_node_tensor = node_tensor[dev_idx]
            dev_targets = y[dev_idx]

            node_tensor = node_tensor[train_idx]
            y = y[train_idx]
        else:
            dev_node_tensor = dev_data

        # batch size should be a multiple of len(self.gpu_device_list)
        dev_rmd = dev_node_tensor.shape[0] % len(self.gpu_device_list)
        if dev_rmd != 0:
            dev_node_tensor = dev_node_tensor[:-dev_rmd]
            dev_targets = dev_targets[:-dev_rmd]

        train_rmd = node_tensor.shape[0] % len(self.gpu_device_list)
        if train_rmd != 0:
            node_tensor = node_tensor[:-train_rmd]
            y = y[:-train_rmd]

        size_train = node_tensor.shape[0]
        iters_per_epoch = size_train // batch_size + (0 if size_train % batch_size == 0 else 1)
        best_dev_cost = np.inf
        lib.plot.set_output_dir(output_dir)
        if logging:
            logger = lib.logger.CSVLogger('run.csv', output_dir,
                                          ['epoch', 'cost', 'acc', 'auc', 'dev_cost', 'dev_acc', 'dev_auc'])
        for epoch in range(epochs):
            permute = np.random.permutation(size_train)
            node_tensor = node_tensor[permute]
            y = y[permute]

            for i in range(iters_per_epoch):
                _node_tensor, _labels \
                    = node_tensor[i * batch_size: (i + 1) * batch_size], \
                      y[i * batch_size: (i + 1) * batch_size]

                self.sess.run(self.train_op,
                              feed_dict={self.node_input_ph: _node_tensor,
                                         self.labels: _labels,
                                         self.global_step: i,
                                         self.hf_iters_per_epoch: iters_per_epoch // 2,
                                         self.is_training_ph: True}
                              )

            train_cost, train_acc, train_auc = \
                self.evaluate(node_tensor, y, batch_size)
            lib.plot.plot('train_cost', train_cost)
            lib.plot.plot('train_acc', train_acc)
            lib.plot.plot('train_auc', train_auc)

            dev_cost, dev_acc, dev_auc = \
                self.evaluate(dev_node_tensor, dev_targets, batch_size)
            lib.plot.plot('dev_cost', dev_cost)
            lib.plot.plot('dev_acc', dev_acc)
            lib.plot.plot('dev_auc', dev_auc)

            if logging:
                logger.update_with_dict({
                    'epoch': epoch,
                    'cost': train_cost,
                    'acc': train_acc,
                    'auc': train_auc,
                    'dev_cost': dev_cost,
                    'dev_acc': dev_acc,
                    'dev_auc': dev_auc
                })

            lib.plot.flush()
            lib.plot.tick()

            if dev_cost < best_dev_cost:
                best_dev_cost = dev_cost
                save_path = self.saver.save(self.sess, checkpoints_dir, global_step=epoch)
                print('Validation sample acc improved. Saved to path %s\n' % (save_path), flush=True)
            else:
                print('\n', flush=True)

        print('Loading best weights %s' % (save_path), flush=True)
        self.saver.restore(self.sess, save_path)
        if logging:
            logger.close()

    def evaluate(self, X, y, batch_size):
        node_tensor = X
        all_cost = 0.
        iters_per_epoch = len(node_tensor) // batch_size + (0 if len(node_tensor) % batch_size == 0 else 1)
        for i in range(iters_per_epoch):
            _node_tensor, _labels \
                = node_tensor[i * batch_size: (i + 1) * batch_size], \
                  y[i * batch_size: (i + 1) * batch_size]
            cost, _, _ = self.sess.run([self.cost, self.acc_update_op, self.auc_update_op],
                                       feed_dict={self.node_input_ph: _node_tensor,
                                                  self.labels: _labels,
                                                  self.is_training_ph: False}
                                       )
            all_cost += cost * len(_node_tensor)
        acc, auc = self.sess.run([self.acc_val, self.auc_val])
        self.sess.run(self.local_init)
        return all_cost / len(node_tensor), acc, auc

    def predict(self, X, batch_size, y=None):
        node_tensor = X
        all_predicton = []
        iters = len(node_tensor) // batch_size + (0 if len(node_tensor) % batch_size == 0 else 1)
        for i in range(iters):
            _node_tensor = node_tensor[i * batch_size:(i + 1) * batch_size]
            feed_dict = {
                self.inference_node_ph: _node_tensor,
                self.is_training_ph: False
            }
            feed_tensor = [self.inference_output]
            if y is not None:
                _labels = y[i * batch_size:(i + 1) * batch_size]
                feed_dict[self.labels] = _labels
                feed_tensor += [self.inference_acc_update_op, self.inference_auc_update_op]

            all_predicton.append(self.sess.run(feed_tensor, feed_dict)[0])
        all_predicton = np.array(all_predicton)

        if y is not None:
            acc, auc = self.sess.run([self.inference_acc_val, self.inference_auc_val])
            self.sess.run(self.local_init)
            return all_predicton, acc, auc
        else:
            return all_predicton

    def delete(self):
        tf.reset_default_graph()
        self.sess.close()

    def load(self, chkp_path):
        self.saver.restore(self.sess, chkp_path)
