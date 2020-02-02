import os
import sys
import time
import math
import numpy as np
import tensorflow as tf
import subprocess as sp
from Bio.Align.Applications import ClustalwCommandline

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Model import _stats
from lib.rgcn_utils import normalize
import lib.plot, lib.logger, lib.clr
import lib.ops.LSTM, lib.ops.Linear, lib.ops.Conv1D
from lib.tf_ghm_loss import get_ghm_weights
from lib.AMSGrad import AMSGrad


class JMRT:

    def __init__(self, node_dim, embedding_vec, gpu_device, **kwargs):
        self.node_dim = node_dim
        self.embedding_vec = embedding_vec
        self.vocab_size = embedding_vec.shape[0]
        self.gpu_device = gpu_device

        # hyperparams
        self.units = kwargs.get('units', 32)
        self.pool_steps = kwargs.get('pool_steps', 10)
        self.lstm_encoder = kwargs.get('lstm_encoder', True)
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.learning_rate = kwargs.get('learning_rate', 2e-4)
        self.use_clr = kwargs.get('use_clr', False)
        self.use_momentum = kwargs.get('use_momentum', False)
        self.use_bn = kwargs.get('use_bn', False)

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
                    self.learning_rate * self.lr_multiplier,
                    beta2=0.999
                )

            with tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE):
                self._build_ggnn()
                self._loss()
                self._train()
                self._merge()
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    self.train_op = self.optimizer.apply_gradients(self.gv)
                _stats('Joint_MRT', self.gv)
                self.saver = tf.train.Saver(max_to_keep=5)
                self.init = tf.global_variables_initializer()
                self.local_init = tf.local_variables_initializer()
                self.g.finalize()
        self._init_session()

    def _placeholders(self):
        self.node_input_ph = tf.placeholder(tf.int32, shape=[None, ])  # nb_nodes
        # nb_nodes x nb_nodes

        self.labels = tf.placeholder(tf.int32, shape=[None, None, ])
        self.max_len = tf.placeholder(tf.int32, shape=())
        self.segment_length = tf.placeholder(tf.int32, shape=[None, ])

        self.is_training_ph = tf.placeholder(tf.bool, ())
        self.global_step = tf.placeholder(tf.int32, ())
        self.hf_iters_per_epoch = tf.placeholder(tf.int32, ())
        if self.use_clr:
            self.lr_multiplier = lib.clr. \
                cyclic_learning_rate(self.global_step, 0.5, 5.,
                                     self.hf_iters_per_epoch, mode='exp_range')
        else:
            self.lr_multiplier = 1.

    def _build_ggnn(self):
        embedding = tf.get_variable('embedding_layer', shape=(self.vocab_size, self.node_dim),
                                    initializer=tf.constant_initializer(self.embedding_vec), trainable=False)
        output = tf.nn.embedding_lookup(embedding, self.node_input_ph)
        self.node_tensor = output
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

        with tf.variable_scope('seq_scan'):
            # paddings will influence the prediction results, even unavoidable if batch norm is used
            # but the influence will be very small, enough to ignore it
            output = lib.ops.Conv1D.conv1d('conv1', self.node_dim, self.units, 10, output, biases=False,
                                           pad_mode='SAME', variables_on_cpu=False)
            output = normalize('bn1', output, self.use_bn, self.is_training_ph)
            output = tf.nn.relu(output)
            output = tf.layers.dropout(output, self.dropout_rate, training=self.is_training_ph)

            output = lib.ops.Conv1D.conv1d('conv2', self.units, self.units, 10, output, biases=False,
                                           pad_mode='SAME', variables_on_cpu=False)
            output = normalize('bn2', output, self.use_bn, self.is_training_ph)
            output = tf.nn.relu(output)
            output = tf.layers.dropout(output, self.dropout_rate, training=self.is_training_ph)

        with tf.variable_scope('set2set_pooling'):
            output = lib.ops.LSTM.set2set_pooling('set2set_pooling', output, self.pool_steps, self.dropout_rate,
                                                  self.is_training_ph, self.lstm_encoder, mask_offset,
                                                  variables_on_cpu=False)

        self.nuc_embedding = tf.get_collection('nuc_emb')[0]  # will depend on if bilstm encoder is used or not
        self.nuc_output = lib.ops.Linear.linear('bilstm_nuc_output',
                                                self.units * 2 if self.lstm_encoder else self.units, 2,
                                                self.nuc_embedding)
        self.output = lib.ops.Linear.linear('OutputMapping', output.get_shape().as_list()[-1],
                                            2, output, variables_on_cpu=False)  # categorical logits

    def _loss(self):
        self.prediction = tf.nn.softmax(self.output)
        self.nuc_prediction = tf.nn.softmax(self.nuc_output)

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
            self.nuc_cost = tf.reduce_sum(
                get_ghm_weights(self.nuc_prediction, self.labels, self.mask,
                                bins=10, alpha=0.75, name='GHM_NUC_EMB') * \
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.nuc_output,
                    labels=tf.one_hot(self.labels, depth=2),
                ) / tf.cast(tf.reduce_sum(self.segment_length), tf.float32)
            )
        else:
            self.nuc_cost = tf.reduce_sum(
                self.mask * \
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.nuc_output,
                    labels=tf.one_hot(self.labels, depth=2),
                )) / tf.cast(tf.reduce_sum(self.segment_length), tf.float32)

        self.cost = self.mixing_ratio * self.graph_cost + (1. - self.mixing_ratio) * self.nuc_cost

    def _train(self):
        self.gv = self.optimizer.compute_gradients(self.cost,
                                                   var_list=[var for var in tf.trainable_variables()],
                                                   colocate_gradients_with_ops=True)

    def _merge(self):
        # If the example contains a binding site: more global
        self.seq_acc_val, self.seq_acc_update_op = tf.metrics.accuracy(
            labels=tf.reduce_max(self.labels, axis=-1),
            predictions=tf.to_int32(tf.argmax(self.prediction, axis=-1)),
        )

        # nucleotide level accuracy of containing a binding site
        self.nuc_acc_val, self.nuc_acc_update_op = tf.metrics.accuracy(
            labels=tf.reduce_max(self.labels, axis=-1),
            predictions=tf.to_int32(tf.reduce_max(
                tf.argmax(self.nuc_prediction, axis=-1), axis=-1)),
        )

        self.acc_val = [self.seq_acc_val, self.nuc_acc_val]
        self.acc_update_op = [self.seq_acc_update_op, self.nuc_acc_update_op]

        # graph level ROC AUC
        self.auc_val, self.auc_update_op = tf.metrics.auc(
            labels=tf.reduce_max(self.labels, axis=-1),
            predictions=self.prediction[:, 1],
        )

        self.g_nodes = tf.gradients(self.prediction[:, 1], self.node_tensor)[0]

    def _init_session(self):
        gpu_options = tf.GPUOptions()
        gpu_options.per_process_gpu_memory_fraction = 0.2
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
    def indexing_iterable(cls, iterable, idx):
        return [item[idx] for item in iterable]

    @classmethod
    def random_crop(cls, node_tensor, raw_seq, y, pos_read_retention_rate=0.5):
        m_seq, m_label, m_sg, m_data, m_row_col = [], [], [], [], []
        for seq, _raw_seq, label in zip(node_tensor, raw_seq, y):
            if np.max(label) == 0:
                # negative sequence
                pseudo_label = (np.array(list(_raw_seq)) <= 'Z').astype(np.int32)
                pos_idx = np.where(pseudo_label == 1)[0]
            else:
                pos_idx = np.where(label == 1)[0]
                # keep more than 3/4 of the sequence (length), and random start
                read_length = len(pos_idx)
                rate = min(max(pos_read_retention_rate, np.random.rand()), 0.9)
                winsize = int(rate * read_length)
                surplus = read_length - winsize + 1
                start_idx = np.random.choice(range(int(surplus / 4), math.ceil(surplus * 3 / 4)))
                label = [0] * (pos_idx[0] + start_idx) + [1] * winsize + [0] * \
                        (len(seq) - winsize - start_idx - pos_idx[0])

            left_truncate = int(np.random.rand() * pos_idx[0])
            right_truncate = int(np.random.rand() * (len(seq) - pos_idx[-1] - 1))

            if not right_truncate > 0:
                right_truncate = -len(seq)

            seq = seq[left_truncate: -right_truncate]
            label = label[left_truncate: -right_truncate]
            m_seq.append(seq)
            m_sg.append(len(seq))
            m_label.append(label)

        return np.array(m_seq), np.array(m_sg), np.array(m_label)

    def fit(self, X, y, epochs, batch_size, output_dir, logging=False, epoch_to_start=0, random_crop=False):
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
        lib.plot.set_output_dir(output_dir)
        if logging:
            logger = lib.logger.CSVLogger('run.csv', output_dir,
                                          ['epoch', 'cost', 'graph_cost', 'nuc_cost',
                                           'seq_acc', 'nuc_acc', 'auc',
                                           'dev_cost', 'dev_graph_cost', 'dev_nuc_cost',
                                           'dev_seq_acc', 'dev_nuc_acc', 'dev_auc'])

        for epoch in range(epoch_to_start, epochs):

            permute = np.random.permutation(size_train)
            node_tensor, segment_length, raw_seq = self.indexing_iterable(X, permute)
            y = train_targets[permute]

            if random_crop:
                # augmentation
                node_tensor, segment_length, y = \
                    self.random_crop(node_tensor, raw_seq, y)

            prepro_time = 0.
            training_time = 0.
            for i in range(iters_per_epoch):
                prepro_start = time.time()
                _node_tensor, _segment, _labels \
                    = node_tensor[i * batch_size: (i + 1) * batch_size], \
                      segment_length[i * batch_size: (i + 1) * batch_size], \
                      y[i * batch_size: (i + 1) * batch_size]

                _max_len = max(_segment)
                _labels = np.array([np.pad(label, [_max_len - len(label), 0], mode='constant') for label in _labels])

                feed_dict = {
                    self.node_input_ph: np.concatenate(_node_tensor, axis=0),
                    self.labels: _labels,
                    self.max_len: _max_len,
                    self.segment_length: _segment,
                    self.global_step: i,
                    self.hf_iters_per_epoch: iters_per_epoch // 2,
                    self.is_training_ph: True
                }
                prepro_end = time.time()
                prepro_time += (prepro_end - prepro_start)
                self.sess.run(self.train_op, feed_dict)
                training_time += (time.time() - prepro_end)
            print('preprocessing time: %.4f, training time: %.4f' % (prepro_time / (i + 1), training_time / (i + 1)))
            train_cost, train_acc, train_auc = self.evaluate(X, train_targets, batch_size)
            lib.plot.plot('train_cost', train_cost[0])
            lib.plot.plot('train_graph_cost', train_cost[1])
            lib.plot.plot('train_nuc_cost', train_cost[2])
            lib.plot.plot('train_seq_acc', train_acc[0])
            lib.plot.plot('train_nuc_acc', train_acc[1])
            lib.plot.plot('train_auc', train_auc)

            dev_cost, dev_acc, dev_auc = self.evaluate(dev_data, dev_targets, batch_size)
            lib.plot.plot('dev_cost', dev_cost[0])
            lib.plot.plot('dev_graph_cost', dev_cost[1])
            lib.plot.plot('dev_nuc_cost', dev_cost[2])
            lib.plot.plot('dev_seq_acc', dev_acc[0])
            lib.plot.plot('dev_nuc_acc', dev_acc[1])
            lib.plot.plot('dev_auc', dev_auc)

            logger.update_with_dict({
                'epoch': epoch, 'cost': train_cost[0], 'graph_cost': train_cost[1],
                'nuc_cost': train_cost[2], 'seq_acc': train_acc[0], 'nuc_acc': train_acc[1],
                'auc': train_auc,
                'dev_cost': dev_cost[0], 'dev_graph_cost': dev_cost[1],
                'dev_nuc_cost': dev_cost[2], 'dev_seq_acc': dev_acc[0],
                'dev_nuc_acc': dev_acc[1], 'dev_auc': dev_auc,
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

    def evaluate(self, X, y, batch_size, random_crop=False):
        node_tensor, segment_length, raw_seq = X
        if random_crop:
            # augmentation
            node_tensor, segment_length, y = \
                self.random_crop(node_tensor, raw_seq, y)
        all_cost = 0.
        all_graph_cost = 0.
        all_bilstm_nuc_cost = 0.
        iters_per_epoch = len(node_tensor) // batch_size + (0 if len(node_tensor) % batch_size == 0 else 1)
        for i in range(iters_per_epoch):
            _node_tensor, _segment, _labels \
                = node_tensor[i * batch_size: (i + 1) * batch_size], \
                  segment_length[i * batch_size: (i + 1) * batch_size], \
                  y[i * batch_size: (i + 1) * batch_size]

            _max_len = max(_segment)
            _labels = np.array([np.pad(label, [_max_len - len(label), 0], mode='constant') for label in _labels])

            feed_dict = {
                self.node_input_ph: np.concatenate(_node_tensor, axis=0),
                self.labels: _labels,
                self.max_len: _max_len,
                self.segment_length: _segment,
                self.is_training_ph: False
            }

            cost, graph_cost, bilstm_nuc_cost, _, _ = self.sess.run(
                [self.cost, self.graph_cost, self.nuc_cost,
                 self.acc_update_op, self.auc_update_op], feed_dict)
            all_cost += cost * len(_node_tensor)
            all_graph_cost += graph_cost * len(_node_tensor)
            all_bilstm_nuc_cost += bilstm_nuc_cost * len(_node_tensor)
        acc, auc = self.sess.run([self.acc_val, self.auc_val])
        self.sess.run(self.local_init)
        return (all_cost / len(node_tensor), all_graph_cost / len(node_tensor),
                all_bilstm_nuc_cost / len(node_tensor)), acc, auc

    def predict(self, X, batch_size):
        node_tensor, segment_length, raw_seq = X
        preds = []
        iters_per_epoch = len(node_tensor) // batch_size + (0 if len(node_tensor) % batch_size == 0 else 1)
        for i in range(iters_per_epoch):
            _node_tensor, _segment \
                = node_tensor[i * batch_size: (i + 1) * batch_size], \
                  segment_length[i * batch_size: (i + 1) * batch_size]

            _max_len = max(_segment)

            feed_dict = {
                self.node_input_ph: np.concatenate(_node_tensor, axis=0),
                self.max_len: _max_len,
                self.segment_length: _segment,
                self.is_training_ph: False
            }
            preds.append(self.sess.run(self.prediction, feed_dict))

        return np.concatenate(np.array(preds), axis=0)

    def integrated_gradients(self, X, y, ids, interp_steps=100, save_path=None, max_plots=np.inf):
        counter = 0
        for _node_tensor, _segment, _, _label, _id in zip(*X, y, ids):
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

            feed_dict = {
                self.node_tensor: np.concatenate(np.array(new_node_tensor), axis=0),
                self.max_len: _segment,
                self.segment_length: [_segment] * (interp_steps + 1),
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

    def extract_sequence_motifs(self, X, interp_steps=100, save_path=None, max_examples=4000, mer_size=12):
        counter = 0
        all_mers = []
        all_scores = []

        for _node_tensor, _segment, _raw_seq in zip(*X):
            if counter >= max_examples:
                break
            _meshed_node_tensor = np.array([self.embedding_vec[idx] for idx in _node_tensor])
            _meshed_reference_input = np.zeros_like(_meshed_node_tensor)
            new_node_tensor = []
            for i in range(0, interp_steps + 1):
                new_node_tensor.append(
                    _meshed_reference_input + i / interp_steps * (_meshed_node_tensor - _meshed_reference_input))

            feed_dict = {
                self.node_tensor: np.concatenate(np.array(new_node_tensor), axis=0),
                self.max_len: _segment,
                self.segment_length: [_segment] * (interp_steps + 1),
                self.is_training_ph: False
            }

            grads = self.sess.run(self.g_nodes, feed_dict).reshape((interp_steps + 1, _segment, 4))
            grads = (grads[:-1] + grads[1:]) / 2.0
            node_scores = np.sum(np.average(grads, axis=0) * (_meshed_node_tensor - _meshed_reference_input), axis=-1)
            mer_scores = []
            for start in range(len(node_scores) - mer_size + 1):
                mer_scores.append(np.sum(node_scores[start: start + mer_size]))
            max_scores = np.max(node_scores)
            all_mers.append(_raw_seq[np.argmax(mer_scores): np.argmax(mer_scores) + mer_size].upper().replace('T', 'U'))
            all_scores.append(max_scores)
            counter += 1

        FNULL = open(os.devnull, 'w')
        for top_rank in [100, 500, 1000, 2000]:
            # align top_rank mers
            best_mers = np.array(all_mers)[:top_rank]
            fasta_path = os.path.join(save_path, 'top%d_mers.fa' % (top_rank))
            with open(fasta_path, 'w') as f:
                for i, seq in enumerate(best_mers):
                    print('>{}'.format(i), file=f)
                    print(seq, file=f)
            # multiple sequence alignment
            out_fasta_path = os.path.join(save_path, 'aligned_top%d_mers.fa' % (top_rank))
            cline = ClustalwCommandline("clustalw2", infile=fasta_path, type="DNA", outfile=out_fasta_path,
                                        output="FASTA")
            sp.call(str(cline), shell=True, stdout=FNULL)
            motif_path = os.path.join(save_path, 'top%d-sequence_motif.jpg' % (top_rank))
            lib.plot.plot_weblogo(out_fasta_path, motif_path)

        even_mers = all_mers[::2]
        fasta_path = os.path.join(save_path, 'even_mers.fa')
        with open(fasta_path, 'w') as f:
            for i, seq in enumerate(even_mers):
                print('>{}'.format(i), file=f)
                print(seq, file=f)
        # multiple sequence alignment
        out_fasta_path = os.path.join(save_path, 'aligned_even_mers.fa')
        cline = ClustalwCommandline("clustalw2", infile=fasta_path, type="DNA", outfile=out_fasta_path, output="FASTA")
        sp.call(str(cline), shell=True, stdout=FNULL)
        motif_path = os.path.join(save_path, 'top1000-even-sequence_motif.jpg')
        lib.plot.plot_weblogo(out_fasta_path, motif_path)

        odd_mers = all_mers[1::2]
        fasta_path = os.path.join(save_path, 'odd_mers.fa')
        with open(fasta_path, 'w') as f:
            for i, seq in enumerate(odd_mers):
                print('>{}'.format(i), file=f)
                print(seq, file=f)
        # multiple sequence alignment
        out_fasta_path = os.path.join(save_path, 'aligned_odd_mers.fa')
        cline = ClustalwCommandline("clustalw2", infile=fasta_path, type="DNA", outfile=out_fasta_path, output="FASTA")
        sp.call(str(cline), shell=True, stdout=FNULL)
        motif_path = os.path.join(save_path, 'top1000-odd-sequence_motif.jpg')
        lib.plot.plot_weblogo(out_fasta_path, motif_path)

    def delete(self):
        tf.reset_default_graph()
        self.sess.close()

    def load(self, chkp_path):
        self.saver.restore(self.sess, chkp_path)
