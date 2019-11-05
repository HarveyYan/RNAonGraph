import tensorflow as tf
from lib.resutils import OptimizedResBlockDisc1, resblock
import lib.ops.LSTM, lib.ops.Linear, lib.ops.Conv1D, lib.logger
import locale


def _stats(name, grads_and_vars):
    # show all trainable weights
    print("{} Params:".format(name))
    total_param_count = 0
    for g, v in grads_and_vars:
        shape = v.get_shape()
        shape_str = ",".join([str(x) for x in v.get_shape()])

        param_count = 1
        for dim in shape:
            param_count *= int(dim)
        total_param_count += param_count

        if g == None:
            print("\t{} ({}) [no grad!]".format(v.name, shape_str))
        else:
            print("\t{} ({})".format(v.name, shape_str))
    print("Total param count: {}".format(
        locale.format("%d", total_param_count, grouping=True)
    ))


class Discriminator:

    def __init__(self, nb_emb, embedding_vec, generator, gpu_device, **kwargs):
        self.nb_emb = nb_emb
        self.embedding_vec = embedding_vec
        self.vocab_size = embedding_vec.shape[0]
        self.generator = generator
        self.gpu_device = gpu_device

        # hyperparams
        self.nb_layers = kwargs.get('nb_layers', 4)
        self.resample = kwargs.get('resample', 'down')
        self.filter_size = kwargs.get('filter_size', 3)
        self.residual_connection = kwargs.get('residual_connection', 1.0)
        self.output_dim = kwargs.get('output_dim', 32)
        self.learning_rate = kwargs.get('learning_rate', 2e-4)

        self.g = self.generator.g
        with self.g.as_default():
            self._placeholders()
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate * self.lr_multiplier, beta1=0., beta2=0.9)

            with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
                for type in ['real', 'fake', 'gp']:
                    self._build_residual_classifier(type)
                self._loss()
                self._train()
            self.train_op = self.optimizer.apply_gradients(self.gv)
            _stats('Discriminator', self.gv)
            self.saver = tf.train.Saver(max_to_keep=5, var_list=self.var_list)
        self._init_session()

    def _placeholders(self):
        self.input_real_ph = tf.placeholder(tf.int32, shape=[None, None])
        self.lr_multiplier = tf.placeholder_with_default(1., ())

    def _build_residual_classifier(self, type):
        if type == 'real':
            embedding = tf.get_variable('embedding_layer', shape=(self.vocab_size, self.nb_emb),
                                        initializer=tf.constant_initializer(self.embedding_vec), trainable=False)
            output = tf.nn.embedding_lookup(embedding, self.input_real_ph)
            self.input_real = output
        elif type == 'fake':
            output = self.generator.output
        elif type == 'gp':
            alpha = tf.random_uniform(
                shape=[tf.shape(self.input_real_ph)[0], 1, 1],
                minval=0.,
                maxval=1.
            )
            differences = self.generator.output - self.input_real
            interpolates = self.input_real + alpha * differences
            output = interpolates
        else:
            raise ValueError('Type %s not understood.' % (type))

        for i in range(self.nb_layers):
            if i == 0:
                output = OptimizedResBlockDisc1(output, self.nb_emb, self.output_dim, resample=self.resample,
                                                filter_size=self.filter_size)
            else:
                output = resblock('ResBlock%d' % (i), self.output_dim, self.output_dim, self.filter_size, output,
                                  self.resample, None, use_bn=False, r=self.residual_connection)
        output = tf.nn.relu(output)
        att_output = tf.reduce_mean(output, axis=[1])
        output = tf.reshape(lib.ops.Linear.linear('AMOutput', self.output_dim, 1, att_output), [-1])

        if type == 'real':
            self.output_real = output
        elif type == 'fake':
            self.output_fake = output
        elif type == 'gp':
            self.interpolates = interpolates
            self.inter_output = output

    def _loss(self):
        real_cost = -tf.reduce_mean(self.output_real)
        fake_cost = tf.reduce_mean(self.output_fake)
        disc_cost = real_cost + fake_cost

        '''gradient penalty'''
        gradients = tf.gradients(self.inter_output, [self.interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-10)
        gradient_penalty = 10 * tf.reduce_mean((slopes - 1.) ** 2)

        self.disc_cost, self.real_cost, self.fake_cost, self.gradient_penalty = \
            disc_cost, real_cost, fake_cost, gradient_penalty

    def _train(self):
        self.var_list = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]
        self.gv = self.optimizer.compute_gradients(self.disc_cost +
                                                   self.gradient_penalty, var_list=self.var_list)

    def _init_session(self):
        self.sess = self.generator.sess

    def reset_session(self):
        del self.saver
        with self.g.as_default():
            self.saver = tf.train.Saver(max_to_keep=200, var_list=self.var_list)

    def train_on_batch(self, x_batch, lr_multiplier=1.):
        n_samples = x_batch.shape[0]
        self.sess.run(self.train_op,
                      feed_dict={self.input_real_ph: x_batch,
                                 self.lr_multiplier: lr_multiplier,
                                 self.generator.batch_size: n_samples,
                                 self.generator.is_training_ph: False}
                      )

    def evaluate(self, X, batch_size):
        iters_per_epoch = len(X) // batch_size + (0 if len(X) % batch_size == 0 else 1)
        disc_cost, gp = 0., 0.,
        for i in range(iters_per_epoch):
            _data = X[i * batch_size: (i + 1) * batch_size]
            _disc_cost, _gp \
                = self.sess.run([self.disc_cost, self.gradient_penalty],
                                feed_dict={self.input_real_ph: _data,
                                           self.generator.batch_size: _data.shape[0],
                                           self.generator.is_training_ph: False}
                                )
            disc_cost += _disc_cost * _data.shape[0]
            gp += _gp * _data.shape[0]
        return -disc_cost / len(X), gp / len(X)

    def delete(self):
        self.sess.close()

    def save(self, save_path, epoch):
        self.saver.save(self.sess, save_path, global_step=epoch)

    def restore(self, chkp_path):
        self.saver.restore(self.sess, chkp_path)
