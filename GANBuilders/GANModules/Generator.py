import tensorflow as tf
from lib.resutils import resblock
import lib.ops.LSTM, lib.ops.Linear, lib.ops.Conv1D
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


class Generator:

    def __init__(self, nb_emb, gpu_device, **kwargs):
        self.nb_emb = nb_emb
        self.gpu_device = gpu_device

        # hyperparams
        self.z_dim = kwargs.get('z_dim', 32)
        self.nb_layers = kwargs.get('nb_layers', 4)
        self.filter_size = kwargs.get('filter_size', 3)
        self.residual_connection = kwargs.get('residual_connection', 1.0)
        self.output_dim = kwargs.get('output_dim', 32)
        self.att_dim = kwargs.get('att_dim', 50)
        self.learning_rate = kwargs.get('learning_rate', 2e-4)

        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate * self.lr_multiplier, beta1=0., beta2=0.9)
            with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
                self._build_residual_network()
            self._init_session()

    def build_loss(self, discriminator):
        self.discriminator = discriminator
        with self.g.as_default():
            self._loss()
            self._train()
            self.train_op = self.optimizer.apply_gradients(self.gv)
            _stats('Generator', self.gv)
            self.saver = tf.train.Saver(max_to_keep=5, var_list=self.var_list)
            self.init = tf.global_variables_initializer()
        self.g.finalize()
        self.sess.run(self.init)

    def _placeholders(self):
        self.batch_size = tf.placeholder(tf.int32, ())
        self.noise = tf.random_normal([self.batch_size, self.z_dim])
        self.is_training_ph = tf.placeholder(tf.bool, ())
        self.lr_multiplier = tf.placeholder_with_default(1., ())

    def _build_residual_network(self):
        base_len = 336 // 2 ** self.nb_layers  # for 4 layers the base length would be 21
        output = lib.ops.Linear.linear('LinearTransform', self.z_dim, base_len * self.output_dim, self.noise)
        output = tf.reshape(output, [-1, base_len, self.output_dim])
        for i, stride in enumerate([2] * self.nb_layers):
            output = resblock('ResBlock%d' % (i), self.output_dim, self.output_dim, self.filter_size, output,
                              'up', self.is_training_ph, use_bn=False,
                              r=self.residual_connection, stride=stride)
        # output = normalize('NormFinal', output, self.is_training_ph)
        output = tf.nn.relu(output)
        output = lib.ops.Conv1D.conv1d('OutputConv', self.output_dim, self.nb_emb, 1, output, he_init=False)
        output = tf.nn.softmax(output)  # [batch_size, length, n_emb]

        self.output = output

    def _loss(self):
        gen_cost = -tf.reduce_mean(self.discriminator.output_fake)
        self.gen_cost = gen_cost

    def _train(self):
        self.var_list = [var for var in tf.trainable_variables() if 'Generator' in var.name]
        gv = self.optimizer.compute_gradients(self.gen_cost, var_list=self.var_list)
        self.gv = gv

    def _init_session(self):
        gpu_options = tf.GPUOptions()
        gpu_options.per_process_gpu_memory_fraction = 0.7
        if type(self.gpu_device) is list:
            gpu_options.visible_device_list = ','.join([device[-1] for device in self.gpu_device])
        else:
            gpu_options.visible_device_list = self.gpu_device[-1]
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(gpu_options=gpu_options))

    def reset_session(self):
        del self.saver
        with self.g.as_default():
            self.saver = tf.train.Saver(max_to_keep=5, var_list=self.var_list)
        self.sess.run(self.init)

    def train_on_batch(self, n_samples, lr_multiplier=1.):
        self.sess.run(self.train_op,
                      feed_dict={
                          self.batch_size: n_samples,
                          self.lr_multiplier: lr_multiplier,
                          self.is_training_ph: True}
                      )

    def sample_generator(self, n_samples, noise=None):
        if noise is None:
            noise = self.sess.run(self.noise, {self.batch_size: n_samples})
        raw_seqs = self.sess.run(self.output, {
            self.batch_size: n_samples,
            self.noise: noise,
            self.is_training_ph: False})

        return raw_seqs

    def delete(self):
        self.sess.close()

    def save(self, save_path, epoch):
        self.saver.save(self.sess, save_path, global_step=epoch)

    def restore(self, chkp_path):
        self.saver.restore(self.sess, chkp_path)
