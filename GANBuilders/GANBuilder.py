import os
import sys
import shutil
import inspect
import datetime
import numpy as np
import tensorflow as tf
from collections import defaultdict

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GANBuilders.GANModules import Generator
from GANBuilders.GANModules import Discriminator
import lib.plot, lib.logger, lib.graphprot_dataloader

tf.logging.set_verbosity(tf.logging.FATAL)
tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_integer('epochs', 100, '')
FLAGS = tf.app.flags.FLAGS

Z_DIM = 32
GEN_BS_MULTIPLE = 2
BATCH_SIZE = 128
EPOCHS = FLAGS.epochs  # How many iterations to train for
MAX_LEN = 336
N_EMB = 4
N_CRITIC = 5
DEVICES = ['/gpu:6']


def convert_to_bits(seq):
    tmp = seq * np.log2(seq)
    tmp[np.isnan(tmp)] = 0.
    return (2. + np.sum(tmp, axis=-1, keepdims=True)) * seq


if __name__ == "__main__":
    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if FLAGS.output_dir == '':
        output_dir = os.path.join(basedir, 'output', 'GAN', cur_time)
    else:
        output_dir = os.path.join(basedir, 'output', 'GAN', cur_time + '-' + FLAGS.output_dir)

    os.makedirs(output_dir)
    lib.plot.set_output_dir(output_dir)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints/')
    os.makedirs(checkpoint_dir)
    samples_dir = os.path.join(output_dir, 'samples/')
    os.makedirs(samples_dir)
    model_dir = {}
    for model in ['generator', 'discriminator']:
        subdir = os.path.join(checkpoint_dir, model) + '/'
        os.makedirs(subdir)
        model_dir[model] = subdir

    # backup python scripts, for future reference
    backup_dir = os.path.join(output_dir, 'backup')
    os.makedirs(backup_dir)
    shutil.copy(__file__, backup_dir)
    shutil.copy(inspect.getfile(Generator), backup_dir)
    shutil.copy(inspect.getfile(Discriminator), backup_dir)

    dataset = lib.graphprot_dataloader.load_short_seq(MAX_LEN, 50000)

    # all components in a GAN
    generator = Generator(N_EMB, DEVICES, learning_date=2e-4)
    discriminator = Discriminator(N_EMB, dataset['VOCAB_VEC'], generator, DEVICES)
    generator.build_loss(discriminator)

    logger = lib.logger.CSVLogger('log.csv', output_dir,
                                  ['epoch', 'neg_critic_loss', 'gp', 'dev_neg_critic_loss', 'dev_gp'])

    all_data = dataset['seq']
    dev_idx = np.random.choice(range(all_data.shape[0]), int(all_data.shape[0] * 0.1), False)
    dev_data = all_data[dev_idx]
    train_idx = np.delete(np.arange(all_data.shape[0]), dev_idx)
    train_data = all_data[train_idx]
    size_train = len(train_idx)

    for epoch in range(EPOCHS):
        permute = np.random.permutation(np.arange(size_train))
        disc_data = train_data[permute]
        train_ret_dict = defaultdict(lambda: 0.)
        iters_per_epoch = size_train // BATCH_SIZE + (0 if size_train % BATCH_SIZE == 0 else 1)

        for i in range(iters_per_epoch):
            _data = disc_data[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            ''' wgan-gp training phase '''
            _critic_data = _data
            for _ in range(N_CRITIC):
                discriminator.train_on_batch(_critic_data)
                random_index = int((np.random.rand() * (size_train - _critic_data.shape[0])))
                _critic_data = disc_data[random_index: random_index + _critic_data.shape[0]]

            '''generator training phase'''
            generator.train_on_batch(GEN_BS_MULTIPLE * BATCH_SIZE)

        neg_critic_loss, gp = discriminator.evaluate(disc_data, BATCH_SIZE)
        dev_neg_critic_loss, dev_gp = discriminator.evaluate(dev_data, BATCH_SIZE)

        logger.update_with_dict({
            'epoch': epoch,
            'neg_critic_loss': neg_critic_loss,
            'gp': gp,
            'dev_neg_critic_loss': dev_neg_critic_loss,
            'dev_gp': dev_gp
        })
        lib.plot.plot('neg_critic_loss', neg_critic_loss)
        lib.plot.plot('dev_neg_critic_loss', dev_neg_critic_loss)
        lib.plot.plot('gp', gp)
        lib.plot.plot('dev_gp', dev_gp)
        lib.plot.flush()
        lib.plot.tick()

        # sampling
        noise = np.random.randn(5, Z_DIM)
        samples = generator.sample_generator(5, noise)
        sample_bits = convert_to_bits(samples)
        for i in range(5):
            save_path = os.path.join(samples_dir, 'sample_%d.jpg' % (i))
            lib.plot.plot_weights(sample_bits[i, 100:250, :], save_path=save_path)

        generator.save(model_dir['generator'], epoch)
        discriminator.save(model_dir['discriminator'], epoch)
