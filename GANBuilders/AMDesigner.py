import os
import sys
import numpy as np
import tensorflow as tf
import datetime

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import lib.logger, lib.plot

tf.logging.set_verbosity(tf.logging.FATAL)
tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_integer('iterations', 10000, 'Training iterations')
FLAGS = tf.app.flags.FLAGS

gen_chpk_path = '../output/GAN/20191102-162330-nearest-neighbour-e200/checkpoints/generator/-199.meta'
pred_chpk_path = tf.train.latest_checkpoint(
    '../output/Joint-MRT-Graphprot-debiased/20191031-193602-debiased-e300-clr-AMSGrad-PARCLIP_PUM2/fold0/checkpoints/') + '.meta'


def convert_to_bits(seq):
    tmp = seq * np.log2(seq)
    tmp[np.isnan(tmp)] = 0.
    return (2. + np.sum(tmp, axis=-1, keepdims=True)) * seq


if __name__ == "__main__":
    if FLAGS.output_dir == '':
        output_dir = os.path.join(basedir, 'output', 'AM',
                                  datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        output_dir = os.path.join(basedir, 'output', 'AM',
                                  datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + FLAGS.output_dir)

    os.makedirs(output_dir)
    lib.plot.set_output_dir(output_dir)
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir)

    lib.plot.set_output_dir(output_dir)

    gpu_options = tf.GPUOptions()
    gpu_options.visible_device_list = '7'
    gpu_options.allow_growth = True
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)

    '''loading generator'''
    max_len, batch_size, latent_dim = 336, 128, 32
    latent_codes = tf.Variable(tf.random_normal(shape=(batch_size, latent_dim)), name='latent_codes')

    '''replace random_normal noise tensor and labels with Variables '''
    saver = tf.train.import_meta_graph(gen_chpk_path, input_map={
        'random_normal:0': latent_codes,
    }, import_scope='import_gen')
    saver.restore(sess, gen_chpk_path[:-5])
    graph = sess.graph

    '''locate generator output and global training placeholder'''
    gen_output = graph.get_tensor_by_name('import_gen/Generator/Softmax:0')
    # generator does not make use of the is_training_ph switch
    # global_training_ph = graph.get_tensor_by_name('import_gen/Placeholder:0')  # True or False

    '''locate predictor output'''
    pred_saver = tf.train.import_meta_graph(pred_chpk_path, input_map={
        'Placeholder_4:0': tf.constant(False, dtype=tf.bool),
        'Classifier/TensorArrayStack/TensorArrayGatherV3:0': gen_output,  # output
        'Classifier/TensorArrayStack_1/TensorArrayGatherV3:0': tf.zeros((batch_size,), dtype=tf.int32),  # mask_offset
    }, import_scope='import_pred')

    pred_saver.restore(sess, pred_chpk_path[:-5])
    pred_output = graph.get_tensor_by_name('import_pred/Classifier/OutputMapping/BiasAdd:0')
    prediction = tf.nn.softmax(pred_output)
    labels = tf.ones((batch_size,), dtype=tf.int32)

    cp_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=pred_output))
    acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.to_int32(tf.argmax(pred_output, -1)),
                tf.to_int32(tf.ones((batch_size,), dtype=tf.int32)),
            ),
            tf.float32
        )
    )

    # latent_reg_coeff = 1e-4
    # latent_reg = latent_reg_coeff * tf.reduce_mean(tf.reduce_sum(latent_codes ** 2, axis=1))

    # define train op
    gradients = tf.gradients(ys=cp_loss, xs=latent_codes)[0]  # + tf.gradients(ys=latent_reg, xs=latent_codes)[0]
    optimizer = tf.train.AdamOptimizer(1e-1)  # , beta1=0.0, beta2=0.9)
    train_op = optimizer.apply_gradients([(gradients, latent_codes)])

    # momentum_initializers = [var.initializer for var in tf.global_variables() if 'Momentum' in var.name]
    # sess.run([latent_codes.initializer, *momentum_initializers])

    adam_initializers = [var.initializer for var in tf.global_variables() if
                         'Adam' in var.name or ('beta' in var.name and '/BN/beta' not in var.name)]
    sess.run([latent_codes.initializer, *adam_initializers])  # , latent_reg_coeff.initializer, *adam_initializers])
    logger = lib.logger.CSVLogger('log.csv', output_dir,
                                  ['iteration', 'cross_entropy_loss', 'acc', 'pred_pos'])

    from tqdm import tqdm

    for iteration in tqdm(range(FLAGS.iterations)):
        sess.run(train_op)
        if iteration % 100 == 99:
            print('Iteration: {}'.format(iteration))
            _loss, _acc, _pred = sess.run([cp_loss, acc, prediction])
            pred_pos = np.mean(_pred[:, 1])
            lib.plot.plot('loss', _loss)
            lib.plot.plot('acc', _acc)
            lib.plot.plot('pred_pos', pred_pos)
            lib.plot.flush()
            lib.plot.tick()

            logger.update_with_dict({
                'iteration': iteration,
                'cross_entropy_loss': _loss,
                'acc': _acc,
                'pred_pos': pred_pos
            })

        if iteration % 1000 == 999:
            np.save(os.path.join(samples_dir, 'latent_codes.npy'), sess.run(latent_codes))
            seqs = sess.run(gen_output)
            np.save(os.path.join(samples_dir, 'seqs.npy'), seqs)
            for i in range(5):
                save_path = os.path.join(samples_dir, 'sample_%d.jpg' % (i))
                lib.plot.plot_weights(convert_to_bits(seqs[i, 150:186, :]), save_path=save_path)

    logger.close()
