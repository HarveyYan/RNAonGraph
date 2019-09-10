import os
import sys
import shutil
import inspect
import datetime
import functools
import numpy as np
import tensorflow as tf
from importlib import reload
import multiprocessing as mp
import lib.plot, lib.dataloader, lib.rgcn_utils, lib.logger, lib.ops.LSTM
from lib.general_utils import Pool
from Model.Dense_Bayesian_RGCN import RGCN

tf.logging.set_verbosity(tf.logging.FATAL)
tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_integer('epochs', 200, '')
tf.app.flags.DEFINE_integer('nb_gpus', 1, '')
tf.app.flags.DEFINE_bool('use_clr', True, '')
tf.app.flags.DEFINE_bool('use_momentum', False, '')
tf.app.flags.DEFINE_integer('parallel_processes', 1, '')
# some experiment settings
tf.app.flags.DEFINE_bool('use_attention', False, '')
tf.app.flags.DEFINE_bool('expr_simplified_attention', False, '')
tf.app.flags.DEFINE_bool('lstm_ggnn', True, '')
tf.app.flags.DEFINE_bool('use_embedding', False, '')
tf.app.flags.DEFINE_bool('use_conv', True, '')
tf.app.flags.DEFINE_integer('nb_layers', 40, '')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 128  # * FLAGS.nb_gpus if FLAGS.nb_gpus > 0 else 200
EPOCHS = FLAGS.epochs  # How many iterations to train for
DEVICES = ['/gpu:%d' % (i) for i in range(FLAGS.nb_gpus)] if FLAGS.nb_gpus > 0 else ['/cpu:0']
RBP_LIST = lib.dataloader.all_rbps
MAX_LEN = 101

hp = {
    'learning_rate': 2e-4,
    'dropout_rate': 0.5,
    'use_clr': FLAGS.use_clr,
    'use_momentum': FLAGS.use_momentum,
    'use_bn': False,
    'units': 32,
    'reuse_weights': True,  # highly suggested
    'layers': FLAGS.nb_layers,
    'test_gated_nn': True,  # must be set to True
    'lstm_ggnn': FLAGS.lstm_ggnn,
    'use_conv': FLAGS.use_conv,
}


def Logger(q):
    logger = lib.logger.CSVLogger('rbp-results.csv', output_dir, ['RBP', 'acc', 'auc'])
    while True:
        msg = q.get()
        if msg == 'kill':
            logger.close()
            break
        logger.update_with_dict(msg)


def run_one_rbp(idx, q):
    rbp = RBP_LIST[idx]
    rbp_output = os.path.join(output_dir, rbp)
    os.makedirs(rbp_output)

    # outfile = open(os.path.join(rbp_output, str(os.getpid())) + ".out", "w")
    # sys.stdout = outfile
    # sys.stderr = outfile

    print('training', RBP_LIST[idx])
    dataset = lib.dataloader.load_clip_seq([rbp], use_embedding=FLAGS.use_embedding, fold_algo='rnafold')[
        0]  # load one at a time

    model = RGCN(MAX_LEN, dataset['VOCAB_VEC'].shape[1], len(lib.dataloader.BOND_TYPE),
                 dataset['VOCAB_VEC'], DEVICES, **hp)
    train_data = [dataset['train_seq'], dataset['train_adj_mat']]

    load_weights = True
    if load_weights:
        weight_path = 'output/RGCN/20190903-204843-MCD0.5/%s/pretrain/checkpoints' % (rbp)
        model.load(tf.train.latest_checkpoint(weight_path))
    else:
        # pretraining phase
        pretrain_dir = os.path.join(rbp_output, 'pretrain')
        os.makedirs(pretrain_dir)
        model.pretrain(train_data, dataset['train_label'], EPOCHS, BATCH_SIZE, pretrain_dir, logging=True)
    reload(lib.plot)
    reload(lib.logger)

    # start stochastic training
    pool = Pool(30)
    model.train_sampled_structures(dataset['train_seq'], dataset['train_label'], 10,
                                   BATCH_SIZE, output_dir, pool, 10, 10)
    averaged_preds, averaged_std, auc, acc = \
        model.evaluate_stochastic(dataset['test_seq'], dataset['test_label'],
                                  BATCH_SIZE * 10, pool, 10, 10)
    np.save('averaged_preds.npy', averaged_preds)
    np.save('averaged_std.npy', averaged_std)
    pool.close()
    pool.join()

    model.delete()
    del dataset
    reload(lib.plot)
    reload(lib.logger)
    q.put({
        'RBP': rbp,
        'acc': acc,
        'auc': auc
    })


if __name__ == "__main__":

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if FLAGS.output_dir == '':
        output_dir = os.path.join('output', 'Bayesian-RGCN', cur_time)
    else:
        output_dir = os.path.join('output', 'Bayesian-RGCN', cur_time + '-' + FLAGS.output_dir)

    os.makedirs(output_dir)
    lib.plot.set_output_dir(output_dir)

    # backup python scripts, for future reference
    backup_dir = os.path.join(output_dir, 'backup/')
    os.makedirs(backup_dir)
    shutil.copy(__file__, backup_dir)
    shutil.copy(inspect.getfile(lib.rgcn_utils), backup_dir)
    shutil.copy(inspect.getfile(RGCN), backup_dir)
    shutil.copy(inspect.getfile(lib.ops.LSTM), backup_dir)
    shutil.copy(inspect.getfile(lib.dataloader), backup_dir)

    run_one_rbp(0, None)
    exit()

    manager = mp.Manager()
    q = manager.Queue()
    pool = Pool(FLAGS.parallel_processes + 1)
    logger_thread = pool.apply_async(Logger, (q,))
    pool.map(functools.partial(run_one_rbp, q=q), list(range(len(RBP_LIST))), chunksize=1)

    q.put('kill')  # terminate logger thread
    pool.close()
    pool.join()
