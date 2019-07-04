import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import inspect
import datetime
import functools
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from lib.general_utils import Pool
import lib.plot, lib.dataloader, lib.gat_utils, lib.logger, lib.ops.LSTM
import lib.rna_utils
from Model.GAT import GAT

tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_integer('epochs', 200, '')
tf.app.flags.DEFINE_integer('nb_gpus', 1, '')
tf.app.flags.DEFINE_bool('use_clr', True, '')
tf.app.flags.DEFINE_bool('use_momentum', False, '')
tf.app.flags.DEFINE_integer('parallel_processes', 1, '')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 200 * FLAGS.nb_gpus if FLAGS.nb_gpus > 0 else 200
EPOCHS = FLAGS.epochs  # How many iterations to train for
DEVICES = ['/gpu:1'] # ['/gpu:%d' % (i) for i in range(FLAGS.nb_gpus)] if FLAGS.nb_gpus > 0 else ['/cpu:0']
RBP_LIST = ['1_PARCLIP_AGO1234_hg19']
MAX_LEN = 101
N_NODE_EMB = len(lib.dataloader.vocab)
N_EDGE_EMB = len(lib.dataloader.BOND_TYPE)

hp = {
    'arch': 1,
    'learning_rate': 1e-3,
    'dropout_rate': 0.2,
    'use_clr': FLAGS.use_clr,
    'use_momentum': FLAGS.use_momentum,
    'units': [32, 32, 32, 64, 64, 64, 2],
    'heads': [4, 4, 4, 4, 4, 4, 4],
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
    model = GAT(MAX_LEN, N_NODE_EMB, N_EDGE_EMB, DEVICES, **hp)
    rbp = RBP_LIST[idx]
    rbp_output = os.path.join(output_dir, rbp)
    os.makedirs(rbp_output)

    dataset = lib.dataloader.load_clip_seq([rbp])[0]  # load one at a time

    # RNA secondary structures are sparse, therefore separating attention by relation would be a bad idea
    train_bias_mat = lib.rna_utils.adj_to_bias(np.greater(dataset['train_adj_mat'], 0).astype(np.int32))
    test_bias_mat = lib.rna_utils.adj_to_bias(np.greater(dataset['test_adj_mat'], 0).astype(np.int32))
    model.fit((dataset['train_seq'], train_bias_mat), dataset['train_label'],
              EPOCHS, EPOCHS, rbp_output, logging=True)
    all_prediction, acc, auc = model.predict((dataset['test_seq'], test_bias_mat),
                                             BATCH_SIZE, y=dataset['test_label'])
    print('Evaluation on held-out test set, acc: %.3f, auc: %.3f' % (acc, auc))
    q.put({
        'RBP': rbp,
        'acc': acc,
        'auc': auc
    })
    model.delete()
    del dataset


if __name__ == "__main__":

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if FLAGS.output_dir == '':
        output_dir = os.path.join('output', 'GAT', cur_time)
    else:
        output_dir = os.path.join('output', 'GAT', cur_time + '-' + FLAGS.output_dir)

    os.makedirs(output_dir)
    lib.plot.set_output_dir(output_dir)

    # backup python scripts, for future reference
    backup_dir = os.path.join(output_dir, 'backup/')
    os.makedirs(backup_dir)
    shutil.copy(__file__, backup_dir)
    shutil.copy(inspect.getfile(lib.gat_utils), backup_dir)
    shutil.copy(inspect.getfile(lib.rna_utils), backup_dir)
    shutil.copy(inspect.getfile(GAT), backup_dir)
    shutil.copy(inspect.getfile(lib.ops.LSTM), backup_dir)
    shutil.copy(inspect.getfile(lib.dataloader), backup_dir)

    manager = mp.Manager()
    q = manager.Queue()
    pool = Pool(FLAGS.parallel_processes+1)
    logger_thread = pool.apply_async(Logger, (q,))
    pool.map(functools.partial(run_one_rbp, q=q), list(range(len(RBP_LIST))), chunksize=1)

    q.put('kill')  # terminate logger thread
    pool.close()
    pool.join()
