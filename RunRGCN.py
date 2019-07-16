import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import inspect
import datetime
import functools
import tensorflow as tf
import multiprocessing as mp
import lib.plot, lib.dataloader, lib.rgcn_utils, lib.logger, lib.ops.LSTM
from lib.general_utils import Pool
from Model.RGCN import RGCN

tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_integer('epochs', 100, '')
tf.app.flags.DEFINE_integer('nb_gpus', 1, '')
tf.app.flags.DEFINE_bool('use_clr', True, '')
tf.app.flags.DEFINE_bool('use_momentum', False, '')
tf.app.flags.DEFINE_integer('parallel_processes', 1, '')
tf.app.flags.DEFINE_bool('use_attention', False, '')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 200 # * FLAGS.nb_gpus if FLAGS.nb_gpus > 0 else 200
EPOCHS = FLAGS.epochs  # How many iterations to train for
DEVICES =  ['/gpu:%d' % (i) for i in range(FLAGS.nb_gpus)] if FLAGS.nb_gpus > 0 else ['/cpu:0']
RBP_LIST = lib.dataloader.all_rbps
MAX_LEN = 101
N_NODE_EMB = len(lib.dataloader.vocab)
N_EDGE_EMB = len(lib.dataloader.BOND_TYPE)

hp = {
    'learning_rate': 2e-4,
    'dropout_rate': 0.2,
    'use_clr': FLAGS.use_clr,
    'use_momentum': FLAGS.use_momentum,
    'use_attention': FLAGS.use_attention,
    'use_bn': False,
    'units': 64,
    'reuse_weights': True,
    'layers': 40,
    'test_gated_nn': False,
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
    print('training', RBP_LIST[idx])
    model = RGCN(MAX_LEN, N_NODE_EMB, N_EDGE_EMB, DEVICES, **hp)
    rbp = RBP_LIST[idx]
    rbp_output = os.path.join(output_dir, rbp)
    os.makedirs(rbp_output)

    dataset = lib.dataloader.load_clip_seq([rbp])[0] # load one at a time
    model.fit((dataset['train_seq'], dataset['train_adj_mat']), dataset['train_label'],
              EPOCHS, BATCH_SIZE, rbp_output, logging=True)
    all_prediction, acc, auc = model.predict((dataset['test_seq'], dataset['test_adj_mat']),
                                             BATCH_SIZE, y=dataset['test_label'])
    print('Evaluation on held-out test set, acc: %.3f, auc: %.3f' % (acc, auc))
    model.delete()
    del dataset
    lib.plot.reset()
    q.put({
        'RBP': rbp,
        'acc': acc,
        'auc': auc
    })


if __name__ == "__main__":

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if FLAGS.output_dir == '':
        output_dir = os.path.join('output', 'RGCN', cur_time)
    else:
        output_dir = os.path.join('output', 'RGCN', cur_time + '-' + FLAGS.output_dir)

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

    manager = mp.Manager()
    q = manager.Queue()
    pool = Pool(FLAGS.parallel_processes + 1)
    logger_thread = pool.apply_async(Logger, (q,))
    pool.map(functools.partial(run_one_rbp, q=q), list(range(len(RBP_LIST))), chunksize=1)

    q.put('kill')  # terminate logger thread
    pool.close()
    pool.join()
