import os
import sys
import shutil
import inspect
import datetime
import functools
import tensorflow as tf
from importlib import reload
import multiprocessing as mp
import lib.plot, lib.dataloader, lib.rgcn_utils, lib.logger, lib.ops.LSTM, lib.rna_utils
from lib.general_utils import Pool
from Model.RGCN import RGCN

tf.logging.set_verbosity(tf.logging.FATAL)
tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_integer('epochs', 50, '')
tf.app.flags.DEFINE_integer('nb_gpus', 1, '')
tf.app.flags.DEFINE_bool('use_clr', True, '')
tf.app.flags.DEFINE_bool('use_momentum', False, '')
tf.app.flags.DEFINE_integer('parallel_processes', 1, '')
# some experiment settings
tf.app.flags.DEFINE_bool('use_attention', False, '')
tf.app.flags.DEFINE_bool('expr_simplified_attention', False, '')
tf.app.flags.DEFINE_bool('lstm_ggnn', True, '')
tf.app.flags.DEFINE_bool('use_embedding', False, '')
tf.app.flags.DEFINE_bool('augment_features', False, '')
tf.app.flags.DEFINE_bool('use_conv', True, '')
tf.app.flags.DEFINE_integer('nb_layers', 40, '')
tf.app.flags.DEFINE_bool('probabilistic', True, '')
tf.app.flags.DEFINE_string('fold_algo', 'rnaplfold', '')
tf.app.flags.DEFINE_bool('force_folding', False, '')
tf.app.flags.DEFINE_string('train_rbp_id', '10_PARCLIP_ELAVL1A_hg19', '')
FLAGS = tf.app.flags.FLAGS

assert (FLAGS.fold_algo in ['rnafold', 'rnasubopt', 'rnaplfold'])
if FLAGS.fold_algo == 'rnafold':
    assert (FLAGS.probabilistic is False)
if FLAGS.fold_algo == 'rnaplfold':
    assert (FLAGS.probabilistic is True)

BATCH_SIZE = 200  # * FLAGS.nb_gpus if FLAGS.nb_gpus > 0 else 200
TRAIN_RBP_ID = FLAGS.train_rbp_id
EPOCHS = FLAGS.epochs  # How many iterations to train for
DEVICES = ['/gpu:%d' % (i) for i in range(FLAGS.nb_gpus)] if FLAGS.nb_gpus > 0 else ['/cpu:0']
RBP_LIST = lib.dataloader.all_rbps
MAX_LEN = 101
assert (TRAIN_RBP_ID in RBP_LIST)

hp = {
    'learning_rate': 2e-4,
    'dropout_rate': 0.2,
    'use_clr': FLAGS.use_clr,
    'use_momentum': FLAGS.use_momentum,
    'use_attention': FLAGS.use_attention,
    'use_bn': False,
    'units': 32,
    'reuse_weights': True,  # highly suggested
    'layers': FLAGS.nb_layers,
    'test_gated_nn': True,  # must be set to True
    'expr_simplified_attention': FLAGS.expr_simplified_attention,
    'lstm_ggnn': FLAGS.lstm_ggnn,
    'augment_features': FLAGS.augment_features,
    'use_conv': FLAGS.use_conv,
    'probabilistic': FLAGS.probabilistic
}


def Logger(q):
    logger = lib.logger.CSVLogger('results.csv', output_dir, ['fold', 'acc', 'auc'])
    while True:
        msg = q.get()
        if msg == 'kill':
            logger.close()
            break
        logger.update_with_dict(msg)


def run_one_rbp(fold_idx, q):
    fold_output = os.path.join(output_dir, 'fold-%d'%(fold_idx))
    os.makedirs(fold_output)

    outfile = open(os.path.join(fold_output, str(os.getpid())) + ".out", "w")
    sys.stdout = outfile
    sys.stderr = outfile

    print('training fold', fold_idx)
    train_idx, test_idx = dataset['splits'][fold_idx]

    if FLAGS.augment_features:
        hp['features_dim'] = dataset['features'].shape[-1]
    model = RGCN(MAX_LEN, dataset['VOCAB_VEC'].shape[1], len(lib.dataloader.BOND_TYPE),
                 dataset['VOCAB_VEC'], DEVICES, **hp)
    # preparing data for training
    if FLAGS.probabilistic:
        train_data = [dataset['seq'][train_idx], (dataset['adj_mat'][train_idx], dataset['prob_mat'][train_idx])]
    else:
        train_data = [dataset['seq'][train_idx], dataset['adj_mat'][train_idx]]
    if FLAGS.augment_features:
        train_data.append(dataset['features'][train_idx])
    model.fit(train_data, dataset['label'][train_idx], EPOCHS, BATCH_SIZE, fold_output, logging=True)
    # preparing data for testing
    if FLAGS.probabilistic:
        test_data = [dataset['seq'][test_idx], (dataset['adj_mat'][test_idx], dataset['prob_mat'][test_idx])]
    else:
        test_data = [dataset['seq'][test_idx], dataset['adj_mat'][test_idx]]
    if FLAGS.augment_features:
        test_data.append(dataset['features'][test_idx])
    all_prediction, acc, auc = model.predict(test_data, BATCH_SIZE, y=dataset['label'][test_idx])
    print('Evaluation on test set, acc: %.3f, auc: %.3f' % (acc, auc))
    model.delete()
    del dataset
    reload(lib.plot)
    reload(lib.logger)
    q.put({
        'fold': fold_idx,
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
    shutil.copy(inspect.getfile(lib.rna_utils), backup_dir)

    dataset = lib.dataloader.load_clip_seq([TRAIN_RBP_ID], use_embedding=FLAGS.use_embedding, fold_algo=FLAGS.fold_algo,
                                           force_folding=FLAGS.force_folding, augment_features=FLAGS.augment_features,
                                           probabilistic=FLAGS.probabilistic, use_cross_validation=True)[0]  # load one at a time

    manager = mp.Manager()
    q = manager.Queue()
    pool = Pool(FLAGS.parallel_processes + 1)
    logger_thread = pool.apply_async(Logger, (q,))
    pool.map(functools.partial(run_one_rbp, q=q), list(range(len(RBP_LIST))), chunksize=1)

    q.put('kill')  # terminate logger thread
    pool.close()
    pool.join()
