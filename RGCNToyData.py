import os
import sys
import shutil
import inspect
import datetime
import numpy as np
import tensorflow as tf
import lib.plot, lib.dataloader, lib.rgcn_utils, lib.ops.LSTM
from Model.RGCN import RGCN
from Model.RNATracker import RNATracker

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_integer('epochs', 100, '')
tf.app.flags.DEFINE_integer('nb_gpus', 1, '')
tf.app.flags.DEFINE_bool('use_clr', True, '')
tf.app.flags.DEFINE_bool('use_momentum', False, '')
tf.app.flags.DEFINE_bool('use_attention', False, '')
tf.app.flags.DEFINE_bool('expr_simplified_attention', False, '')
tf.app.flags.DEFINE_bool('lstm_ggnn', False, '')
tf.app.flags.DEFINE_bool('use_embedding', False, '')

tf.app.flags.DEFINE_bool('return_label', False, '')
tf.app.flags.DEFINE_bool('load_hairpin', False, '')
tf.app.flags.DEFINE_string('element_symbol', 'm', 'must be specified if load_hairpin is turned off')

tf.app.flags.DEFINE_bool('use_baseline_rnatracker', False, '')
tf.app.flags.DEFINE_bool('use_const_seq', False, '')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 200 * FLAGS.nb_gpus if FLAGS.nb_gpus > 0 else 200
EPOCHS = FLAGS.epochs
DEVICES = ['/gpu:%d' % (i) for i in range(FLAGS.nb_gpus)] if FLAGS.nb_gpus > 0 else ['/cpu:0']
MAX_LEN = 101

if FLAGS.use_baseline_rnatracker:
    hp = {
        'learning_rate': 1e-3,
        'dropout_rate': 0.2,
        'use_clr': FLAGS.use_clr,
        'use_momentum': FLAGS.use_momentum,
        'use_attention': FLAGS.use_attention,
        'use_bn': False,
    }
else:
    hp = {
        'learning_rate': 1e-3,
        'dropout_rate': 0.2,
        'use_clr': FLAGS.use_clr,
        'use_momentum': FLAGS.use_momentum,
        'use_attention': FLAGS.use_attention,
        'use_bn': False,
        'units': 32,
        'reuse_weights': True,
        'layers': 20,
        'test_gated_nn': True,
        'expr_simplified_attention': FLAGS.expr_simplified_attention,
        'lstm_ggnn': FLAGS.lstm_ggnn,
    }

if __name__ == "__main__":

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cur_time += ('-rnatracker' if FLAGS.use_baseline_rnatracker else '')
    cur_time += ('-%s' % (FLAGS.output_dir) if len(FLAGS.output_dir) > 0 else '')

    output_dir = os.path.join('output', 'ToyData', 'hairpin' if FLAGS.load_hairpin else FLAGS.element_symbol,
                              'label' if FLAGS.return_label else 'annotation', cur_time)

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

    outfile = open(os.path.join(output_dir, str(os.getpid())) + ".out", "w")
    sys.stdout = outfile
    sys.stderr = outfile

    dataset = lib.dataloader.load_toy_data(FLAGS.load_hairpin, FLAGS.return_label,
                                           element_symbol=FLAGS.element_symbol)
    if FLAGS.use_baseline_rnatracker:
        model = RNATracker
        node_dim = 3  # .()
        embedding_vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32)
        train_data, test_data = dataset['train_struct'], dataset['test_struct']
    else:
        model = RGCN
        node_dim = dataset['VOCAB_VEC'].shape[1]
        embedding_vec = dataset['VOCAB_VEC']
        if FLAGS.use_const_seq:
            train_data, test_data = (np.zeros_like(dataset['train_seq']), dataset['train_adj_mat']), \
                                    (np.zeros_like(dataset['test_seq']), dataset['test_adj_mat'])
        else:
            train_data, test_data = (dataset['train_seq'], dataset['train_adj_mat']), \
                                    (dataset['test_seq'], dataset['test_adj_mat'])

    model = model(MAX_LEN, node_dim, len(lib.dataloader.BOND_TYPE),
                  embedding_vec, DEVICES, FLAGS.return_label, **hp)
    model.fit(train_data, dataset['train_label'], EPOCHS, BATCH_SIZE, output_dir, logging=True)
    all_prediction, acc, auc = model.predict(test_data, BATCH_SIZE, y=dataset['test_label'])
    print('Evaluation on held-out test set, acc: %.3f, auc: %.3f' % (acc, auc))
    model.delete()
    del dataset
    lib.plot.reset()
