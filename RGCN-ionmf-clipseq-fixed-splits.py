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
from Model.Dense_RGCN import RGCN

tf.logging.set_verbosity(tf.logging.FATAL)
tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_integer('epochs', 100, '')
tf.app.flags.DEFINE_list('gpu_device', '0,1', '')
tf.app.flags.DEFINE_bool('use_clr', True, '')
tf.app.flags.DEFINE_bool('use_momentum', False, '')
tf.app.flags.DEFINE_integer('parallel_processes', 1, '')
tf.app.flags.DEFINE_bool('share_device', False, '')
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
# switch dataset
tf.app.flags.DEFINE_bool('use_smaller_clip_seq', False, '')
FLAGS = tf.app.flags.FLAGS

if FLAGS.use_smaller_clip_seq:
    lib.dataloader.path_template = lib.dataloader.path_template.replace('30000', '5000')

assert (FLAGS.fold_algo in ['rnafold', 'rnasubopt', 'rnaplfold'])
if FLAGS.fold_algo == 'rnafold':
    assert (FLAGS.probabilistic is False)
if FLAGS.fold_algo == 'rnaplfold':
    assert (FLAGS.probabilistic is True)

BATCH_SIZE = 128  # * FLAGS.nb_gpus if FLAGS.nb_gpus > 0 else 200
EPOCHS = FLAGS.epochs  # How many iterations to train for
DEVICES = ['/gpu:%s' % (device) for device in FLAGS.gpu_device] if len(FLAGS.gpu_device) > 0 else ['/cpu:0']
RBP_LIST = lib.dataloader.all_rbps
MAX_LEN = 101

if FLAGS.share_device:
    DEVICES *= 2
    print('Warning, sharing devices. Make sure you have enough video card memory!')

if FLAGS.parallel_processes > len(DEVICES):
    print('Warning: parallel_processes %d is larger than available devices %d. Adjusting to %d.' % \
          (FLAGS.parallel_processes, len(DEVICES), len(DEVICES)))
    FLAGS.parallel_processes = len(DEVICES)

hp = {
    'learning_rate': 2e-4,
    'dropout_rate': 0.5,
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
    import time
    registered_gpus = {}
    logger = lib.logger.CSVLogger('results.csv', output_dir, ['RBP', 'acc', 'auc'])
    while True:
        msg = q.get()
        print(msg)
        if type(msg) is str and msg == 'kill':
            logger.close()
            break
        elif type(msg) is str and msg.startswith('worker'):
            process_id = int(msg.split('_')[-1])
            if process_id in registered_gpus:
                print(process_id, 'found, returning', registered_gpus[process_id])
                q.put('master_%d_'%(process_id)+registered_gpus[process_id])
            else:
                print(process_id, 'not found')
                all_registered_devices = list(registered_gpus.values())
                from collections import Counter
                c1 = Counter(DEVICES)
                c2 = Counter(all_registered_devices)
                free_devices = list((c1 - c2).elements())
                # free_devices = list(set(DEVICES).difference(set(all_registered_devices)))
                if len(free_devices) > 0:
                    print('free device', free_devices[0])
                    q.put('master_%d_'%(process_id)+free_devices[0])
                    registered_gpus[process_id] = free_devices[0]
                else:
                    print('no free device!')
                    print(registered_gpus)
                    q.put('master_%d_/cpu:0'%(process_id))
        elif type(msg) is dict:
            logger.update_with_dict(msg)
        else:
            q.put(msg)
        time.sleep(np.random.rand()*5)
        # print('here')

def run_one_rbp(idx, q):
    rbp = RBP_LIST[idx]
    rbp_output = os.path.join(output_dir, rbp)
    os.makedirs(rbp_output)

    outfile = open(os.path.join(rbp_output, str(os.getpid())) + ".out", "w")
    sys.stdout = outfile
    sys.stderr = outfile

    import time
    # todo: replace _identity with pid and let logger check if pid still alive
    process_id = mp.current_process()._identity[0]
    print('sending process id', mp.current_process()._identity[0])
    q.put('worker_%d' % (process_id))
    while True:
        msg = q.get()
        if type(msg) is str and msg.startswith('master'):
            print('worker %d received' % (process_id), msg, str(int(msg.split('_')[1])))
            if int(msg.split('_')[1]) == process_id:
                device = msg.split('_')[-1]
                print('Process', mp.current_process(), 'received', device)
                break
        q.put(msg)
        time.sleep(np.random.rand() * 2)

    print('training', RBP_LIST[idx])
    dataset = lib.dataloader.load_clip_seq([rbp], use_embedding=FLAGS.use_embedding, fold_algo=FLAGS.fold_algo,
                                           force_folding=FLAGS.force_folding, augment_features=FLAGS.augment_features,
                                           probabilistic=FLAGS.probabilistic)[0]  # load one at a time
    if FLAGS.augment_features:
        hp['features_dim'] = dataset['train_features'].shape[-1]
    model = RGCN(MAX_LEN, dataset['VOCAB_VEC'].shape[1], len(lib.dataloader.BOND_TYPE),
                 dataset['VOCAB_VEC'], [device], **hp) # getting only 1 device
    # preparing data for training
    if FLAGS.probabilistic:
        train_data = [dataset['train_seq'], (dataset['train_adj_mat'], dataset['train_prob_mat'])]
    else:
        train_data = [dataset['train_seq'], dataset['train_adj_mat']]
    if FLAGS.augment_features:
        train_data.append(dataset['train_features'])
    model.fit(train_data, dataset['train_label'], EPOCHS, BATCH_SIZE, rbp_output, logging=True)
    # preparing data for testing
    if FLAGS.probabilistic:
        test_data = [dataset['test_seq'], (dataset['test_adj_mat'], dataset['test_prob_mat'])]
    else:
        test_data = [dataset['test_seq'], dataset['test_adj_mat']]
    if FLAGS.augment_features:
        test_data.append(dataset['test_features'])
    all_prediction, acc, auc = model.predict(test_data, BATCH_SIZE, y=dataset['test_label'])
    print('Evaluation on held-out test set, acc: %.3f, auc: %.3f' % (acc, auc))
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
