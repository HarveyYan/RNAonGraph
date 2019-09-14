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
import lib.plot, lib.graphprot_dataloader, lib.rgcn_utils, lib.logger, lib.ops.LSTM, lib.rna_utils
from lib.general_utils import Pool
from Model.Joint_SMRGCN import JSMRGCN

tf.logging.set_verbosity(tf.logging.FATAL)
tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_integer('epochs', 50, '')
tf.app.flags.DEFINE_list('gpu_device', '0,1', '')
tf.app.flags.DEFINE_bool('use_clr', True, '')
tf.app.flags.DEFINE_bool('use_momentum', False, '')
tf.app.flags.DEFINE_integer('parallel_processes', 1, '')
tf.app.flags.DEFINE_integer('batch_size', 128, '')
tf.app.flags.DEFINE_bool('share_device', False, '')
# some experiment settings
tf.app.flags.DEFINE_bool('use_attention', False, '')
tf.app.flags.DEFINE_bool('expr_simplified_attention', False, '')
tf.app.flags.DEFINE_bool('lstm_ggnn', True, '')
tf.app.flags.DEFINE_bool('use_embedding', False, '')
tf.app.flags.DEFINE_integer('nb_layers', 40, '')
tf.app.flags.DEFINE_bool('probabilistic', True, '')
tf.app.flags.DEFINE_string('fold_algo', 'rnaplfold', '')
# major changes !
tf.app.flags.DEFINE_string('train_rbp_id', 'PARCLIP_PUM2', '')
tf.app.flags.DEFINE_bool('force_folding', False, '')
FLAGS = tf.app.flags.FLAGS

assert (FLAGS.fold_algo in ['rnafold', 'rnasubopt', 'rnaplfold'])
if FLAGS.fold_algo == 'rnafold':
    assert (FLAGS.probabilistic is False)
if FLAGS.fold_algo == 'rnaplfold':
    assert (FLAGS.probabilistic is True)

lib.graphprot_dataloader._initialize()
TRAIN_RBP_ID = FLAGS.train_rbp_id
BATCH_SIZE = FLAGS.batch_size
EPOCHS = FLAGS.epochs  # How many iterations to train for
DEVICES = ['/gpu:%s' % (device) for device in FLAGS.gpu_device] if len(FLAGS.gpu_device) > 0 else ['/cpu:0']
RBP_LIST = lib.graphprot_dataloader.all_rbps
assert (TRAIN_RBP_ID in RBP_LIST)

if FLAGS.share_device:
    DEVICES *= 2
    print('Warning, sharing devices. Make sure you have enough video card memory!')

if FLAGS.parallel_processes > len(DEVICES):
    print('Warning: parallel_processes %d is larger than available devices %d. Adjusting to %d.' % \
          (FLAGS.parallel_processes, len(DEVICES), len(DEVICES)))
    FLAGS.parallel_processes = len(DEVICES)

hp = {
    'learning_rate': 2e-4,
    'dropout_rate': 0.2,
    'use_clr': FLAGS.use_clr,
    'use_momentum': FLAGS.use_momentum,
    'use_bn': False,
    'units': 32,
    'reuse_weights': True,  # highly suggested
    'layers': FLAGS.nb_layers,
    'lstm_ggnn': FLAGS.lstm_ggnn,
    'probabilistic': FLAGS.probabilistic,
    'lstm_encoder': True,
}


def Logger(q):
    import time
    all_auc = []
    registered_gpus = {}
    logger = lib.logger.CSVLogger('results.csv', output_dir, ['fold', 'nuc_acc', 'pos_acc', 'seq_acc', 'auc'])
    while True:
        msg = q.get()
        print(msg)
        if type(msg) is str and msg == 'kill':
            logger.close()
            print('%s ROC AUC: %.3f\u00B1%.3f' % (TRAIN_RBP_ID, np.mean(all_auc), np.std(all_auc)))
            break
        elif type(msg) is str and msg.startswith('worker'):
            process_id = int(msg.split('_')[-1])
            if process_id in registered_gpus:
                print(process_id, 'found, returning', registered_gpus[process_id])
                q.put('master_%d_' % (process_id) + registered_gpus[process_id])
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
                    q.put('master_%d_' % (process_id) + free_devices[0])
                    registered_gpus[process_id] = free_devices[0]
                else:
                    print('no free device!')
                    print(registered_gpus)
                    q.put('master_%d_/cpu:0' % (process_id))
        elif type(msg) is dict:
            logger.update_with_dict(msg)
            all_auc.append(msg['auc'])
        else:
            q.put(msg)
        time.sleep(np.random.rand() * 5)
        # print('here')


def run_one_rbp(idx, q):
    fold_output = os.path.join(output_dir, 'fold%d' % (idx))
    os.makedirs(fold_output)

    # outfile = open(os.path.join(fold_output, str(os.getpid())) + ".out", "w")
    # sys.stdout = outfile
    # sys.stderr = outfile
    #
    # import time
    # # todo: replace _identity with pid and let logger check if pid still alive
    # process_id = mp.current_process()._identity[0]
    # print('sending process id', mp.current_process()._identity[0])
    # q.put('worker_%d' % (process_id))
    # while True:
    #     msg = q.get()
    #     if type(msg) is str and msg.startswith('master'):
    #         print('worker %d received' % (process_id), msg, str(int(msg.split('_')[1])))
    #         if int(msg.split('_')[1]) == process_id:
    #             device = msg.split('_')[-1]
    #             print('Process', mp.current_process(), 'received', device)
    #             break
    #     q.put(msg)
    #     time.sleep(np.random.rand() * 2)

    print('training fold', idx)
    train_idx, test_idx = dataset['splits'][idx]
    model = JSMRGCN(dataset['VOCAB_VEC'].shape[1], len(lib.graphprot_dataloader.BOND_TYPE) - 1,
                    # excluding no bond
                    dataset['VOCAB_VEC'], ['/gpu:0'], **hp)

    train_data = [dataset['seq'][train_idx], dataset['all_data'][train_idx],
                  dataset['all_row_col'][train_idx], dataset['segment_size'][train_idx]]
    model.fit(train_data, dataset['label'][train_idx], EPOCHS, BATCH_SIZE, fold_output, logging=True)

    test_data = [dataset['seq'][test_idx], dataset['all_data'][test_idx],
                 dataset['all_row_col'][test_idx], dataset['segment_size'][test_idx]]

    cost, acc, auc = model.evaluate(test_data, dataset['label'][test_idx], BATCH_SIZE)

    print('Evaluation (with masking) on held-out test set, acc (pos, seq): %s, auc: %.3f' % (acc, auc))

    model.delete()
    reload(lib.plot)
    reload(lib.logger)
    q.put({
        'fold': idx,
        'nuc_acc': acc[0],
        'pos_acc': acc[1],
        'seq_acc': acc[2],
        'auc': auc
    })


if __name__ == "__main__":

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if FLAGS.output_dir == '':
        output_dir = os.path.join('output', 'SMRGCN-Graphprot', cur_time)
    else:
        output_dir = os.path.join('output', 'SMRGCN-Graphprot', cur_time + '-' + FLAGS.output_dir)

    os.makedirs(output_dir)
    lib.plot.set_output_dir(output_dir)

    # backup python scripts, for future reference
    backup_dir = os.path.join(output_dir, 'backup/')
    os.makedirs(backup_dir)
    shutil.copy(__file__, backup_dir)
    shutil.copy(inspect.getfile(lib.rgcn_utils), backup_dir)
    shutil.copy(inspect.getfile(JSMRGCN), backup_dir)
    shutil.copy(inspect.getfile(lib.ops.LSTM), backup_dir)
    shutil.copy(inspect.getfile(lib.graphprot_dataloader), backup_dir)
    shutil.copy(inspect.getfile(lib.rna_utils), backup_dir)

    dataset = \
        lib.graphprot_dataloader.load_clip_seq([TRAIN_RBP_ID], use_embedding=FLAGS.use_embedding,
                                               fold_algo=FLAGS.fold_algo, force_folding=FLAGS.force_folding,
                                               probabilistic=FLAGS.probabilistic,
                                               nucleotide_label=True)[0]  # load one at a time
    np.save(os.path.join(output_dir, 'splits.npy'), dataset['splits'])
    run_one_rbp(0, None)
    manager = mp.Manager()
    q = manager.Queue()
    pool = Pool(FLAGS.parallel_processes + 1)
    logger_thread = pool.apply_async(Logger, (q,))
    pool.map(functools.partial(run_one_rbp, q=q), list(range(len(dataset['splits']))), chunksize=1)

    q.put('kill')  # terminate logger thread
    pool.close()
    pool.join()
