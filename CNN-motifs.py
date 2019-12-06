import os
import numpy as np
import tensorflow as tf
import lib.graphprot_dataloader
from Model.Joint_MRT import JMRT
from lib.general_utils import Pool
import multiprocessing as mp

tf.logging.set_verbosity(tf.logging.FATAL)

lib.graphprot_dataloader._initialize()
BATCH_SIZE = 128
RBP_LIST = lib.graphprot_dataloader.all_rbps
expr_path_list = os.listdir('output/Joint-MRT-Graphprot-debiased')
expr_name = [dirname.split('-')[-1] for dirname in expr_path_list]
DEVICES = ['/gpu:0', '/gpu:1', '/gpu:0', '/gpu:1']

hp = {
    'learning_rate': 2e-4,
    'dropout_rate': 0.2,
    'use_clr': False,
    'use_momentum': False,
    'use_bn': False,
    'units': 32,
    'lstm_encoder': True,
    'mixing_ratio': 0.05,
    'use_ghm': False,
}


def Logger(q):
    import time
    registered_gpus = {}

    while True:
        msg = q.get()
        print(msg)
        if type(msg) is str and msg == 'kill':
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
        else:
            q.put(msg)
        time.sleep(np.random.rand() * 5)


def plot_one_rbp(rbp):
    print('analyzing', rbp)
    if rbp not in expr_name:
        return

    # original dataset
    dataset = \
        lib.graphprot_dataloader.load_clip_seq([rbp], use_embedding=False,
                                               load_mat=False,
                                               nucleotide_label=True)[0]  # load one at a time
    expr_path = expr_path_list[expr_name.index(rbp)]
    if not os.path.exists(os.path.join('output/Joint-MRT-Graphprot-debiased', expr_path, 'splits.npy')):
        print('Warning, fold split file is missing; skipping...')
        return
    dataset['splits'] = np.load(os.path.join('output/Joint-MRT-Graphprot-debiased', expr_path, 'splits.npy'),
                                allow_pickle=True)

    full_expr_path = os.path.join('output/Joint-MRT-Graphprot-debiased', expr_path)
    motif_dir = os.path.join(full_expr_path, 'old_motifs')
    if not os.path.exists(motif_dir):
        os.makedirs(motif_dir)

    import time
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

    fold_idx = 0
    fold_output = os.path.join(full_expr_path, 'fold%d' % (fold_idx))
    if os.path.exists(fold_output):
        model = JMRT(dataset['VOCAB_VEC'].shape[1],
                     # excluding no bond
                     dataset['VOCAB_VEC'], device, **hp)

        checkpoint_path = tf.train.latest_checkpoint(os.path.join(fold_output, 'checkpoints'))
        if checkpoint_path is None:
            print('Warning, latest checkpoint of %s is None...' % (fold_output))
            return
        try:
            model.load(checkpoint_path)
        except:
            print('cannot load back weights; skipping...')
            return

        all_data = [dataset['seq'], dataset['segment_size'], dataset['raw_seq']]

        model.extract_sequence_motifs(
            all_data, dataset['label'], save_path=motif_dir,
            max_examples=4000, mer_size=10)

        model.delete()
    else:
        print('Warning, %s doesd not exist...' % (fold_output))


if __name__ == "__main__":
    manager = mp.Manager()
    q = manager.Queue()
    pool = Pool(4 + 1)
    logger_thread = pool.apply_async(Logger, (q,))
    pool = Pool(4)
    pool.map(plot_one_rbp, RBP_LIST)
