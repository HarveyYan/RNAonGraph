import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import lib.graphprot_dataloader, lib.rna_utils
from Model.Joint_ada_sampling_model import JointAdaModel
from lib.general_utils import Pool
import multiprocessing as mp
import pandas as pd

tf.logging.set_verbosity(tf.logging.FATAL)

lib.graphprot_dataloader._initialize()
BATCH_SIZE = 128
RBP_LIST = lib.graphprot_dataloader.all_rbps
expr_path_list = os.listdir('output/Joint-ada-sampling-debiased')
expr_name = [dirname.split('-')[-1] for dirname in expr_path_list]
DEVICES = ['/gpu:0', '/gpu:1', '/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3', '/gpu:2', '/gpu:3']

hp = {
    'learning_rate': 2e-4,
    'dropout_rate': 0.2,
    'use_clr': True,
    'use_momentum': False,
    'use_bn': False,
    'units': 32,
    'reuse_weights': True,  # highly suggested
    'layers': 10,
    'lstm_ggnn': True,
    'probabilistic': True,
    'mixing_ratio': 0.05,
    'use_ghm': True,
    'use_attention': False,
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

    pool = Pool(8)

    # original dataset
    dataset = lib.graphprot_dataloader.load_clip_seq([rbp], p=pool, load_mat=False, modify_leaks=False, nucleotide_label=True)[0]
    expr_path = expr_path_list[expr_name.index(rbp)]
    if not os.path.exists(os.path.join('output/Joint-ada-sampling-debiased', expr_path, 'splits.npy')):
        print('Warning, fold split file is missing; skipping...')
        return
    dataset['splits'] = np.load(os.path.join('output/Joint-ada-sampling-debiased', expr_path, 'splits.npy'),
                                allow_pickle=True)

    full_expr_path = os.path.join('output/Joint-ada-sampling-debiased', expr_path)
    motif_dir = os.path.join(full_expr_path, 'wholeseq_high_gradient_ranked_motifs_100_4000')
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
        model = JointAdaModel(dataset['VOCAB_VEC'].shape[1],
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

        _, test_idx = dataset['splits'][0]

        test_pred_res = pd.read_csv(os.path.join(full_expr_path, 'fold%d'%(fold_idx), 'predictions.csv'))
        test_pred_ranked_idx = np.argsort(np.array(test_pred_res['pred_pos']))[::-1]

        # pos_idx = np.where(np.array([np.max(label) for label in dataset['label']]) == 1)[0]
        all_data = [dataset['seq'][test_idx][test_pred_ranked_idx],
                    dataset['segment_size'][test_idx][test_pred_ranked_idx],
                    dataset['raw_seq'][test_idx][test_pred_ranked_idx]]
        print(rbp, 'number of positive examples', len(all_data[0]))

        model.rank_extract_motifs(
            all_data, save_path=motif_dir, mer_size=10, max_examples=100, p=pool)

        model.delete()
        pool.close()
        pool.join()
    else:
        print('Warning, %s doesd not exist...' % (fold_output))


if __name__ == "__main__":
    manager = mp.Manager()
    q = manager.Queue()
    pool = Pool(8 + 1)
    logger_thread = pool.apply_async(Logger, (q,))
    # the first batch

    pool.map(plot_one_rbp, ['PTBv1', 'CLIPSEQ_SFRS1', 'PARCLIP_QKI', 'PARCLIP_PUM2',
                            'ICLIP_TDP43', 'PARCLIP_FUS', 'PARCLIP_TAF15', 'PARCLIP_IGF2BP123'])

    # second batch
    pool.map(plot_one_rbp, ['CLIPSEQ_AGO2', 'PARCLIP_MOV10_Sievers', 'PARCLIP_AGO1234', 'ZC3H7B_Baltz2012',
                            'CAPRIN1_Baltz2012', 'C22ORF28_Baltz2012', 'C17ORF85_Baltz2012', 'ALKBH5_Baltz2012'])
    # last batch
    # pool.map(plot_one_rbp, ['CLIPSEQ_ELAVL1', 'ICLIP_HNRNPC', 'ICLIP_TIA1', 'ICLIP_TIAL1',
    #                         'PARCLIP_ELAVL1', 'PARCLIP_ELAVL1A', 'PARCLIP_EWSR1', 'PARCLIP_HUR'])



