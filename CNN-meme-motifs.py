import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import lib.graphprot_dataloader
from Model.Joint_MRT import JMRT
from lib.general_utils import Pool
import multiprocessing as mp
import matplotlib.pyplot as plt
import subprocess

tf.logging.set_verbosity(tf.logging.FATAL)

lib.graphprot_dataloader._initialize()
BATCH_SIZE = 128
RBP_LIST = lib.graphprot_dataloader.all_rbps
expr_path_list = os.listdir('output/Joint-MRT-Graphprot-debiased')
expr_name = [dirname.split('-')[-1] for dirname in expr_path_list]
DEVICES = ['/gpu:8'] * 4 + ['/gpu:9'] * 4

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
    motif_dir = os.path.join(full_expr_path, 'meme-motif10-slide1')
    if not os.path.exists(motif_dir):
        os.makedirs(motif_dir)

    if not os.path.exists(os.path.join(motif_dir, 'all_mers.npy')):
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

        pos_idx = np.where(np.array([np.max(label) for label in dataset['label']]) == 1)[0]
        all_data = [
            dataset['seq'][pos_idx],
            dataset['segment_size'][pos_idx],
            dataset['raw_seq'][pos_idx]]

        pos_preds = model.predict(all_data, BATCH_SIZE)[:, 1]
        ranked_idx = np.argsort(pos_preds)[::-1]
        print(rbp, 'top pos prediction average [100, 500, 1000, 2000]', pos_preds[ranked_idx][:100].mean(), pos_preds[ranked_idx][:500].mean(),
              pos_preds[ranked_idx][:1000].mean(), pos_preds[ranked_idx][:2000].mean())
        all_data = [dataset['seq'][pos_idx][ranked_idx],
                    dataset['segment_size'][pos_idx][ranked_idx],
                    dataset['raw_seq'][pos_idx][ranked_idx]]
        print(rbp, 'number of positive examples', len(all_data[0]))

        def compute_ig(_node_tensor, _segment, _raw_seq, interp_steps=100, mer_size=10):
            _meshed_node_tensor = np.array([model.embedding_vec[idx] for idx in _node_tensor])
            _meshed_reference_input = np.zeros_like(_meshed_node_tensor)
            new_node_tensor = []
            for i in range(0, interp_steps + 1):
                new_node_tensor.append(
                    _meshed_reference_input + i / interp_steps * (_meshed_node_tensor - _meshed_reference_input))

            feed_dict = {
                model.node_tensor: np.concatenate(np.array(new_node_tensor), axis=0),
                model.max_len: _segment,
                model.segment_length: [_segment] * (interp_steps + 1),
                model.is_training_ph: False
            }

            grads = model.sess.run(model.g_nodes, feed_dict).reshape((interp_steps + 1, _segment, 4))
            grads = (grads[:-1] + grads[1:]) / 2.0
            node_scores = np.sum(np.average(grads, axis=0) * (_meshed_node_tensor - _meshed_reference_input), axis=-1)
            mers, mer_scores = [], []
            for start in range(len(node_scores) - mer_size + 1):
                mer_scores.append(np.sum(node_scores[start: start + mer_size]))
                mers.append(_raw_seq[start: start + mer_size])
            return mers, mer_scores

        counter = 0
        all_mers, all_mer_scores = [], []
        for _node_tensor, _segment_size, _raw_seq in zip(*all_data):
            if counter > 1999:
                break
            mers, mer_scores = compute_ig(_node_tensor, _segment_size, _raw_seq)
            all_mers.append(mers)
            all_mer_scores.append(mer_scores)
            counter += 1

        np.save(os.path.join(motif_dir, 'all_mers.npy'), all_mers)
        np.save(os.path.join(motif_dir, 'all_mer_scores.npy'), all_mer_scores)

        fig, axes = plt.subplots(10, 10, figsize=(40, 40), dpi=350)
        for i in range(10):
            for j in range(10):
                axes[i][j].plot(all_mer_scores[10 * i + j])
        plt.savefig(os.path.join(motif_dir, 'ig_maps.png'), dpi=350)
        plt.close(fig)
        plt.hist(np.concatenate(all_mer_scores, axis=0), bins=50000, range=(-0.5, 0.5))
        plt.savefig(os.path.join(motif_dir, 'ig_histograms.png'), dpi=350)

        model.delete()
    else:
        all_mers = np.load(os.path.join(motif_dir, 'all_mers.npy'), allow_pickle=True)
        all_mer_scores = np.load(os.path.join(motif_dir, 'all_mer_scores.npy'), allow_pickle=True)

    prev_count = 0
    thres_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if rbp == 'ALKBH5_Baltz2012':
        thres_list = [0.01 * thres for thres in thres_list]
    for thres in reversed(thres_list):
        best_mers = np.concatenate(all_mers, axis=0)[np.concatenate(all_mer_scores, axis=0) > thres]
        print('protein %s, threshold %f gives %d mers out of total %d' % (rbp, thres, len(best_mers), sum([len(mers) for mers in all_mers])))
        if len(best_mers) == prev_count or len(best_mers) < 10:
            continue
        else:
            prev_count = len(best_mers)
        thres_dir = os.path.join(motif_dir, 'meme-%f-%d-bm-applied' % (thres, len(best_mers)))
        if not os.path.exists(thres_dir):
            os.makedirs(thres_dir)
        fasta_path = os.path.join(thres_dir, 'fasta.fa')
        with open(fasta_path, 'w') as f:
            for i, seq in enumerate(best_mers):
                print('>{}'.format(i), file=f)
                print(seq.upper().replace('T', 'U'), file=f)

        meme_order = 'meme %s -rna -oc %s -mod zoops -nostatus -nmotifs 5 -minw 4 -maxw 10 -objfun classic -markov_order 0 -bfile ~/bm.txt' \
                     % (fasta_path, thres_dir)
        subprocess.check_output(meme_order, shell=True)
        mast_order = 'mast %s %s -oc . -nostatus' % (os.path.join(thres_dir, 'meme.xml'), fasta_path)
        subprocess.check_output(mast_order, shell=True)


if __name__ == "__main__":
    manager = mp.Manager()
    q = manager.Queue()
    pool = Pool(24 + 1)

    logger_thread = pool.apply_async(Logger, (q,))
    pool.map(plot_one_rbp, ['CAPRIN1_Baltz2012', 'PARCLIP_IGF2BP123', 'CLIPSEQ_AGO2', 'PARCLIP_AGO1234',
                            'ZC3H7B_Baltz2012', 'PARCLIP_MOV10_Sievers', 'C22ORF28_Baltz2012', 'C17ORF85_Baltz2012',
                            'PARCLIP_PUM2', 'PARCLIP_QKI', 'CLIPSEQ_SFRS1', 'PTBv1',
                            'ICLIP_TDP43', 'PARCLIP_FUS', 'PARCLIP_TAF15', 'ALKBH5_Baltz2012',
                            'CLIPSEQ_ELAVL1', 'ICLIP_HNRNPC', 'PARCLIP_EWSR1', 'ICLIP_TIA1',
                            'ICLIP_TIAL1', 'PARCLIP_ELAVL1', 'PARCLIP_ELAVL1A', 'PARCLIP_HUR'
                            ])

