import numpy as np
import os
import sys

vocab = 'ACGTN'
basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import lib.rna_utils
from lib.general_utils import Pool

clip_data_path = os.path.join(basedir, 'Data', 'Clip-seq-data')
all_rbps = [dir for dir in os.listdir(clip_data_path) if dir.split('_')[0].isdigit()]
path_template = os.path.join(basedir, 'Data', 'Clip-seq-data', '{}', '30000', '{}', '{}')

BOND_TYPE = {
    0: 'No Bond',
    1: '5\'UTR to 3\'UTR Covalent Bond',
    2: 'reversed Covalent bond',
    3: 'Hydrogen Bond',
    4: 'reversed hydrogen bond',
}


def load_clip_seq(rbp_list=None, p=None):
    '''
    A multiprocessing pool should be provided in case secondary structures and adjacency matrix
    needs to be computed at the data loading stage
    '''
    if p is None:
        pool = Pool(8)
    else:
        pool = p

    clip_data = []

    rbp_list = all_rbps if rbp_list is None else rbp_list
    for rbp in rbp_list:
        dataset = {}

        all_id, all_seq, adjacency_matrix = lib.rna_utils. \
            fold_rna_from_file(path_template.format(rbp, 'training_sample_0', 'sequences.fa.gz'), pool)

        permute = np.random.permutation(len(all_id))
        dataset['train_label'] = np.array([int(id.split(' ')[-1].split(':')[-1]) for id in all_id])[permute]
        dataset['train_seq'] = np.array([[vocab.index(c) for c in seq] for seq in all_seq])[permute]
        dataset['train_adj_mat'] = adjacency_matrix[permute]

        # # address class imbalance issue
        # all_idx = list(np.where(dataset['train_label'] == 0)[0][:6000]) + list(np.where(dataset['train_label'] == 1)[0])
        # np.random.shuffle(all_idx)
        # dataset['train_label'] = dataset['train_label'][all_idx]
        # dataset['train_seq'] = dataset['train_seq'][all_idx]
        # dataset['train_adj_mat'] = dataset['train_adj_mat'][all_idx]

        all_id, all_seq, adjacency_matrix = lib.rna_utils. \
            fold_rna_from_file(path_template.format(rbp, 'test_sample_0', 'sequences.fa.gz'), pool)

        dataset['test_label'] = np.array([int(id.split(' ')[-1].split(':')[-1]) for id in all_id])
        dataset['test_seq'] = np.array([[vocab.index(c) for c in seq] for seq in all_seq])
        dataset['test_adj_mat'] = adjacency_matrix

        clip_data.append(dataset)

    if p is None:
        pool.close()
        pool.join()

    return clip_data


if __name__ == "__main__":
    dataset = load_clip_seq(['11_CLIPSEQ_ELAVL1_hg19'])
    print(len(np.where(dataset[0]['train_label'] == 1)[0]))
