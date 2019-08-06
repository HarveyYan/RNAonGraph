import numpy as np
import os
import sys
from gensim.models import Word2Vec

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


def load_clip_seq(rbp_list=None, p=None, **kwargs):
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
        dataset['train_adj_mat'] = adjacency_matrix[permute]

        use_embedding = kwargs.get('use_embedding', False)
        kmer_len = kwargs.get('kmer_len', 3)
        window = kwargs.get('window', 12)
        emb_size = kwargs.get('emb_size', 50)
        if use_embedding:
            path = os.path.join(basedir, path_template.format(rbp, 'training_sample_0', 'word2vec_%d_%d_%d.obj' % (kmer_len, window, emb_size)))
            if not os.path.exists(path):
                pretrain_word2vec(all_seq, kmer_len, window, emb_size, path)
            word2vec_model = Word2Vec.load(path)
            VOCAB = ['NOT_FOUND'] + list(word2vec_model.wv.vocab.keys())
            VOCAB_VEC = np.concatenate([np.zeros((1, emb_size)).astype(np.float32), word2vec_model.wv.vectors], axis=0)
            kmers = get_kmers(all_seq, kmer_len)
            dataset['train_seq'] = np.array([[VOCAB.index(c) for c in seq] for seq in kmers])[permute]
        else:
            VOCAB = ['NOT_FOUND', 'A', 'C', 'G', 'T', 'N']
            VOCAB_VEC = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]).astype(
                np.float32)
            dataset['train_seq'] = np.array([[VOCAB.index(c) for c in seq] for seq in all_seq])[permute]

        dataset['VOCAB'] = VOCAB
        dataset['VOCAB_VEC'] = VOCAB_VEC

        # # address class imbalance issue
        # all_idx = list(np.where(dataset['train_label'] == 0)[0][:6000]) + list(np.where(dataset['train_label'] == 1)[0])
        # np.random.shuffle(all_idx)
        # dataset['train_label'] = dataset['train_label'][all_idx]
        # dataset['train_seq'] = dataset['train_seq'][all_idx]
        # dataset['train_adj_mat'] = dataset['train_adj_mat'][all_idx]

        all_id, all_seq, adjacency_matrix = lib.rna_utils. \
            fold_rna_from_file(path_template.format(rbp, 'test_sample_0', 'sequences.fa.gz'), pool)

        dataset['test_label'] = np.array([int(id.split(' ')[-1].split(':')[-1]) for id in all_id])
        dataset['test_adj_mat'] = adjacency_matrix

        if use_embedding:
            kmers = get_kmers(all_seq, kmer_len)
            dataset['test_seq'] = np.array([[VOCAB.index(c) if c in VOCAB else 0 for c in seq] for seq in kmers])
        else:
            dataset['test_seq'] = np.array([[VOCAB.index(c) if c in VOCAB else 0 for c in seq] for seq in all_seq])

        clip_data.append(dataset)

    if p is None:
        pool.close()
        pool.join()

    return clip_data


def get_kmers(seqs, kmer_len):
    sentence = []
    left = kmer_len // 2
    right = kmer_len - left
    for seq in seqs:
        kmers = []
        length = len(seq)
        seq = 'N'*left + seq + 'N' * (right - 1)
        for j in range(left, left + length):
            kmers.append(seq[j-left: j + right])
        sentence.append(kmers)
    return sentence


def pretrain_word2vec(seqs, kmer_len, window, embedding_size, save_path):
    print('word2vec pretaining')
    sentence = get_kmers(seqs, kmer_len)
    model = Word2Vec(sentence, window=window, size=embedding_size,
                     min_count=0, workers=15)
    # to capture as much dependency as possible
    model.train(sentence, total_examples=len(sentence), epochs=100)
    model.save(save_path)



def load_toy_data(p=None):
    if p is None:
        pool = Pool(8)
    else:
        pool = p

    dataset = {}

    all_seq, adjacency_matrix, all_labels = lib.rna_utils.generate_toy_dataset(80000, 101, pool)

    pos_idx = np.where(all_labels == 1)[0]
    neg_idx = np.where(all_labels == 0)[0]
    all_idx = np.concatenate([pos_idx,
                              neg_idx[np.random.permutation(len(neg_idx))[:2 * len(pos_idx)]]], axis=0)
    size = len(all_idx)
    permute = np.random.permutation(size)

    all_seq = all_seq[all_idx][permute]
    adjacency_matrix = adjacency_matrix[all_idx][permute]
    all_labels = all_labels[all_idx][permute]

    dataset['test_label'] = all_labels[:int(0.1 * size)]
    dataset['test_seq'] = all_seq[:int(0.1 * size)]
    dataset['test_adj_mat'] = adjacency_matrix[:int(0.1 * size)]

    dataset['train_label'] = all_labels[int(0.1 * size):]
    dataset['train_seq'] = all_seq[int(0.1 * size):]
    dataset['train_adj_mat'] = adjacency_matrix[int(0.1 * size):]

    if p is None:
        pool.close()
        pool.join()

    return dataset


if __name__ == "__main__":
    dataset = load_clip_seq(use_embedding=True)
    # print(len(np.where(dataset[0]['train_label'] == 1)[0]))
    # dataset = load_toy_data()
    # print(dataset['train_seq'].shape)
    # print(dataset['train_label'].shape)
