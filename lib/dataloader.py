import os
import sys
import gzip
import itertools
import numpy as np
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


def augment_features(path):
    '''
    try to avoid this as much as possible.
    we are mostly interested in an end-to-end learning scenario
    '''
    # region type: 101 x 5
    region_types = np.loadtxt(gzip.open(os.path.join(path, "matrix_RegionType.tab.gz")), skiprows=1)
    assert (region_types.shape[1] == 505)  # 4 region types
    region_types = np.transpose(region_types.reshape((region_types.shape[0], 5, 101)), [0, 2, 1])

    coclip = np.loadtxt(gzip.open(os.path.join(path, "matrix_Cobinding.tab.gz")), skiprows=1)
    assert (coclip.shape[1] % 101 == 0)
    nb_exprs = coclip.shape[1] // 101
    coclip = np.transpose(coclip.reshape((coclip.shape[0], nb_exprs, 101)), [0, 2, 1])

    return np.concatenate([region_types, coclip], axis=-1)


def load_clip_seq(rbp_list=None, p=None, **kwargs):
    '''
    A multiprocessing pool should be provided in case secondary structures and adjacency matrix
    needs to be computed at the data loading stage
    '''
    if p is None:
        pool = Pool(int(os.cpu_count()*2/3))
    else:
        pool = p

    clip_data = []

    fold_algo = kwargs.get('fold_algo', 'rnafold')
    sampling = kwargs.get('sampling', False)

    rbp_list = all_rbps if rbp_list is None else rbp_list
    for rbp in rbp_list:
        dataset = {}

        all_id, all_seq, matrix, all_struct = lib.rna_utils. \
            fold_rna_from_file(path_template.format(rbp, 'training_sample_0', 'sequences.fa.gz'), pool,
                               fold_algo, sampling)

        if sampling:
            adjacency_matrix, probability_matrix = matrix
        else:
            adjacency_matrix = matrix

        permute = np.random.permutation(len(all_id))
        dataset['train_label'] = np.array([int(id.split(' ')[-1].split(':')[-1]) for id in all_id])[permute]
        dataset['train_adj_mat'] = adjacency_matrix[permute]
        if sampling:
            dataset['train_prob_mat'] = probability_matrix[permute]

        if kwargs.get('augment_features', False):
            # additional features: [size, length, features_dim]
            dataset['train_features'] = augment_features(path_template.format(rbp, 'training_sample_0', ''))[permute]

        # dot bracket structure features
        VOCAB_STRUCT = ['.', '(', ')']
        VOCAB_STRUCT_VEC = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32)
        '''todo: won't be this simple anymore! we will have a list of all sampled/selected structures'''
        # dataset['train_struct'] = np.array([[VOCAB_STRUCT.index(c) for c in struct] for struct in all_struct])[permute]

        use_embedding = kwargs.get('use_embedding', False)
        kmer_len = kwargs.get('kmer_len', 3)
        window = kwargs.get('window', 12)
        emb_size = kwargs.get('emb_size', 50)
        if use_embedding:
            path = os.path.join(basedir, path_template.format(rbp, 'training_sample_0',
                                                              'word2vec_%d_%d_%d.obj' % (kmer_len, window, emb_size)))
            if not os.path.exists(path):
                pretrain_word2vec(all_seq, kmer_len, window, emb_size, path)
            word2vec_model = Word2Vec.load(path)
            VOCAB = ['NOT_FOUND'] + list(word2vec_model.wv.vocab.keys())
            VOCAB_VEC = np.concatenate([np.zeros((1, emb_size)).astype(np.float32), word2vec_model.wv.vectors], axis=0)
            kmers = get_kmers(all_seq, kmer_len)
            dataset['train_seq'] = np.array([[VOCAB.index(c) for c in seq] for seq in kmers])[permute]
        else:
            VOCAB = ['NOT_FOUND', 'A', 'C', 'G', 'T', 'N']
            VOCAB_VEC = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 1]]).astype(
                np.float32)
            dataset['train_seq'] = np.array([[VOCAB.index(c) for c in seq] for seq in all_seq])[permute]

        # load test set
        all_id, all_seq, matrix, all_struct = lib.rna_utils. \
            fold_rna_from_file(path_template.format(rbp, 'test_sample_0', 'sequences.fa.gz'), pool,
                               fold_algo, sampling)

        if sampling:
            adjacency_matrix, probability_matrix = matrix
        else:
            adjacency_matrix = matrix

        dataset['test_label'] = np.array([int(id.split(' ')[-1].split(':')[-1]) for id in all_id])
        dataset['test_adj_mat'] = adjacency_matrix
        if sampling:
            dataset['test_prob_mat'] = probability_matrix
        if kwargs.get('augment_features', False):
            dataset['test_features'] = augment_features(path_template.format(rbp, 'test_sample_0', ''))

        # dataset['test_struct'] = np.array([['.()'.index(c) for c in struct] for struct in all_struct])

        if use_embedding:
            kmers = get_kmers(all_seq, kmer_len)
            dataset['test_seq'] = np.array([[VOCAB.index(c) if c in VOCAB else 0 for c in seq] for seq in kmers])
        else:
            dataset['test_seq'] = np.array([[VOCAB.index(c) if c in VOCAB else 0 for c in seq] for seq in all_seq])

        if kwargs.get('merge_seq_and_struct', False):
            mode = kwargs.get('merge_mode', 'product')
            TOTAL_VOCAB = []
            TOTAL_VOCAB_VEC = []
            for se, st in itertools.product(VOCAB[1:], VOCAB_STRUCT):
                TOTAL_VOCAB.append(se + st)
                vec_seq = VOCAB_VEC[VOCAB.index(se)]
                vec_struct = VOCAB_STRUCT_VEC[VOCAB_STRUCT.index(st)]
                if mode == 'concatenation':
                    TOTAL_VOCAB_VEC.append(np.concatenate([vec_seq, vec_struct]))  # dim = 5 + 3
                elif mode == 'product':
                    vec = [0] * (len(VOCAB[1:]) * len(VOCAB_STRUCT))  # dim = 5 * 3
                    vec[(VOCAB.index(se) - 1) * len(VOCAB_STRUCT) + VOCAB_STRUCT.index(st)] = 1
                    TOTAL_VOCAB_VEC.append(vec)
                else:
                    raise ValueError('Unknown merge mode')
            TOTAL_VOCAB_VEC = np.array(TOTAL_VOCAB_VEC).astype(np.float32)

            for prefix in ['train', 'test']:
                all_seq = []
                for seq, struct in zip(dataset['%s_seq' % (prefix)], dataset['%s_struct' % (prefix)]):
                    merged_seq = []
                    for se_idx, st_idx in zip(seq, struct):
                        merged_seq.append(TOTAL_VOCAB.index(VOCAB[se_idx] + VOCAB_STRUCT[st_idx]))
                    all_seq.append(merged_seq)
                dataset['%s_seq' % (prefix)] = np.array(all_seq)

            dataset['VOCAB'] = TOTAL_VOCAB
            dataset['VOCAB_VEC'] = TOTAL_VOCAB_VEC
        else:
            # only using nucleotide information then
            dataset['VOCAB'] = VOCAB
            dataset['VOCAB_VEC'] = VOCAB_VEC
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
        seq = 'N' * left + seq + 'N' * (right - 1)
        for j in range(left, left + length):
            kmers.append(seq[j - left: j + right])
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


def load_toy_data(load_hairpin, return_label, p=None, element_symbol='m'):
    if p is None:
        pool = Pool(8)
    else:
        pool = p

    dataset = {}

    if load_hairpin:
        all_seq, adjacency_matrix, all_labels, all_struct = lib.rna_utils. \
            generate_hairpin_dataset(80000, 101, pool, return_label)
    else:
        all_seq, adjacency_matrix, all_labels, all_struct = lib.rna_utils. \
            generate_element_dataset(80000, 101, element_symbol, pool, return_label)

    if return_label:
        pos_idx = np.where(all_labels == 1)[0]
        neg_idx = np.where(all_labels == 0)[0]
    else:
        pos_idx = np.where(np.count_nonzero(all_labels, axis=-1) > 0)[0]
        neg_idx = np.where(np.count_nonzero(all_labels, axis=-1) == 0)[0]

    all_idx = np.concatenate([pos_idx,
                              neg_idx[np.random.permutation(len(neg_idx))[:2 * len(pos_idx)]]], axis=0)
    size = len(all_idx)
    permute = np.random.permutation(size)

    all_seq = all_seq[all_idx][permute]
    adjacency_matrix = adjacency_matrix[all_idx][permute]
    all_labels = all_labels[all_idx][permute]
    all_struct = all_struct[all_idx][permute]

    dataset['test_label'] = all_labels[:int(0.1 * size)]
    dataset['test_seq'] = all_seq[:int(0.1 * size)]
    dataset['test_adj_mat'] = adjacency_matrix[:int(0.1 * size)]
    dataset['test_struct'] = all_struct[:int(0.1 * size)]

    dataset['train_label'] = all_labels[int(0.1 * size):]
    dataset['train_seq'] = all_seq[int(0.1 * size):]
    dataset['train_adj_mat'] = adjacency_matrix[int(0.1 * size):]
    dataset['train_struct'] = all_struct[int(0.1 * size):]

    dataset['VOCAB'] = 'ACGT'
    dataset['VOCAB_VEC'] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).astype(np.float32)

    if p is None:
        pool.close()
        pool.join()

    return dataset


if __name__ == "__main__":
    dataset = load_clip_seq(use_embedding=False, sampling=False)
    # print(len(np.where(dataset[0]['train_label'] == 1)[0]))
    # dataset = load_toy_data()
    # print(dataset['train_seq'].shape)
    # print(dataset['train_label'].shape)
