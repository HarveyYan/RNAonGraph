import os
import sys
import numpy as np
import scipy.sparse as sp
import multiprocessing as mp
from gensim.models import Word2Vec
from sklearn.model_selection import KFold

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import lib.rna_utils
from lib.general_utils import Pool

clip_data_path = os.path.join(basedir, 'Data', 'GraphProt_CLIP_sequences')
all_rbps = [dir for dir in os.listdir(clip_data_path) if os.path.isdir(os.path.join(clip_data_path, dir))]
# rbp, split, label
path_template = os.path.join(basedir, 'Data', 'GraphProt_CLIP_sequences', '{}', '{}', '{}', 'data.fa')

BOND_TYPE = {
    0: 'No Bond',
    1: '5\'UTR to 3\'UTR Covalent Bond',
    2: 'reversed Covalent bond',
    3: 'Hydrogen Bond',
    4: 'reversed hydrogen bond',
}


def _initialize():
    if len(all_rbps) == 0:
        print('reorganizing graphprot dataset')

        def __reorganize_dir():
            all_rbps = set([dir.split('.')[0] for dir in os.listdir(clip_data_path)])
            path_template = os.path.join(basedir, 'Data', 'GraphProt_CLIP_sequences', '{}.{}.{}.fa')
            for rbp in all_rbps:
                for split in ['train', 'ls']:
                    for label in ['positives', 'negatives']:
                        dir_to = os.path.join(os.path.dirname(path_template), rbp, split, label)
                        if not os.path.exists(dir_to):
                            os.makedirs(dir_to)
                        os.rename(path_template.format(rbp, split, label), os.path.join(dir_to, 'data.fa'))

        __reorganize_dir()


# def _merge_sparse_submatrices(data, row_col, segments):
#     '''
#     merge sparse submatrices
#     '''
#     all_tensors = []
#     for i in range(4):
#         all_data, all_row_col = [], []
#         size = 0
#         for _data, _row_col, _segment in zip(data, row_col, segments):
#             all_data.append(_data[i])
#             all_row_col.append(np.array(_row_col[i]) + size)
#             size += _segment
#         all_tensors.append(
#             tf.compat.v1.SparseTensorValue(
#                 np.concatenate(all_row_col),
#                 np.concatenate(all_data),
#                 (size, size)
#             )
#         )
#     # return 4 matrices, one for each relation, max_len and segment_length
#     return all_tensors
#
#
# def dataset_map_function(X, y):
#     all_tensors = _merge_sparse_submatrices(*X)
#     return all_tensors, y


def split_matrix_by_relation(mat):
    '''
    split the sparse adjacency matrix by different relations
    '''
    length = mat.shape[0]
    a, b, c, d = sp.triu(mat, 1), sp.triu(mat, 2), sp.tril(mat, -1), sp.tril(mat, -2)
    rel_mats = [(a - b).tocoo(), (c - d).tocoo(), b, d]
    return [rm.data for rm in rel_mats], \
           [np.stack([rm.row, rm.col], axis=1) for rm in rel_mats], length


def split_matrices_by_relation(sparse_matrices, pool):
    '''
    merge sparse submatrices, and split into 4 matrices by relation
    '''
    ret = np.array(list(pool.imap(split_matrix_by_relation, sparse_matrices)))
    return ret[:, 0], ret[:, 1], ret[:, 2]


def split_matrix_triu(mat):
    '''
    split the sparse adjacency matrix by different relations
    '''
    length = mat.shape[0]
    mat = sp.triu(mat, 2)
    return mat.data, np.stack([mat.row, mat.col], axis=1), length


def split_matrices_triu(sparse_matrices, pool):
    '''
    merge sparse submatrices, and split into 4 matrices by relation
    '''
    from tqdm import tqdm
    ret = np.array(list(tqdm(pool.imap(split_matrix_triu, sparse_matrices))))
    return ret[:, 0], ret[:, 1], ret[:, 2]


def load_clip_seq(rbp_list=None, p=None, **kwargs):
    '''
    A multiprocessing pool should be provided in case secondary structures and adjacency matrix
    needs to be computed at the data loading stage
    '''
    if p is None:
        pool = Pool(min(int(mp.cpu_count() * 2 / 3), 12))
    else:
        pool = p

    fold_algo = kwargs.get('fold_algo', 'rnafold')
    probabilistic = kwargs.get('probabilistic', False)
    load_mat = kwargs.get('load_mat', True)
    force_folding = kwargs.get('force_folding', False)
    nucleotide_label = kwargs.get('nucleotide_label', False)

    clip_data = []

    rbp_list = all_rbps if rbp_list is None else rbp_list
    for rbp in rbp_list:
        dataset = {}

        pos_id, pos_seq = lib.rna_utils.load_seq(path_template.format(rbp, 'train', 'positives'))
        neg_id, neg_seq = lib.rna_utils.load_seq(path_template.format(rbp, 'train', 'negatives'))
        all_id = pos_id + neg_id
        all_seq = pos_seq + neg_seq

        if load_mat:
            # load sparse matrices
            pos_matrix = lib.rna_utils.load_mat(path_template.format(rbp, 'train', 'positives')
                                                , pool, fold_algo, probabilistic, load_dense=False,
                                                force_folding=force_folding)
            neg_matrix = lib.rna_utils.load_mat(path_template.format(rbp, 'train', 'negatives')
                                                , pool, fold_algo, probabilistic, load_dense=False,
                                                force_folding=force_folding)
            if probabilistic:
                # we can do this simply because the secondary structure is not a multigraph
                pos_adjacency_matrix, pos_probability_matrix = pos_matrix
                neg_adjacency_matrix, neg_probability_matrix = neg_matrix
                adjacency_matrix = np.concatenate([pos_probability_matrix, neg_probability_matrix], axis=0)
            else:
                pos_adjacency_matrix = pos_matrix
                neg_adjacency_matrix = neg_matrix
                adjacency_matrix = np.concatenate([pos_adjacency_matrix, neg_adjacency_matrix], axis=0)
                adjacency_matrix = np.array([(rmat > 0).astype(np.float32) for rmat in adjacency_matrix])

            # first step, split them sparse csr matrices by relations
            all_data, all_row_col, segment_size = split_matrices_by_relation(adjacency_matrix, pool)

            dataset['all_data'] = all_data
            dataset['all_row_col'] = all_row_col
            dataset['segment_size'] = segment_size

        if nucleotide_label:
            size_pos = len(pos_id)
            # nucleotide level label
            all_label = []
            for i, seq in enumerate(all_seq):
                if i < size_pos:
                    all_label.append((np.array(list(seq)) <= 'Z').astype(np.int32))
                else:
                    all_label.append(np.array([0]*len(seq)))
            dataset['label'] = np.array(all_label)
        else:
            dataset['label'] = np.array([1] * len(pos_id) + [0] * (len(neg_id)))


        all_seq = [seq.upper().replace('U', 'T') for seq in all_seq]

        use_embedding = kwargs.get('use_embedding', False)
        kmer_len = kwargs.get('kmer_len', 3)
        window = kwargs.get('window', 12)
        emb_size = kwargs.get('emb_size', 50)
        if use_embedding:
            path = os.path.join(basedir, os.path.dirname(path_template).format(rbp, 'train', 'word2vec_%d_%d_%d.obj' % (
                kmer_len, window, emb_size)))
            if not os.path.exists(path):
                pretrain_word2vec(all_seq, kmer_len, window, emb_size, path)
            word2vec_model = Word2Vec.load(path)
            VOCAB = ['NOT_FOUND'] + list(word2vec_model.wv.vocab.keys())
            VOCAB_VEC = np.concatenate([np.zeros((1, emb_size)).astype(np.float32), word2vec_model.wv.vectors], axis=0)
            kmers = get_kmers(all_seq, kmer_len)
            dataset['seq'] = np.array([[VOCAB.index(c) for c in seq] for seq in kmers])
        else:
            VOCAB = ['NOT_FOUND', 'A', 'C', 'G', 'T']
            VOCAB_VEC = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).astype(
                np.float32)
            dataset['seq'] = np.array([[VOCAB.index(c) for c in seq] for seq in all_seq])

        # to ensure segment_size is always included
        if 'segment_size' not in dataset:
            dataset['segment_size'] = np.array([len(seq) for seq in dataset['seq']])

        # only using nucleotide information
        dataset['VOCAB'] = VOCAB
        dataset['VOCAB_VEC'] = VOCAB_VEC

        kf = KFold(n_splits=10)
        splits = kf.split(all_id)
        dataset['splits'] = list(splits)

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


if __name__ == "__main__":
    for i, rbp in enumerate(all_rbps):
        if i < 6:
            continue
        load_clip_seq([rbp], fold_algo='rnaplfold', probabilistic=True)
