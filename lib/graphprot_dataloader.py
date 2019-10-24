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

    probabilistic = kwargs.get('probabilistic', False)
    load_mat = kwargs.get('load_mat', True)
    nucleotide_label = kwargs.get('nucleotide_label', False)

    clip_data = []

    rbp_list = all_rbps if rbp_list is None else rbp_list
    for rbp in rbp_list:
        # rbp, split, label
        path_template = os.path.join(basedir, 'Data', 'GraphProt_CLIP_sequences', '{}', '{}', '{}', 'data.fa')
        dataset = {}

        pos_id, pos_seq = lib.rna_utils.load_seq(path_template.format(rbp, 'train', 'positives'))
        neg_id, neg_seq = lib.rna_utils.load_seq(path_template.format(rbp, 'train', 'negatives'))
        all_id = pos_id + neg_id
        all_seq = pos_seq + neg_seq

        if nucleotide_label:
            size_pos = len(pos_id)
            # nucleotide level label
            all_label = []
            for i, seq in enumerate(all_seq):
                if i < size_pos:
                    all_label.append((np.array(list(seq)) <= 'Z').astype(np.int32))
                else:
                    all_label.append(np.array([0] * len(seq)))
            dataset['label'] = np.array(all_label)
        else:
            dataset['label'] = np.array([1] * len(pos_id) + [0] * (len(neg_id)))

        if kwargs.get('modify_leaks', False):
            path_template = path_template[:-7] + 'modified_data.fa'
            if not os.path.exists(path_template.format(rbp, 'train', 'positives')) or \
                    not os.path.exists(path_template.format(rbp, 'train', 'negatives')):
                all_modified_seq = []
                for seq, label in zip(all_seq, dataset['label']):
                    seq = list(seq)
                    if np.max(label) == 1:
                        pos_idx = np.where(np.array(label) == 1)[0]
                    else:
                        pos_idx = np.where((np.array(seq) <= 'Z').astype(np.int32) == 1)[0]

                    if rbp == 'PARCLIP_IGF2BP123':
                        indices = [pos_idx[-1], pos_idx[0], pos_idx[0] - 1, pos_idx[0] - 2]
                    elif rbp == 'CAPRIN1_Baltz2012':
                        indices = [pos_idx[-1], pos_idx[-1] - 1, pos_idx[0], pos_idx[0] - 1, pos_idx[0] - 2]
                    elif rbp == 'PARCLIP_PUM2':
                        # unable to obtain perfect mask
                        indices = [pos_idx[-1], pos_idx[0] - 1]
                    elif rbp == 'PARCLIP_AGO1234':
                        # unable to obtain perfect mask
                        indices = [pos_idx[-1], pos_idx[0] - 1]
                    elif rbp == 'PARCLIP_MOV10_Sievers':
                        indices = [pos_idx[-1], pos_idx[-1] - 1, pos_idx[0] - 1]
                    elif rbp == 'ZC3H7B_Baltz2012':
                        indices = [pos_idx[-1], pos_idx[-1] - 1, pos_idx[0] + 1, pos_idx[0], pos_idx[0] - 1]
                    else:
                        raise ValueError('TODO: modify_leaks option has not been made available for %s' % (rbp))

                    for idx in indices:
                        try:
                            seq[idx] = np.random.choice(['A', 'C', 'G', 'T'])
                        except IndexError:
                            pass
                    all_modified_seq.append(''.join(seq))

                all_seq = all_modified_seq
                pos_seq = all_seq[:len(pos_id)]
                neg_seq = all_seq[len(pos_id):]

                # cache temporary modified files
                with open(path_template.format(rbp, 'train', 'positives'), 'w') as file:
                    for id, seq in zip(pos_id, pos_seq):
                        file.write('%s\n%s\n' % (id, seq))
                with open(path_template.format(rbp, 'train', 'negatives'), 'w') as file:
                    for id, seq in zip(neg_id, neg_seq):
                        file.write('%s\n%s\n' % (id, seq))
                print('modified sequences have been cached')
            else:
                _, pos_seq = lib.rna_utils.load_seq(path_template.format(rbp, 'train', 'positives'))
                _, neg_seq = lib.rna_utils.load_seq(path_template.format(rbp, 'train', 'negatives'))
                all_seq = pos_seq + neg_seq

        # in nucleotide format
        dataset['raw_seq'] = np.array(all_seq)

        if load_mat:
            # load sparse matrices
            pos_matrix = lib.rna_utils.load_mat(path_template.format(rbp, 'train', 'positives')
                                                , pool, load_dense=False, **kwargs)
            neg_matrix = lib.rna_utils.load_mat(path_template.format(rbp, 'train', 'negatives')
                                                , pool, load_dense=False, **kwargs)
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

        # to ensure segment_size is always included
        if 'segment_size' not in dataset:
            dataset['segment_size'] = np.array([len(seq) for seq in all_seq])

        if kwargs.get('truncate_test', False):
            raise ValueError('truncate_test option is no longer supported at the data loading phase!')

        all_seq = [seq.upper().replace('U', 'T') for seq in all_seq]

        use_embedding = kwargs.get('use_embedding', False)
        kmer_len = kwargs.get('kmer_len', 6)
        window = kwargs.get('window', 5)
        emb_size = kwargs.get('emb_size', 25)
        if use_embedding:
            path = os.path.join(basedir, 'Data/misc/{}'.format('word2vec_%d_%d_%d.obj' % (
                kmer_len, window, emb_size)))
            if not os.path.exists(path):
                tmp_seqs = lib.rna_utils.load_seq(os.path.join(basedir, 'Data/misc/utrs.fa'))[1]
                pretrain_word2vec(tmp_seqs, kmer_len, window, emb_size, path)
            word2vec_model = Word2Vec.load(path)
            VOCAB = ['NOT_FOUND'] + list(word2vec_model.wv.vocab.keys())
            VOCAB_VEC = np.concatenate([np.zeros((1, emb_size)).astype(np.float32), word2vec_model.wv.vectors], axis=0)
            kmers = get_kmers(all_seq, kmer_len)
            dataset['seq'] = np.array([[VOCAB.index(c) if c in VOCAB else 0 for c in seq] for seq in kmers])
        else:
            VOCAB = ['NOT_FOUND', 'A', 'C', 'G', 'T']
            VOCAB_VEC = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).astype(
                np.float32)
            dataset['seq'] = np.array([[VOCAB.index(c) for c in seq] for seq in all_seq])

        if kwargs.get('permute', False):
            permute = np.random.permutation(len(all_id))
            dataset['label'] = dataset['label'][permute]
            dataset['seq'] = dataset['seq'][permute]
            dataset['segment_size'] = dataset['segment_size'][permute]
            if 'all_data' in dataset:
                dataset['all_data'] = dataset['all_data'][permute]
                dataset['all_row_col'] = dataset['all_row_col'][permute]

        # only using nucleotide information
        dataset['VOCAB'] = VOCAB
        dataset['VOCAB_VEC'] = VOCAB_VEC
        dataset['id'] = np.array(all_id)

        kf = KFold(n_splits=10, shuffle=True)
        splits = kf.split(all_id)
        dataset['splits'] = list(splits)

        clip_data.append(dataset)

    if p is None:
        pool.close()
        pool.join()

    return clip_data


def get_kmers(seqs, kmer_len):
    sentence = []
    for seq in seqs:
        kmers = []
        seq = 'N' * (kmer_len - 1) + seq
        for i in range(len(seq) - kmer_len + 1):
            kmers.append(seq[i:i + kmer_len])
        sentence.append(kmers)
    return sentence


def pretrain_word2vec(seqs, kmer_len, window, embedding_size, save_path):
    # use skip grams and negative sampling of 5
    print('word2vec pretaining')
    sentence = get_kmers(seqs, kmer_len)
    model = Word2Vec(sentence, window=window, size=embedding_size,
                     min_count=5, workers=15, sg=1, iter=5, batch_words=1000)
    # to capture as much dependency as possible
    model.train(sentence, total_examples=len(sentence), epochs=model.iter)
    model.save(save_path)


def test_overlapping(id_list_1, id_list_2):
    identity_list_1, identity_list_2 = [], []
    for _id in id_list_1:
        identity_list_1.append(_id.rstrip().split(';')[-1].split(','))
    for _id in id_list_2:
        identity_list_2.append(_id.rstrip().split(';')[-1].split(','))

    overlapping = 0
    for identity_1 in identity_list_1:
        for identity_2 in identity_list_2:
            if identity_1[0] == identity_2[0] and \
                    identity_1[-1] == identity_2[-1] and \
                    not (identity_1[2] < identity_2[1] or identity_2[2] < identity_1[1]):
                overlapping += 1

    print('id_list_1 in id_list_2: %d/%d' % (overlapping, len(identity_list_2)))

    overlapping = 0
    for identity_2 in identity_list_2:
        for identity_1 in identity_list_1:
            if identity_2[0] == identity_1[0] and \
                    identity_2[-1] == identity_2[-1] and \
                    not (identity_1[2] < identity_2[1] or identity_2[2] < identity_1[1]):
                overlapping += 1

    print('id_list_2 in id_list_1: %d/%d' % (overlapping, len(identity_list_1)))


if __name__ == "__main__":
    load_clip_seq(['CAPRIN1_Baltz2012'], use_embedding=False,
                  fold_algo='rnaplfold',
                  probabilistic=True, w=150,
                  nucleotide_label=True, modify_leaks=True)[0]  # load one at a time
