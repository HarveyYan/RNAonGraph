import os
import sys
import numpy as np
from gensim.models import Word2Vec

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


def load_clip_seq(rbp_list=None, p=None, **kwargs):
    '''
    A multiprocessing pool should be provided in case secondary structures and adjacency matrix
    needs to be computed at the data loading stage
    '''
    if p is None:
        pool = Pool(8)
    else:
        pool = p

    fold_algo = kwargs.get('fold_algo', 'rnafold')
    probabilistic = kwargs.get('probabilistic', False)
    load_mat = kwargs.get('load_mat', True)

    clip_data = []

    rbp_list = all_rbps if rbp_list is None else rbp_list
    for rbp in rbp_list:
        dataset = {}

        pos_id, pos_seq = lib.rna_utils.load_seq(path_template.format(rbp, 'train', 'positives'))
        neg_id, neg_seq = lib.rna_utils.load_seq(path_template.format(rbp, 'train', 'negatives'))
        all_id = pos_id + neg_id
        all_seq = pos_seq + neg_seq
        permute = np.random.permutation(len(all_id))

        if load_mat:
            pos_matrix = lib.rna_utils.load_mat(path_template.format(rbp, 'train', 'positives')
                                                , pool, fold_algo, probabilistic)
            neg_matrix = lib.rna_utils.load_mat(path_template.format(rbp, 'train', 'negatives')
                                                , pool, fold_algo, probabilistic)
            if probabilistic:
                pos_adjacency_matrix, pos_probability_matrix = pos_matrix
                neg_adjacency_matrix, neg_probability_matrix = neg_matrix
                probability_matrix = np.concatenate([pos_probability_matrix, neg_probability_matrix], axis=0)
                dataset['train_prob_mat'] = probability_matrix[permute]
            else:
                pos_adjacency_matrix = pos_matrix
                neg_adjacency_matrix = neg_matrix
            adjacency_matrix = np.concatenate([pos_adjacency_matrix, neg_adjacency_matrix], axis=0)
            dataset['train_adj_mat'] = adjacency_matrix[permute]

        dataset['train_label'] = np.array([1] * len(pos_id) + [0] * (len(neg_id)))[permute]

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
            dataset['train_seq'] = np.array([[VOCAB.index(c) for c in seq] for seq in kmers])[permute]
        else:
            VOCAB = ['NOT_FOUND', 'A', 'C', 'G', 'T']
            VOCAB_VEC = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).astype(
                np.float32)
            dataset['train_seq'] = np.array([[VOCAB.index(c) for c in seq] for seq in all_seq])[permute]

        # load test set
        pos_id, pos_seq = lib.rna_utils.load_seq(path_template.format(rbp, 'ls', 'positives'))
        neg_id, neg_seq = lib.rna_utils.load_seq(path_template.format(rbp, 'ls', 'negatives'))
        all_id = pos_id + neg_id
        all_seq = pos_seq + neg_seq

        if load_mat:
            pos_matrix = lib.rna_utils.load_mat(path_template.format(rbp, 'ls', 'positives')
                                                , pool, fold_algo, probabilistic)
            neg_matrix = lib.rna_utils.load_mat(path_template.format(rbp, 'ls', 'negatives')
                                                , pool, fold_algo, probabilistic)
            if probabilistic:
                pos_adjacency_matrix, pos_probability_matrix = pos_matrix
                neg_adjacency_matrix, neg_probability_matrix = neg_matrix
                probability_matrix = np.concatenate([pos_probability_matrix, neg_probability_matrix], axis=0)
                dataset['test_prob_mat'] = probability_matrix
            else:
                pos_adjacency_matrix = pos_matrix
                neg_adjacency_matrix = neg_matrix
            adjacency_matrix = np.concatenate([pos_adjacency_matrix, neg_adjacency_matrix], axis=0)
            dataset['test_adj_mat'] = adjacency_matrix

        dataset['test_label'] = np.array([1] * len(pos_id) + [0] * (len(neg_id)))

        all_seq = [seq.upper().replace('U', 'T') for seq in all_seq]

        if use_embedding:
            kmers = get_kmers(all_seq, kmer_len)
            dataset['test_seq'] = np.array([[VOCAB.index(c) if c in VOCAB else 0 for c in seq] for seq in kmers])
        else:
            dataset['test_seq'] = np.array([[VOCAB.index(c) if c in VOCAB else 0 for c in seq] for seq in all_seq])

        # only using nucleotide information
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


if __name__ == "__main__":
    dataset = load_clip_seq(use_embedding=False, rbp_list=['ALKBH5_Baltz2012'], fold_algo='rnafold', probabilistic=False)[0]
    print(dataset['train_adj_mat'].shape)
    print(dataset['train_adj_mat'][0].shape)
