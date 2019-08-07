import re
import os
import sys
import gzip
import pickle
import numpy as np
from RNA import fold
import scipy.sparse as sp
from functools import partial
import forgi.graph.bulge_graph as fgb

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.general_utils import Pool


def adj_to_bias(adj, nhood=1):
    # [batch_size, nb_nodes, nb_nodes]
    mt = np.stack([np.eye(adj.shape[1])] * adj.shape[0], axis=0)  # self-connection
    for _ in range(nhood):
        mt = np.matmul(mt, (adj + np.stack([np.eye(adj.shape[1])] * adj.shape[0], axis=0)))
    mt = np.greater(mt, 0.).astype(np.float32)
    return -1e9 * (1.0 - mt)


def fold_seq(seq):
    struct = fold(seq)[0]
    matrix = adj_mat(struct)
    return struct, matrix


def adj_mat(struct):
    # create sparse matrix
    row_col, data = [], []
    length = len(struct)
    for i in range(length):
        if i != length - 1:
            row_col.append((i, i + 1))
            data.append(1)
        if i != 0:
            row_col.append((i, i - 1))
            data.append(2)
    bg = fgb.BulgeGraph.from_dotbracket(struct)
    for i, ele in enumerate(struct):
        if ele == '(':
            row_col.append((i, bg.pairing_partner(i + 1) - 1))
            data.append(3)
        elif ele == ')':
            row_col.append((i, bg.pairing_partner(i + 1) - 1))
            data.append(4)
    return sp.csr_matrix((data, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])),
                         shape=(length, length))


def fold_rna_from_file(filepath, p=None):
    if filepath.endswith('.fa'):
        file = open(filepath, 'r')
    else:
        file = gzip.open(filepath)

    all_id = []
    all_seq = []
    seq = ''
    for row in file:
        if type(row) is bytes:
            row = row.decode('utf-8')
        row = row.rstrip()
        if row.startswith('>'):
            all_id.append(row)
            if seq != '':
                all_seq.append(seq)
                seq = ''
        else:
            seq += row
    all_seq.append(seq)

    if os.path.exists(os.path.join(os.path.dirname(filepath), 'adj_mat.obj')):
        sp_adj_matrix = pickle.load(open(os.path.join(os.path.dirname(filepath), 'adj_mat.obj'), 'rb'))
    else:
        print('Parsing', filepath)
        if p is None:
            pool = Pool(8)
        else:
            pool = p

        res = list(pool.imap(fold_seq, all_seq))

        sp_adj_matrix = []
        with open(os.path.join(os.path.dirname(filepath), 'structures.fa'), 'w') as file:
            for id, (struct, matrix) in zip(all_id, res):
                file.writelines('>%s\n%s\n' % (id, struct))
                sp_adj_matrix.append(matrix)

        pickle.dump(sp_adj_matrix, open(os.path.join(os.path.dirname(filepath), 'adj_mat.obj'), 'wb'))

        if p is None:
            pool.close()
            pool.join()

    adjacency_matrix = np.stack([mat.toarray() for mat in sp_adj_matrix], axis=0)

    return all_id, all_seq, adjacency_matrix


def fold_and_check_hairpin(seq, return_label=True):
    regex = r'^\(\(\(\.\.\.\)\)\)[\.\(]|[\.\)]\(\(\(\.\.\.\)\)\)$|[\.\)]\(\(\(\.\.\.\)\)\)|\(\(\(\.\.\.\)\)\)[\.\(]'
    '''return label, or an annotation over the entire seq'''
    '''fold rna and check if the structure contains a hairpin of 3 loose nucleotide connected by a stem of 3 basepairs'''
    struct = fold(seq)[0]
    mat = adj_mat(struct)
    if return_label:
        match = re.search(regex, struct)
        return struct, mat, int(match is not None)
    else:
        annotation = [0] * len(seq)
        for match in re.finditer(regex, struct):
            start_idx = seq[match.start(): match.end()].index('(((...)))') + match.start()
            annotation[start_idx:start_idx + 9] = [1] * 9
        return struct, mat, annotation


def fold_and_check_element(seq, element_symbol, return_label=True):
    '''simply use forgi to annotate the whole string and check if it contains the element'''
    '''return label, or an annotation over the entire seq'''
    '''fold rna and check if the structure contains a hairpin of 3 loose nucleotide connected by a stem of 3 basepairs'''
    struct = fold(seq)[0]
    mat = adj_mat(struct)
    bg = fgb.BulgeGraph.from_dotbracket(struct)
    annotation = bg.to_element_string(())
    if return_label:
        return struct, mat, int(element_symbol in annotation)
    else:
        return struct, mat, [int(element_symbol == c) for c in annotation]


def generate_hairpin_dataset(n, length, p=None, return_label=True):
    '''
    generate toy dataset
    positive examples: RNA sequences that contain specific structural motifs:
        1. a hairpin of three nucleotides connected by a stem of 3 base-pairs
        2. nucleotidal composition does not matter.
    negative examples: RNA sequences that do not contain this specific motifs
    '''
    data_path = os.path.join(basedir, 'Data/toy-data/hairpin/%s' % ('label' if return_label else 'annotation'))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if os.path.exists(os.path.join(data_path, 'seq-and-struct.fa')) and \
            os.path.exists(os.path.join(data_path, 'adj_mat.obj')):
        all_labels = []
        all_seqs = []
        with open(os.path.join(data_path, 'seq-and-struct.fa'), 'r') as file:
            for line in file:
                if line[0] == '>':
                    label = line.rstrip().split(' ')[-1].split(':')[-1]
                    if return_label:
                        all_labels.append(int(label))
                    else:
                        all_labels.append([int(c) for c in label.split(',')])
                elif line[0] in 'ACGT':
                    all_seqs.append(['ACGT'.index(c) for c in line.rstrip()])

        all_seqs = np.array(all_seqs)
        sp_adj_matrix = pickle.load(open(os.path.join(data_path, 'adj_mat.obj'), 'rb'))
    else:
        all_seqs = np.zeros((n, length), dtype=int)
        for j in range(length):
            all_seqs[:, j] = np.random.choice([0, 1, 2, 3], n, p=[0.25, 0.25, 0.25, 0.25])
        seqs_str = [''.join(['ACGT'[c] for c in seq]) for seq in all_seqs]

        if p is None:
            pool = Pool(8)
        else:
            pool = p

        res = list(pool.imap(partial(fold_and_check_hairpin, return_label=return_label), seqs_str))

        sp_adj_matrix = []
        all_labels = []
        with open(os.path.join(data_path, 'seq-and-struct.fa'), 'w') as file:
            for seq, (struct, mat, label) in zip(seqs_str, res):
                file.writelines('> label:%s\n%s\n%s\n' %
                                (label if return_label else ','.join([str(c) for c in label]), seq, struct))
                sp_adj_matrix.append(mat)
                all_labels.append(label)

        pickle.dump(sp_adj_matrix, open(os.path.join(data_path, 'adj_mat.obj'), 'wb'))
        if p is None:
            pool.close()
            pool.join()

    all_labels = np.array(all_labels)
    adjacency_matrix = np.stack([mat.toarray() for mat in sp_adj_matrix], axis=0)
    return all_seqs, adjacency_matrix, all_labels


def generate_element_dataset(n, length, element_symbol, p=None, return_label=True):
    '''
    generate toy dataset
    positive examples: RNA sequences that contain specific structural motifs:
        1. a hairpin of three nucleotides connected by a stem of 3 base-pairs
        2. nucleotidal composition does not matter.
    negative examples: RNA sequences that do not contain this specific motifs
    '''
    assert (len(element_symbol) == 1 and str.isalpha(element_symbol))
    data_path = os.path.join(basedir,
                             'Data/toy-data/%s/%s' % (element_symbol, 'label' if return_label else 'annotation'))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if os.path.exists(os.path.join(data_path, 'seq-and-struct.fa')) and \
            os.path.exists(os.path.join(data_path, 'adj_mat.obj')):
        all_labels = []
        all_seqs = []
        with open(os.path.join(data_path, 'seq-and-struct.fa'), 'r') as file:
            for line in file:
                if line[0] == '>':
                    label = line.rstrip().split(' ')[-1].split(':')[-1]
                    if return_label:
                        all_labels.append(int(label))
                    else:
                        all_labels.append([int(c) for c in label.split(',')])
                elif line[0] in 'ACGT':
                    all_seqs.append(['ACGT'.index(c) for c in line.rstrip()])

        all_seqs = np.array(all_seqs)
        sp_adj_matrix = pickle.load(open(os.path.join(data_path, 'adj_mat.obj'), 'rb'))
    else:
        all_seqs = np.zeros((n, length), dtype=int)
        for j in range(length):
            all_seqs[:, j] = np.random.choice([0, 1, 2, 3], n, p=[0.25, 0.25, 0.25, 0.25])
        seqs_str = [''.join(['ACGT'[c] for c in seq]) for seq in all_seqs]

        if p is None:
            pool = Pool(8)
        else:
            pool = p

        res = list(pool.imap(partial(fold_and_check_element,
                                     element_symbol=element_symbol,
                                     return_label=return_label), seqs_str))

        sp_adj_matrix = []
        all_labels = []
        with open(os.path.join(data_path, 'seq-and-struct.fa'), 'w') as file:
            for seq, (struct, mat, label) in zip(seqs_str, res):
                file.writelines('> label:%s\n%s\n%s\n' %
                                (label if return_label else ','.join([str(c) for c in label]), seq, struct))
                sp_adj_matrix.append(mat)
                all_labels.append(label)

        pickle.dump(sp_adj_matrix, open(os.path.join(data_path, 'adj_mat.obj'), 'wb'))
        if p is None:
            pool.close()
            pool.join()

    all_labels = np.array(all_labels)
    adjacency_matrix = np.stack([mat.toarray() for mat in sp_adj_matrix], axis=0)
    return all_seqs, adjacency_matrix, all_labels


if __name__ == "__main__":
    # annotation for the multiloop elements
    all_seqs, adjacency_matrix, all_labels = generate_element_dataset(80000, 101, 'm', return_label=False)
    print(np.where(all_labels == 1)[0].__len__())
