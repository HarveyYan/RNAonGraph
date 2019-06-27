import os
import sys
import gzip
import pickle
import numpy as np
from RNA import fold
import scipy.sparse as sp
from lib.general_utils import Pool
import forgi.graph.bulge_graph as fgb

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def adj_to_bias(adj, nhood=1):
    # [batch_size, nb_nodes, nb_nodes]
    mt = np.stack([np.eye(adj.shape[1])]*adj.shape[0], axis=0)  # self-connection
    for _ in range(nhood):
        mt = np.matmul(mt, (adj + np.stack([np.eye(adj.shape[1])]*adj.shape[0], axis=0)))
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
            pool= Pool(8)
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
