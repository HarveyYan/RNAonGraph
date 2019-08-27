import re
import os
import sys
import RNA
import gzip
import pickle
import random
import shutil
import subprocess
import numpy as np
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


'''
equilibrium probability using RNAplfold
'''


def fold_seq_rnaplfold(seq, w, l, cutoff, no_lonely_bps):
    pwd = os.getcwd()
    np.random.seed(random.seed())
    name = str(np.random.rand())
    if 'SLURM_TMPDIR' in os.environ:
        name = os.path.join(os.environ['SLURM_TMPDIR'], name)
    os.makedirs(name)
    os.chdir(name)

    # Call RNAplfold on command line.
    no_lonely_bps_str = ""
    if no_lonely_bps:
        no_lonely_bps_str = "--noLP"
    cmd = 'echo %s | RNAplfold -W %d -L %d -c %.4f %s' % (seq, w, l, cutoff, no_lonely_bps_str)
    ret = subprocess.call(cmd, shell=True)

    # assemble adjacency matrix
    row_col, link, prob = [], [], []
    length = len(seq)
    for i in range(length):
        if i != length - 1:
            row_col.append((i, i + 1))
            link.append(1)
            prob.append(1.)
        if i != 0:
            row_col.append((i, i - 1))
            link.append(2)
            prob.append(1.)
    # Extract base pair information.
    start_flag = False
    with open('plfold_dp.ps') as f:
        for line in f:
            if start_flag:
                values = line.split()
                if len(values) == 4:
                    source_id = int(values[0]) - 1
                    dest_id = int(values[1]) - 1
                    avg_prob = float(values[2])
                    # source_id < dest_id
                    row_col.append((source_id, dest_id))
                    link.append(3)
                    prob.append(avg_prob)
                    row_col.append((dest_id, source_id))
                    link.append(4)
                    prob.append(avg_prob)
            if 'start of base pair probability data' in line:
                start_flag = True
    # delete RNAplfold output file.
    os.remove('plfold_dp.ps')
    os.chdir(pwd)
    shutil.rmtree(name)
    # placeholder for dot-bracket structure
    return (sp.csr_matrix((link, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),
            sp.csr_matrix((prob, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),)


'''
Boltzmann sampling using RNAsubopt
'''


def fold_seq_subopt(seq, probabilistic=False, sampling_amount=1000):
    # RNAfold is only suitable for short RNA sequences within 100 nucleotides
    if probabilistic:
        # sampling from a boltzmann ensemble
        cmd = 'echo "%s" | RNAsubopt --stochBT=%d' % (seq, sampling_amount)
        struct_list = subprocess.check_output(cmd, shell=True). \
                          decode('utf-8').rstrip().split('\n')[1:]
    else:
        struct_list, energy_list = [], []

        def collect_subopt_result(structure, energy, *args):
            if not structure == None:
                struct_list.append(structure)
                energy_list.append(energy)

        # Enumerate all structures 100 dacal/mol = 1 kcal/mol around
        # default deltaEnergy is the MFE
        RNA.fold_compound(seq).subopt_cb(100, collect_subopt_result, None)

        # sort
        struct_list = list(np.array(struct_list)[np.argsort(energy_list)])

    # merging all structures into a single adjacency matrix
    # probability returning two matrices
    matrix = adj_mat_subopt(struct_list, probabilistic)
    # process the structures
    return structural_content(struct_list), matrix


def structural_content(struct_list):
    size = len(struct_list)
    length = len(struct_list[0])
    content = np.zeros((length, 3), dtype=np.int32)
    for i in range(length):
        for j in range(size):
            idx = '.()'.index(struct_list[j][i])
            content[i][idx] += 1
    return content.astype(np.float32) / size


def adj_mat_subopt(struct_list, probabilistic):
    # create sparse matrix
    row_col, data = [], []
    length = len(struct_list[0])
    counts = []
    for i in range(length):
        if i != length - 1:
            row_col.append((i, i + 1))
            data.append(1)
            counts.append(0)
        if i != 0:
            row_col.append((i, i - 1))
            data.append(2)
            counts.append(0)
    if probabilistic:
        for struct in struct_list:
            bg = fgb.BulgeGraph.from_dotbracket(struct)
            for i, ele in enumerate(struct):
                if ele == '(':
                    if not (i, bg.pairing_partner(i + 1) - 1) in row_col:
                        row_col.append((i, bg.pairing_partner(i + 1) - 1))
                        data.append(3)
                        counts.append(1)
                    else:
                        idx = row_col.index((i, bg.pairing_partner(i + 1) - 1))
                        counts[idx] += 1
                elif ele == ')':
                    if not (i, bg.pairing_partner(i + 1) - 1) in row_col:
                        row_col.append((i, bg.pairing_partner(i + 1) - 1))
                        data.append(4)
                        counts.append(1)
                    else:
                        idx = row_col.index((i, bg.pairing_partner(i + 1) - 1))
                        counts[idx] += 1
        # normalize each row into probabilities
        for i in range(len(row_col)):
            if counts[i] > 0:
                # have to be a hydrogen bond
                counts[i] /= len(struct_list)
            else:
                # covalent bond that forms the stem
                counts[i] = 1.
        return (sp.csr_matrix((data, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),
                sp.csr_matrix((counts, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),)
    else:
        for struct in struct_list:
            bg = fgb.BulgeGraph.from_dotbracket(struct)
            for i, ele in enumerate(struct):
                if ele == '(':
                    if not (i, bg.pairing_partner(i + 1) - 1) in row_col:
                        row_col.append((i, bg.pairing_partner(i + 1) - 1))
                        data.append(3)
                elif ele == ')':
                    if not (i, bg.pairing_partner(i + 1) - 1) in row_col:
                        row_col.append((i, bg.pairing_partner(i + 1) - 1))
                        data.append(4)
        return sp.csr_matrix((data, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])),
                             shape=(length, length))


'''
MFE structure using RNAfold
'''


def fold_seq_rnafold(seq):
    '''fold sequence using RNAfold'''
    struct = RNA.fold(seq)[0]
    matrix = adj_mat(struct)
    return structural_content([struct]), matrix


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


def load_fasta_format(file):
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
    return all_id, all_seq


def load_dotbracket(filepath, pool=None, fold_algo='rnafold', probabilistic=False, **kwargs):
    prefix = '%s_%s_' % (fold_algo, probabilistic)
    full_path = os.path.join(os.path.dirname(filepath), '{}structures.npy'.format(prefix))
    if not os.path.exists(full_path):
        print(full_path, 'is missing. Begin folding from scratch.')
        fold_rna_from_file(filepath, pool, fold_algo, probabilistic, **kwargs)
    # load secondary structures
    all_struct = np.load(
        os.path.join(os.path.dirname(filepath), '{}structures.npy'.format(prefix)))
    return all_struct


def load_mat(filepath, pool=None, fold_algo='rnafold', probabilistic=False, **kwargs):
    prefix = '%s_%s_' % (fold_algo, probabilistic)
    if not os.path.exists(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix))) or probabilistic and \
            not os.path.exists(os.path.join(os.path.dirname(filepath), '{}prob_mat.obj'.format(prefix))):
        print('adj mat or prob mat is missing. Begin folding from scratch.')
        fold_rna_from_file(filepath, pool, fold_algo, probabilistic, **kwargs)

    load_dense = kwargs.get('load_dense', True)
    sp_rel_matrix = pickle.load(open(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix)), 'rb'))
    if load_dense:
        adjacency_matrix = np.array([mat.toarray() for mat in sp_rel_matrix])
    else:
        adjacency_matrix = np.array(sp_rel_matrix)

    if probabilistic:
        sp_prob_matrix = pickle.load(
            open(os.path.join(os.path.dirname(filepath), '{}prob_mat.obj'.format(prefix)), 'rb'))
        if load_dense:
            probability_matrix = np.array([mat.toarray() for mat in sp_prob_matrix])
        else:
            probability_matrix = np.array(sp_prob_matrix)
        matrix = (adjacency_matrix, probability_matrix)
    else:
        matrix = adjacency_matrix

    return matrix


def load_seq(filepath):
    if filepath.endswith('.fa'):
        file = open(filepath, 'r')
    else:
        file = gzip.open(filepath, 'rb')

    all_id, all_seq = load_fasta_format(file)
    return all_id, all_seq


def fold_rna_from_file(filepath, p=None, fold_algo='rnafold', probabilistic=False, **kwargs):
    assert (fold_algo in ['rnafold', 'rnasubopt', 'rnaplfold'])
    if fold_algo == 'rnafold':
        assert (probabilistic is False)
    if fold_algo == 'rnaplfold':
        assert (probabilistic is True)
    print('Parsing', filepath)
    _, all_seq = load_seq(filepath)

    # compatible with already computed structures with RNAfold
    prefix = '%s_%s_' % (fold_algo, probabilistic)

    if p is None:
        pool = Pool(int(os.cpu_count() * 2 / 3))
    else:
        pool = p

    if fold_algo == 'rnafold':
        fold_func = fold_seq_rnafold
        from tqdm import tqdm
        res = list(tqdm(pool.imap(fold_func, all_seq)))
        sp_rel_matrix = []
        structural_content = []
        for struct, matrix in res:
            structural_content.append(struct)
            sp_rel_matrix.append(matrix)
        np.save(os.path.join(os.path.dirname(filepath), '{}structures.npy'.format(prefix)),
                np.array(structural_content))  # [size, length, 3]
        pickle.dump(sp_rel_matrix,
                    open(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix)), 'wb'))
    elif fold_algo == 'rnasubopt':
        fold_func = partial(fold_seq_subopt, fold_algo=fold_algo, probabilistic=probabilistic)
        res = list(pool.imap(fold_func, all_seq))
        sp_rel_matrix = []
        sp_prob_matrix = []
        structural_content = []
        for struct, matrix in res:
            structural_content.append(struct)
            if probabilistic:
                rel_mat, prob_mat = matrix
                sp_prob_matrix.append(prob_mat)
            else:
                rel_mat = matrix
            sp_rel_matrix.append(rel_mat)
        np.save(os.path.join(os.path.dirname(filepath), '{}structures.npy'.format(prefix)),
                np.array(structural_content))  # [size, length, 3]
        pickle.dump(sp_rel_matrix,
                    open(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix)), 'wb'))
        if probabilistic:
            pickle.dump(sp_prob_matrix,
                        open(os.path.join(os.path.dirname(filepath), '{}prob_mat.obj'.format(prefix)), 'wb'))
    elif fold_algo == 'rnaplfold':
        winsize = kwargs.get('w', 70)
        fold_func = partial(fold_seq_rnaplfold, w=winsize, l=winsize, cutoff=1e-4, no_lonely_bps=True)
        from tqdm import tqdm
        res = list(tqdm(pool.imap(fold_func, all_seq)))
        sp_rel_matrix = []
        sp_prob_matrix = []
        for rel_mat, prob_mat in res:
            sp_rel_matrix.append(rel_mat)
            sp_prob_matrix.append(prob_mat)
        pickle.dump(sp_rel_matrix,
                    open(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix)), 'wb'))
        pickle.dump(sp_prob_matrix,
                    open(os.path.join(os.path.dirname(filepath), '{}prob_mat.obj'.format(prefix)), 'wb'))
    else:
        raise ValueError('Supported folding algorithms are ' + ', '.join(['rnafold', 'rnasubopt', 'rnaplfold']))

    if p is None:
        pool.close()
        pool.join()

    print('Parsing', filepath, 'finished')


def fold_and_check_hairpin(seq, return_label=True):
    regex = r'^\(\(\(\.\.\.\)\)\)[\.\(]|[\.\)]\(\(\(\.\.\.\)\)\)$|[\.\)]\(\(\(\.\.\.\)\)\)|\(\(\(\.\.\.\)\)\)[\.\(]'
    '''return label, or an annotation over the entire seq'''
    '''fold rna and check if the structure contains a hairpin of 3 loose nucleotide connected by a stem of 3 basepairs'''
    struct = RNA.fold(seq)[0]
    mat = adj_mat(struct)
    if return_label:
        match = re.search(regex, struct)
        return struct, mat, int(match is not None)
    else:
        annotation = [0] * len(seq)
        for match in re.finditer(regex, struct):
            start_idx = struct[match.start(): match.end()].index('(((...)))') + match.start()
            annotation[start_idx:start_idx + 9] = [1] * 9
        return struct, mat, annotation


def fold_and_check_element(seq, element_symbol, return_label=True):
    '''simply use forgi to annotate the whole string and check if it contains the element'''
    '''return label, or an annotation over the entire seq'''
    '''fold rna and check if the structure contains a hairpin of 3 loose nucleotide connected by a stem of 3 basepairs'''
    struct = RNA.fold(seq)[0]
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
        all_struct = []
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
                elif line[0] in '.()':
                    all_struct.append(['.()'.index(c) for c in line.rstrip()])

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
        all_struct = []
        with open(os.path.join(data_path, 'seq-and-struct.fa'), 'w') as file:
            for seq, (struct, mat, label) in zip(seqs_str, res):
                file.writelines('> label:%s\n%s\n%s\n' %
                                (label if return_label else ','.join([str(c) for c in label]), seq, struct))
                sp_adj_matrix.append(mat)
                all_labels.append(label)
                all_struct.append(['.()'.index(c) for c in struct])

        pickle.dump(sp_adj_matrix, open(os.path.join(data_path, 'adj_mat.obj'), 'wb'))
        if p is None:
            pool.close()
            pool.join()

    all_labels = np.array(all_labels)
    adjacency_matrix = np.stack([mat.toarray() for mat in sp_adj_matrix], axis=0)
    all_struct = np.array(all_struct)
    return all_seqs, adjacency_matrix, all_labels, all_struct


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
        all_struct = []
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
                elif line[0] in '.()':
                    all_struct.append(['.()'.index(c) for c in line.rstrip()])

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
        all_struct = []
        with open(os.path.join(data_path, 'seq-and-struct.fa'), 'w') as file:
            for seq, (struct, mat, label) in zip(seqs_str, res):
                file.writelines('> label:%s\n%s\n%s\n' %
                                (label if return_label else ','.join([str(c) for c in label]), seq, struct))
                sp_adj_matrix.append(mat)
                all_labels.append(label)
                all_struct.append(['.()'.index(c) for c in struct])

        pickle.dump(sp_adj_matrix, open(os.path.join(data_path, 'adj_mat.obj'), 'wb'))
        if p is None:
            pool.close()
            pool.join()

    all_labels = np.array(all_labels)
    adjacency_matrix = np.stack([mat.toarray() for mat in sp_adj_matrix], axis=0)
    all_struct = np.array(all_struct)
    return all_seqs, adjacency_matrix, all_labels, all_struct


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=100000, )

    res = fold_seq_rnaplfold(
        'TGTGAAGCGCGGCTAGCTGCCGGGGTTCGAGGTGGGTCCCAGGGTTAAAATCCCTTGTTGTCTTACTGGTGGCAGCAAGCTAGGACTATACTCCTCGGTCG',
        101, 101, 0.0001, True)
    print(res[0].todense())
    print(res[1].todense())

    # with open('boltzmann-sampling-acc.txt', 'w') as file:
    #     for amount in [5, 10, 100, 1000, 5000, 10000]:
    #         rel_diff, prob_diff = [], []
    #         for replcate in range(100):
    #             _, res = fold_seq_subopt(
    #                 'TGTGAAGCGCGGCTAGCTGCCGGGGTTCGAGGTGGGTCCCAGGGTTAAAATCCCTTGTTGTCTTACTGGTGGCAGCAAGCTAGGACTATACTCCTCGGTCG',
    #                 'rnafold', True, amount)
    #             _, new_res = fold_seq_subopt(
    #                 'TGTGAAGCGCGGCTAGCTGCCGGGGTTCGAGGTGGGTCCCAGGGTTAAAATCCCTTGTTGTCTTACTGGTGGCAGCAAGCTAGGACTATACTCCTCGGTCG',
    #                 'rnafold', True, amount)
    #
    #             diff = (res[0].todense() != new_res[0].todense()).astype(np.int32)
    #             rel_diff.append(np.sum(diff))
    #
    #             diff = np.abs(res[1].todense() - new_res[1].todense())
    #             prob_diff.append(np.mean(np.max(diff, axis=-1)))
    #         file.writelines('sampling amount %d, relation difference: %.4f\u00b1%.4f, probability difference: %.4f\u00b1%.4f' %
    #               (amount, np.mean(rel_diff), np.std(rel_diff), np.mean(prob_diff), np.std(prob_diff)))
    #         print('sampling amount %d, relation difference: %.4f\u00b1%.4f, probability difference: %.4f\u00b1%.4f' %
    #               (amount, np.mean(rel_diff), np.std(rel_diff), np.mean(prob_diff), np.std(prob_diff)))

    # annotation for the multiloop elements
    # all_seqs, adjacency_matrix, all_labels, _ = generate_element_dataset(80000, 101, 'i', return_label=False)
    # print(all_labels.shape)
    # print(np.where(np.count_nonzero(all_labels, axis=-1) > 0)[0].__len__())
    #
    # all_seqs, adjacency_matrix, all_labels, _ = generate_hairpin_dataset(80000, 101, 'm', return_label=False)
    # print(all_labels.shape)
    # print(np.where(np.count_nonzero(all_labels, axis=-1) > 0)[0].__len__())
