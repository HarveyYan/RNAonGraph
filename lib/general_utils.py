import os
import sys
import multiprocessing as mp
import multiprocessing.pool as pool
import pandas as pd
import matplotlib.pyplot as plt

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class Pool(pool.Pool):
    Process = NoDaemonProcess


def compare_two_csvs(path_to_csv_1, path_to_csv_2, experiment, axis_name_1, axis_name_2, roundto=3):
    file1 = pd.read_csv(path_to_csv_1)
    file2 = pd.read_csv(path_to_csv_2)

    file2['RBP'] = pd.Categorical(
        file2['RBP'],
        categories=file1['RBP'],
        ordered=True
    )
    file2 = file2.sort_values('RBP')
    import numpy as np
    auc_1 = np.array(file1['auc'][:len(file2['auc'])]).round(roundto)
    auc_2 = np.array(file2['auc']).round(roundto)

    fig = plt.figure(figsize=(5, 5))
    fig.add_subplot(111)
    plt.title(experiment)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(axis_name_1 + '\n' + '%.{0}f\u00b1%.{0}f'.format(roundto) % (auc_1.mean(), auc_1.std()))
    plt.ylabel(axis_name_2 + '\n' + '%.{0}f\u00b1%.{0}f'.format(roundto) % (auc_2.mean(), auc_2.std()))

    idx_pos = np.where(auc_1 > auc_2)[0]
    pos = plt.scatter(auc_1[idx_pos], auc_2[idx_pos], color='b', marker='x')

    idx_neg = np.where(auc_1 < auc_2)[0]
    neg = plt.scatter(auc_1[idx_neg], auc_2[idx_neg], color='r', marker='o')

    idx_neu = np.where(auc_1 == auc_2)[0]
    neu = plt.scatter(auc_1[idx_neu], auc_2[idx_neu], color='w')

    plt.legend([pos, neg, neu], ['%s is better:%d' % (axis_name_1, len(idx_pos)),
                                 '%s is better:%d' % (axis_name_2, len(idx_neg)), 'draw:%d' % (len(idx_neu))],
               scatterpoints=1, loc='lower left')

    if not os.path.exists('../Graph'):
        os.mkdir('../Graph')
    plt.savefig('../Graph/%s.png' % (experiment), dpi=300)


if __name__ == "__main__":
    # compare_two_csvs(
    #     '../output/RNATracker/20190813-000624-smaller-arch-clip-seq-5000-augment-features/rbp-results.csv',
    #     '../output/ideep-clipseq-5000.csv',
    #     'RNATracker-vs-ideep-clipseq-5000', 'RNATracker', 'ideep', roundto=3)

    # compare_two_csvs(
    #     '../output/RGCN/20190807-195408-ggnn-attention-32-20-vanilla-attention/rbp-results.csv',
    #     '../output/ideeps.csv',
    #     '(a)gated-rgcn-vs-ideeps', '(a)gated-rgcn', 'ideeps')

    # compare_two_csvs(
    #     '../output/RGCN/20190807-195731-ggnn-attention-32-20-simplified-lstm/rbp-results.csv',
    #     '../output/RGCN/20190807-195408-ggnn-attention-32-20-vanilla-attention/rbp-results.csv',
    #     '(sa)gated-rgcn-vs-(a)gated-rgcn', '(sa)gated-rgcn', '(a)gated-rgcn')

    # compare_two_csvs(
    #     '../output/RNATracker/20190705-141134-set2set-t10-128/rbp-results.csv',
    #     '../output/RGCN/20190815-211234-32-80-lstm-clipseq-30000-rgcn/rbp-results.csv',
    #     'RNATracker-vs-MFE-rgcn', 'RNATracker', 'MFE-rgcn')

    compare_two_csvs(
        '../output/RNATracker/20190705-141134-set2set-t10-128/rbp-results.csv',
        '../output/RGCN/20190820-171941-boltzmann-sampling-rgcn/rbp-results.csv',
        'conv-set2set-vs-boltzmann-rgcn', 'conv-set2set', 'boltzmann-rgcn')
