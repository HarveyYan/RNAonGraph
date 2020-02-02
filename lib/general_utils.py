import os
import sys
import multiprocessing as mp
import multiprocessing.pool as pool
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

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


def compare_two_csvs(path_to_csv_1, path_to_csv_2, experiment, axis_name_1, axis_name_2,
                     roundto=3, entry_name='auc'):
    file1 = pd.read_csv(path_to_csv_1)
    file2 = pd.read_csv(path_to_csv_2)

    file2['RBP'] = pd.Categorical(
        file2['RBP'],
        categories=file1['RBP'],
        ordered=True
    )
    file2 = file2.sort_values('RBP')
    auc_1 = np.array(file1[entry_name][:len(file2[entry_name])]).astype(np.float32).round(roundto)
    auc_2 = np.array(file2[entry_name]).astype(np.float32).round(roundto)

    font = {'fontname': 'Times New Roman', 'size': 14}
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    plt.title(experiment, **font)
    plt.plot([0, 1], [0, 1])
    plt.plot([0, 1], [0.01, 1.01], 'k--')
    plt.plot([0, 1], [-0.01, 0.99], 'k--')
    xlim = 0.6 if 'auc' in entry_name else 0.
    plt.xlim([xlim, 1.0])
    plt.ylim([xlim, 1.05])
    # plt.xlabel(axis_name_1 + '\n' + '%.{0}f\u00b1%.{0}f'.format(roundto) % (auc_1.mean(), auc_1.std()), **font)
    # plt.ylabel(axis_name_2 + '\n' + '%.{0}f\u00b1%.{0}f'.format(roundto) % (auc_2.mean(), auc_2.std()), **font)
    plt.xlabel(axis_name_1, **font)
    plt.ylabel(axis_name_2, **font)

    idx_pos = np.where(auc_1 > auc_2)[0]
    pos = plt.scatter(auc_1[idx_pos], auc_2[idx_pos], color='b', marker='x')

    idx_neg = np.where(auc_1 < auc_2)[0]
    neg = plt.scatter(auc_1[idx_neg], auc_2[idx_neg], color='r', marker='o')
    print(np.argmax(idx_neg))

    idx_neu = np.where(auc_1 == auc_2)[0]
    neu = plt.scatter(auc_1[idx_neu], auc_2[idx_neu], color='w')

    legend = plt.legend([pos, neg, neu], ['%s is better:%d' % (axis_name_1, len(idx_pos)),
                                 '%s is better:%d' % (axis_name_2, len(idx_neg)), 'draw:%d' % (len(idx_neu))],
               scatterpoints=1, loc='lower right')
    plt.setp(legend.texts, **font)
    # for i, (val_1, val_2) in enumerate(zip(auc_1, auc_2)):
    #     if val_2 > val_1 and val_2 - val_1 > 0.1 * val_1:
    #         ax.annotate(file2['RBP'][i], (val_1, val_2))

    plt.tight_layout()
    if not os.path.exists('../Graph'):
        os.mkdir('../Graph')
    plt.savefig('../Graph/%s.png' % (experiment), dpi=300)


def wilcoxon_test(path_to_csv_1, path_to_csv_2, roundto=3, entry_name='auc'):
    file1 = pd.read_csv(path_to_csv_1)
    file2 = pd.read_csv(path_to_csv_2)

    file2['RBP'] = pd.Categorical(
        file2['RBP'],
        categories=file1['RBP'],
        ordered=True
    )
    file2 = file2.sort_values('RBP')

    auc_1 = np.array(file1[entry_name][:len(file2[entry_name])]).round(roundto)
    auc_2 = np.array(file2[entry_name]).round(roundto)
    return wilcoxon(auc_2, auc_1, alternative='greater')


if __name__ == "__main__":

    # compare_two_csvs(
    #     '../output/Joint-MRT-Graphprot-debiased/rnaplfold-results.csv',
    #     '../output/Joint-ada-sampling-debiased/rnaplfold-results.csv',
    #     'RPI-Net(GNN) vs RPI-Net(CNN)', 'RPI-Net(CNN)', 'RPI-Net(GNN)', entry_name='original_auc')
    # print(wilcoxon_test('../output/Joint-MRT-Graphprot-debiased/rnaplfold-results.csv',
    #                     '../output/Joint-ada-sampling-debiased/rnaplfold-results.csv', entry_name='original_auc'))
    #
    # compare_two_csvs(
    #     '../output/graphprot/graphprot_results.csv',
    #     '../output/Joint-ada-sampling-debiased/rnaplfold-results.csv',
    #     'RPI-Net(GNN) vs GraphProt', 'GraphProt', 'RPI-Net(GNN)', entry_name='original_auc')
    # print(wilcoxon_test('../output/graphprot/graphprot_results.csv',
    #                     '../output/Joint-ada-sampling-debiased/rnaplfold-results.csv', entry_name='original_auc'))
    #
    compare_two_csvs(
        '../output/ideepe/ideepe.csv',
        '../output/Joint-ada-sampling-debiased/rnaplfold-results.csv',
        'RPI-Net(GNN) vs iDeepE', 'iDeepE', 'RPI-Net(GNN)', entry_name='original_auc')
    print(wilcoxon_test('../output/ideepe/ideepe.csv',
                  '../output/Joint-ada-sampling-debiased/rnaplfold-results.csv', entry_name='original_auc'))

    # compare_two_csvs(
    #     '../output/mdbn-results.csv',
    #     '../output/Joint-ada-sampling-debiased/rnaplfold-results.csv',
    #     'RPI-Net(GNN) vs mDBN+', 'mDBN+', 'RPI-Net(GNN)', entry_name='original_auc')
    # print(wilcoxon_test('../output/mdbn-results.csv',
    #                     '../output/Joint-ada-sampling-debiased/rnaplfold-results.csv', entry_name='original_auc'))
