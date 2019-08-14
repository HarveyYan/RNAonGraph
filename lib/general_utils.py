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


def compare_two_csvs(path_to_csv_1, path_to_csv_2, experiment, axis_name_1, axis_name_2):
    file1 = pd.read_csv(path_to_csv_1)
    file2 = pd.read_csv(path_to_csv_2)

    file2['RBP'] = pd.Categorical(
        file2['RBP'],
        categories=file1['RBP'],
        ordered=True
    )
    file2 = file2.sort_values('RBP')
    import numpy as np
    print(np.where(np.array(file1['auc'][:len(file2['auc'])]) > np.array(file2['auc']))[0].__len__())
    fig = plt.figure(figsize=(12, 12))
    fig.add_subplot(111)
    plt.title(experiment)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(axis_name_1)
    plt.ylabel(axis_name_2)
    plt.scatter(file1['auc'][:len(file2['auc'])], file2['auc'])
    if not os.path.exists('../Graph'):
        os.mkdir('../Graph')
    plt.savefig('../Graph/%s.png'%(experiment))


if __name__ == "__main__":

    compare_two_csvs('../output/RNATracker/20190705-141134-set2set-t10-128/rbp-results.csv',
                     '../output/RNATracker/20190813-010754-smaller-arch/rbp-results.csv',
                     'smaller-vs-ordinary-rnatracker', 'ordinary-model', 'smaller-model')

    # compare_two_csvs('../output/RGCN/20190807-195408-ggnn-attention-32-20-vanilla-attention/rbp-results.csv',
    #                  '../output/RGCN/20190807-195921-ggnn-attention-32-20-simplified-embedding/rbp-results.csv',
    #                  'attention-vs-simplified-embedding', 'GGNN-attention', 'GGNN-simplified-embedding')
