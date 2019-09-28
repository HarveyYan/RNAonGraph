import os
import numpy as np
import pandas as pd
from lib.logger import CSVLogger
import argparse

def summarize_10fold_results(path, outfile_name):
    logger = CSVLogger(outfile_name, path, ['RBP', 'acc', 'acc_std', 'auc', 'auc_std'])
    for rbp_dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, rbp_dir)):
            rbp_name = rbp_dir.split('-')[-1]
            if os.path.exists(os.path.join(path, rbp_dir, 'results.csv')):
                try:
                    file = pd.read_csv(os.path.join(path, rbp_dir, 'results.csv'))
                except pd.errors.EmptyDataError:
                    print(rbp_name, 'has no results')
                    continue
                acc, auc = list(file['seq_acc']), list(file['auc'])
                if len(acc) < 10:
                    print(rbp_name, 'lacks %d folds'%(10-len(acc)))
                logger.update_with_dict({
                    'RBP': rbp_name,
                    'acc': np.round(np.mean(acc), 3),
                    'acc_std': np.round(np.std(acc), 3),
                    'auc': np.round(np.mean(auc), 3),
                    'auc_std': np.round(np.std(auc), 3)
                })
            else:
                print(rbp_name, 'has no results')
    logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, help="")
    args = parser.parse_args()
    summarize_10fold_results(args.path, 'rnaplfold-results.csv')


