import os
import numpy as np
import pandas as pd
from lib.logger import CSVLogger
import argparse


def summarize_10fold_results(path, outfile_name):
    logger = CSVLogger(outfile_name, path,
                       ['RBP', 'auc', 'auc_std', 'original_auc', 'original_auc_std'])
    all_dicts = []
    for rbp_dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, rbp_dir)):
            rbp_name = rbp_dir.split('-')[-1]
            if os.path.exists(os.path.join(path, rbp_dir, 'results.csv')):
                try:
                    file = pd.read_csv(os.path.join(path, rbp_dir, 'results.csv'))
                except pd.errors.EmptyDataError:
                    print(rbp_name, 'has no results')
                    continue
                auc, original_auc = list(file['auc']), list(file['original_auc'])
                if len(auc) < 10:
                    print(rbp_name, 'lacks %d folds' % (10 - len(auc)))
                all_dicts.append({
                    'RBP': rbp_name,
                    'auc': np.round(np.mean(auc), 3),
                    'auc_std': np.round(np.std(auc), 3),
                    'original_auc': np.round(np.mean(original_auc), 3),
                    'original_auc_std': np.round(np.std(original_auc), 3)
                })
            else:
                print(rbp_name, 'has no results')
    all_dicts = list(np.array(all_dicts)[np.argsort([d['RBP'] for d in all_dicts])])
    logger.update_with_dicts(all_dicts)
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, help="")
    args = parser.parse_args()
    summarize_10fold_results(args.path, 'rnaplfold-results.csv')
