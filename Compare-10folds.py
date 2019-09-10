import os
import numpy as np
import pandas as pd
from lib.logger import CSVLogger
from lib.general_utils import compare_two_csvs

def summarize_10fold_results(path, outfile_name):
    logger = CSVLogger(outfile_name, path, ['RBP', 'acc', 'acc_std', 'auc', 'auc_std'])
    for rbp_dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, rbp_dir)):
            rbp_name = rbp_dir.split('-')[-1]
            if os.path.exists(os.path.join(path, rbp_dir, 'results.csv')):
                try:
                    file = pd.read_csv(os.path.join(path, rbp_dir, 'results.csv'))
                except pd.errors.EmptyDataError:
                    print(rbp_name, 'has lagged behind considerably')
                    continue
                acc, auc = list(file['acc']), list(file['auc'])
                if len(acc) != 10:
                    print(rbp_name, 'hasn\'t yet reached 10 folds')
                logger.update_with_dict({
                    'RBP': rbp_name,
                    'acc': np.round(np.mean(acc), 3),
                    'acc_std': np.round(np.std(acc), 3),
                    'auc': np.round(np.mean(auc), 3),
                    'auc_std': np.round(np.std(auc), 3)
                })
            else:
                print(rbp_name)
    logger.close()

if __name__ == "__main__":
    summarize_10fold_results('output/SMRGCN-Graphprot', 'rnaplfold-results.csv')


