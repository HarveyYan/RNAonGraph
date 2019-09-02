import os
import numpy as np
import pandas as pd
from lib.logger import CSVLogger
from lib.general_utils import compare_two_csvs

def summarize_10fold_results(path, outfile_name):
    logger = CSVLogger(outfile_name, path, ['rbp', 'acc', 'acc_std', 'auc', 'auc_std'])
    for rbp_dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, rbp_dir)):
            rbp_name = rbp_dir.split('rgcn')[-1][1:]
            if os.path.exists(os.path.join(path, rbp_dir, 'results.csv')):
                file = pd.read_csv(os.path.join(path, rbp_dir, 'results.csv'))
                acc, auc = list(file['acc']), list(file['auc'])
                if len(acc) != 10:
                    print(rbp_name)
                logger.update_with_dict({
                    'rbp': rbp_name,
                    'acc': np.round(np.mean(acc), 3),
                    'acc_std': np.round(np.std(acc), 3),
                    'auc': np.round(np.mean(auc), 3),
                    'auc_std': np.round(np.std(auc), 3)
                })
            else:
                print(rbp_name)
    logger.close()

if __name__ == "__main__":
    summarize_10fold_results('output/RGCN10folds/rnasubopt-60epochs', 'rnasubopt-results.csv')


