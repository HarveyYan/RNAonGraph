import argparse
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from Model.Legacy_RNATracker import RNATracker
from tqdm import tqdm
from multiprocessing import Pool


BATCH_SIZE = 1000
EPOCHS = 50
MAX_LEN = 101

hp = {
    'learning_rate': 2e-4,
    'dropout_rate': 0.2,
    'use_clr': True,
    'units': 128,
    'use_bn': False,
}

VOCAB = ['NOT_FOUND', 'A', 'C', 'G', 'T', 'N']
VOCAB_VEC = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]]).astype(np.float32)


def load_seq(fname):
    '''

    :param fname:
    :return:
    '''
    headers=[]; species=[];  seqs=[]; y=[]
    for line in open(fname):
        line = line.strip().split()
        headers.append(line[0])
        species.append(line[1].split('.')[0].replace('_', ''))
        seqs.append(line[2])
        y.append(float(line[3]))

    return headers, species, seqs, y

def fill_df(values):
    index, col, cell_value, df_name = values
    if df_name == 'validate':
        df_validate.loc[index, col] = cell_value
    else:
        df_test.loc[index, col] = cell_value
    return

def fill_y(values):
    index, col, cell_value, df_name = values
    if df_name == 'validate':
        if df_validate.loc[index, col] is not None:
            df_validate.loc[index, col] = cell_value
    else:
        if df_test.loc[index, col] is not None:
            df_test.loc[index, col] = cell_value
    return


def create_df(headers, species, labels, predictions, df_name):
    """

    :param headers:
    :param species:
    :param labels:
    :param predictions:
    :return:
    """

    if df_name == 'validate':
        values_df = [(headers[i], species[i], predictions[i], df_name) for i in range(len(predictions))]
    else:
        values_df = [(headers[i], species[i], predictions[i], df_name) for i in range(len(predictions))]


    p = Pool(16)
    tqdm(p.map(fill_df, values_df))

    if df_name == 'validate':
        values_df = [(headers[i], species[i], labels[i], df_name) for i in range(len(labels))]
    else:
        values_df = [(headers[i], species[i], labels[i], df_name) for i in range(len(labels))]
    tqdm(p.map(fill_y, values_df))

    # for index in tqdm(range(len(predictions))):
    #     df.loc[headers[index], species[index]] = predictions[index]
    #     if df.loc[headers[index], 'y'] is np.nan:
    #         df.loc[headers[index], 'y'] = labels[index]
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("-train", "--train-name", default=None, type=str,
                        help="path to ortho train filename. Default: None")
    parser.add_argument("-validation", "--validation-name", default=None, type=str,
                        help="path to ortho validation filename. Default: None")
    parser.add_argument("-test", "--test-name", default=None, type=str,
                        help="path to ortho test filename. Default: None")
    parser.add_argument("-gpu", "--gpu-card", default=None, type=str,
                        help="GPU Device Number. Default: None")
    parser.add_argument("--info-tree", default=None, type=str,
                        help="path to info tree. Default: None")
    parser.add_argument("--model-weights", default=None, type=str,
                        help="path to saved model weights. Default: None")
    parser.add_argument("--save-path", default=None, type=str,
                        help="path to save predictions. Default: None")

    args = parser.parse_args()

    DEVICES = ['/gpu:'+args.gpu_card] if args.gpu_card is not None else ['/cpu:0']
    # DEVICES = ['/xla_gpu:9']
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_card

    # # load train ortself.gpu_device_list[0]ho
    # train_headers, train_species, train_seqs, train_y = load_seq(args.train_name)
    # train_seqs = np.array([[VOCAB.index(c) for c in seq] for seq in train_seqs])
    # print ('train_seqs:', train_seqs.shape)

    # load validation ortho
    val_headers, val_species, val_seqs, val_y = load_seq(args.validation_name)
    val_seqs = np.array([[VOCAB.index(c) for c in seq] for seq in val_seqs])
    print ('val_seqs:', val_seqs.shape)

    # load test ortho
    test_headers, test_species, test_seqs, test_y = load_seq(args.test_name)
    test_seqs = np.array([[VOCAB.index(c) for c in seq] for seq in test_seqs])
    print ('test_seqs:', test_seqs.shape)

    # load model

    model = RNATracker(MAX_LEN, VOCAB_VEC.shape[1], DEVICES, **hp)
    model.load(tf.train.latest_checkpoint(args.model_weights))

    # # predict on train ortho
    # train_preds = model.predict(train_seqs, BATCH_SIZE)[:,1]
    # # print ('train_preds:', train_preds.shape)

    # predict on validation ortho
    val_preds = model.predict(val_seqs, BATCH_SIZE)[:,1]
    # print ('val_preds:', val_preds)
    # print ('val_preds:', val_preds.shape)

    # predict on test ortho
    test_preds = model.predict(test_seqs, BATCH_SIZE)[:,1]

    # save predictions as df
    # get list of sp ids
    info_tree = pickle.load(open(args.info_tree, 'rb'))
    sp_list = sorted(info_tree['sp_to_id'], key=info_tree['sp_to_id'].get)

    # # create pandas
    # df_train = create_df(train_headers, train_species, train_y, train_preds)
    # # print ('df_train:', df_train)
    # print ('df_train:', df_train.shape)
    # pickle.dump(df_train, open(args.save_path+'/df_train.pkl', 'wb'))

    df_validate = pd.DataFrame(index=set(val_headers), columns=sp_list+['y'])
    create_df(val_headers, val_species, val_y, val_preds, 'validate')
    pickle.dump(df_validate, open(args.save_path+'/df_validate.pkl', 'wb'))
    print ('df_validate:', df_validate)
    print ('df_validate:', df_validate.shape)

    df_test = pd.DataFrame(index=set(test_headers), columns=sp_list+['y'])
    create_df(test_headers, test_species, test_y, test_preds, 'test')
    pickle.dump(df_test, open(args.save_path+'/df_test.pkl', 'wb'))
    print ('df_test:', df_test)
    print ('df_test:', df_test.shape)










