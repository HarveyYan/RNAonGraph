import argparse
import numpy as np
import tensorflow as tf
import pickle
from Model.Legacy_RNATracker import RNATracker


BATCH_SIZE = 128
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
    parser.add_argument("--model-weights", default=None, type=str,
                        help="path to saved model weights. Default: None")

    args = parser.parse_args()

    DEVICES = ['/gpu:'+args.gpu_card] if args.gpu is not None else ['/cpu:0']

    # load train ortho
    train_headers, train_species, train_seqs, train_y = load_seq(args.train_name)
    train_seqs = np.array([[VOCAB.index(c) for c in seq] for seq in train_seqs])
    print ('train_seqs:', train_seqs.shape)

    # load validation ortho
    val_headers, val_species, val_seqs, val_y = load_seq(args.validation_name)
    val_seqs = np.array([[VOCAB.index(c) for c in seq] for seq in val_seqs])
    print ('val_seqs:', val_seqs.shape)

    # load test ortho
    test_headers, test_species, test_seqs, test_y = load_seq(args.train_name)
    test_seqs = np.array([[VOCAB.index(c) for c in seq] for seq in test_seqs])
    print ('test_seqs:', test_seqs.shape)

    # load model

    model = RNATracker(MAX_LEN, VOCAB_VEC.shape[1], DEVICES, **hp)
    model.load(tf.train.latest_checkpoint(args.model_weights))

    # predict on train ortho
    train_preds = model.predict(train_seqs, BATCH_SIZE)
    # print ('train_preds:', train_preds.shape)

    # predict on validation ortho
    val_preds = model.predict(val_seqs, BATCH_SIZE)
    print ('val_preds:', val_preds)
    print ('val_preds:', val_preds.shape)

    # predict on test ortho
    test_preds = model.predict(test_seqs, BATCH_SIZE)

    # save predictions as df










