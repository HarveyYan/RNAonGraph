import argparse
import numpy as np
import tensorflow as tf
from Model.Legacy_RNATracker import RNATracker


BATCH_SIZE = 128
EPOCHS = 50
DEVICES = ['/gpu:0']
MAX_LEN = 100

hp = {
    'learning_rate': 2e-4,
    'dropout_rate': 0.2,
    'use_clr': True,
    'units': 128,
    'use_bn': False,
}


def load_seq(fname):
    '''

    :param fname:
    :return:
    '''
    headers=[]; species=[];  seq=[]; y=[]
    for line in open(fname):
        line = line.strip().split()
        headers.append(line[0])
        species.append(line[1].split('.')[0].replace('_', ''))
        seq.append(line[2])
        y.append(float(line[3]))

    return headers, species, seq, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("-train", "--train-name", default=None, type=str,
                        help="path to ortho train filename. Default: None")
    parser.add_argument("-validation", "--validation-name", default=None, type=str,
                        help="path to ortho validation filename. Default: None")
    parser.add_argument("-test", "--test-name", default=None, type=str,
                        help="path to ortho test filename. Default: None")
    parser.add_argument("--model-weights", default=None, type=str,
                        help="path to saved model weights. Default: None")

    args = parser.parse_args()

    # load train ortho
    train_headers, train_species, train_seq, train_y = load_seq(args.train_name)

    # load validation ortho
    val_headers, val_species, val_seq, val_y = load_seq(args.validation_name)

    # load test ortho
    test_headers, test_species, test_seq, test_y = load_seq(args.train_name)


    # load model
    VOCAB = ['NOT_FOUND', 'A', 'C', 'G', 'T', 'N']
    VOCAB_VEC = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1]]).astype(np.float32)
    model = RNATracker(MAX_LEN, VOCAB_VEC.shape[1], DEVICES, **hp)
    model.load(tf.train.latest_checkpoint(args.model_weights))

    # predict on train ortho
    train_preds = model.predict(train_seq, BATCH_SIZE)

    # predict on validation ortho
    val_preds = model.predict(val_seq, BATCH_SIZE)

    # predict on test ortho
    test_preds = model.predict(test_seq, BATCH_SIZE)

    # save predictions as df
    









