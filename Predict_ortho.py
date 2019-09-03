import argparse

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
        y.append(line[3])

    return headers, species, seq, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("-train", "--train-name", default=None, type=str,
                        help="path to ortho train filename. Default: None")
    parser.add_argument("-validation", "--validation-name", default=None, type=str,
                        help="path to ortho validation filename. Default: None")
    parser.add_argument("-test", "--test-name", default=None, type=str,
                        help="path to ortho test filename. Default: None")
    parser.add_argument("--model-name", default=None, type=str,
                        help="path to saved model. Default: None")

    args = parser.parse_args()

    # load train ortho
    train_headers, train_species, train_seq, train_y = load_seq(args.train_name)

    # load validation ortho
    val_headers, val_species, val_seq, val_y = load_seq(args.validation_name)

    # load test ortho
    test_headers, test_species, test_seq, test_y = load_seq(args.train_name)


    # load model

    # predict on sequences

    # save predictions as df