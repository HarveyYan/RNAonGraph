import os
import sys
import shutil
import inspect
import datetime
import numpy as np
import tensorflow as tf
import lib.plot, lib.logger, lib.ops.LSTM, lib.rna_utils
from Model.OldRNATracker import RNATracker

tf.logging.set_verbosity(tf.logging.FATAL)
tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_integer('epochs', 50, '')
tf.app.flags.DEFINE_integer('nb_gpus', 1, '')
tf.app.flags.DEFINE_bool('use_clr', True, '')
tf.app.flags.DEFINE_integer('units', 128, '')
tf.app.flags.DEFINE_integer('maxlen', 100, '')

tf.app.flags.DEFINE_bool('predict_orthologous', False,
                         'If set to True, a saved model checkpoint file must be specified,'
                         'along with an orthologous file to make predictions on. '
                         'Otherwise, a train, valid and test set path must be each '
                         'provided to train the model from scratch.')
tf.app.flags.DEFINE_string('weight_params_path',
                           'output/RNATracker/20190705-141134-set2set-t10-128/1_PARCLIP_AGO1234_hg19/checkpoints/-20',
                           '')
tf.app.flags.DEFINE_string('ortho_fasta_path', '', '')
# if training the model from scratch
tf.app.flags.DEFINE_string('train_fasta_path', '', '')
tf.app.flags.DEFINE_string('valid_fasta_path', '', '')
tf.app.flags.DEFINE_string('test_fasta_path', '', '')

FLAGS = tf.app.flags.FLAGS

if FLAGS.predict_orthologous:
    assert (os.path.exists(FLAGS.ortho_fasta_path))
else:
    assert (os.path.exists(FLAGS.train_fasta_path))
    assert (os.path.exists(FLAGS.valid_fasta_path))
    assert (os.path.exists(FLAGS.test_fasta_path))

BATCH_SIZE = 200  # 200 * FLAGS.nb_gpus if FLAGS.nb_gpus > 0 else 200
EPOCHS = FLAGS.epochs  # How many iterations to train for
DEVICES = ['/gpu:%d' % (i) for i in range(FLAGS.nb_gpus)] if FLAGS.nb_gpus > 0 else ['/cpu:0']
MAX_LEN = FLAGS.maxlen

hp = {
    'learning_rate': 2e-4,
    'dropout_rate': 0.2,
    'use_clr': FLAGS.use_clr,
    'units': FLAGS.units,
    'use_bn': False,
}


def run():
    # outfile = open(os.path.join(output_dir, str(os.getpid())) + ".out", "w")
    # sys.stdout = outfile
    # sys.stderr = outfile

    # P stands for unknown character, which should not appear at all in any fasta file
    VOCAB = ['NOT_FOUND', 'A', 'C', 'G', 'T', 'N']
    VOCAB_VEC = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1]]).astype(np.float32)

    if FLAGS.predict_orthologous:

        all_id, all_seq = lib.rna_utils.load_seq(FLAGS.ortho_fasta_path)
        # labels = np.array([int(id.split(' ')[-1].split(':')[-1]) for id in all_id])
        seqs = np.array([[VOCAB.index(c) for c in seq] for seq in all_seq])
        assert(seqs.shape[1]==MAX_LEN)
        model = RNATracker(MAX_LEN, VOCAB_VEC.shape[1], DEVICES, **hp)

        model.load(FLAGS.weight_params_path)
        all_preds = model.predict(seqs, BATCH_SIZE)
        with open(os.path.join(output_dir, 'prediction.fa'), 'w') as output_file:
            for header, pred in zip(all_id, all_preds):
                output_file.write('%s %.4f\n' % (header, pred[1]))

    else:

        train_id, train_seq = lib.rna_utils.load_seq(FLAGS.train_fasta_path)
        train_labels = np.array([int(id.split(' ')[-1]) for id in train_id])
        train_seq = np.array([[VOCAB.index(c) for c in seq] for seq in train_seq])
        assert (train_seq.shape[1] == MAX_LEN)
        valid_id, valid_seq = lib.rna_utils.load_seq(FLAGS.valid_fasta_path)
        valid_labels = np.array([int(id.split(' ')[-1]) for id in valid_id])
        valid_seq = np.array([[VOCAB.index(c) for c in seq] for seq in valid_seq])
        assert (valid_seq.shape[1] == MAX_LEN)
        test_id, test_seq = lib.rna_utils.load_seq(FLAGS.test_fasta_path)
        test_labels = np.array([int(id.split(' ')[-1]) for id in test_id])
        test_seq = np.array([[VOCAB.index(c) for c in seq] for seq in test_seq])
        assert (test_seq.shape[1] == MAX_LEN)

        model = RNATracker(MAX_LEN, VOCAB_VEC.shape[1], DEVICES, **hp)
        model.fit(train_seq, train_labels, EPOCHS, BATCH_SIZE, output_dir, valid_seq, valid_labels, logging=True)
        all_predicton, acc, auc = model.predict(test_seq, BATCH_SIZE, y=test_labels)
        print('Evaluation on held-out test set, acc: %.3f, auc: %.3f' % (acc, auc))
        model.delete()
        lib.plot.reset()


if __name__ == "__main__":

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if FLAGS.output_dir == '':
        output_dir = os.path.join('output', 'PhyloPredictions', cur_time)
    else:
        output_dir = os.path.join('output', 'PhyloPredictions', cur_time + '-' + FLAGS.output_dir)

    os.makedirs(output_dir)
    lib.plot.set_output_dir(output_dir)

    # backup python scripts, for future reference
    backup_dir = os.path.join(output_dir, 'backup/')
    os.makedirs(backup_dir)
    shutil.copy(__file__, backup_dir)
    shutil.copy(inspect.getfile(RNATracker), backup_dir)
    shutil.copy(inspect.getfile(lib.ops.LSTM), backup_dir)

    run()
