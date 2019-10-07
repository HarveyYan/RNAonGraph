import os
import datetime
import numpy as np
import tensorflow as tf
from sklearn import manifold
from importlib import reload
import matplotlib.pyplot as plt
import lib.plot, lib.graphprot_dataloader, lib.rgcn_utils, lib.logger, lib.ops.LSTM, lib.rna_utils
from Model.Joint_SMRGCN import JSMRGCN

lib.graphprot_dataloader._initialize()
BATCH_SIZE = 128
RBP_LIST = lib.graphprot_dataloader.all_rbps
expr_path_list = os.listdir('output/Joint-SMRGCN-Graphprot')
expr_name = [dirname.split('-')[-1] for dirname in expr_path_list]

hp = {
    'learning_rate': 2e-4,
    'dropout_rate': 0.2,
    'use_bn': False,
    'units': 32,
    'reuse_weights': True,  # highly suggested
    'layers': 10,
    'lstm_ggnn': True,
    'probabilistic': True,
    'mixing_ratio': 0.05,
    'use_ghm': True,
}


def plot_embedding(X, y, title=None):
    if X.shape[-1] == 3:
        projection = '3d'
        from mpl_toolkits.mplot3d import Axes3D
    else:
        projection = None
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111, projection=projection)
    for i in range(X.shape[0]):
        if X.shape[-1] == 3:
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(y[i]),
                    color=plt.cm.Set1(y[i] / 5.),
                    fontdict={'weight': 'bold', 'size': 9})
        else:
            plt.text(X[i, 0], X[i, 1], str(y[i]),
                     color=plt.cm.Set1(y[i] / 5.),
                     fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def plot_tSNE(rbp, dataset, checkpoint_path):
    train_idx, test_idx = dataset['splits'][0]
    model = JSMRGCN(dataset['VOCAB_VEC'].shape[1], len(lib.graphprot_dataloader.BOND_TYPE) - 1,
                    # excluding no bond
                    dataset['VOCAB_VEC'], ['/cpu:0'], **hp)

    model.load(checkpoint_path)

    node_tensor, all_rel_data, all_row_col, segment_length = dataset['seq'][test_idx], \
                                                             dataset['all_data'][test_idx], \
                                                             dataset['all_row_col'][test_idx], \
                                                             dataset['segment_size'][test_idx]
    node_level_label = np.concatenate(dataset['label'][test_idx], axis=0)

    all_hidden_tensor = []
    batch_size = 1000
    iterations = len(node_tensor) // batch_size + (0 if len(node_tensor) % batch_size == 0 else 1)
    for i in range(iterations):
        _node_tensor, _rel_data, _row_col, _segment \
            = node_tensor[i * batch_size: (i + 1) * batch_size], \
              all_rel_data[i * batch_size: (i + 1) * batch_size], \
              all_row_col[i * batch_size: (i + 1) * batch_size], \
              segment_length[i * batch_size: (i + 1) * batch_size]
        all_adj_mat = model._merge_sparse_submatrices(_rel_data, _row_col, _segment)

        feed_dict = {
            model.node_input_ph: np.concatenate(_node_tensor, axis=0),
            **{model.adj_mat_ph[i]: all_adj_mat[i] for i in range(4)},
            model.max_len: max(_segment),
            model.segment_length: _segment,
            model.is_training_ph: False
        }
        gnn_nuc_emb = model.sess.run(model.bilstm_nuc_embedding, feed_dict)
        for segment, gne in zip(_segment, gnn_nuc_emb):
            all_hidden_tensor.append(gne[max(_segment) - segment:])
    all_hidden_tensor = np.concatenate(all_hidden_tensor, axis=0)

    pos_idx = np.random.choice(np.where(node_level_label == 1)[0], 2000, False)
    neg_idx = np.random.choice(np.where(node_level_label == 0)[0], 2000, False)
    tensor_data = np.concatenate([all_hidden_tensor[pos_idx], all_hidden_tensor[neg_idx]], axis=0)
    rna_label = np.concatenate([node_level_label[pos_idx], node_level_label[neg_idx]], axis=0)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    nuc_tsne = tsne.fit_transform(tensor_data)

    plot_embedding(nuc_tsne, rna_label)
    plt.savefig(os.path.join(output_dir, '%s-tSNE.png'%(rbp)))

    model.delete()
    reload(lib.plot)
    reload(lib.logger)


if __name__ == "__main__":
    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join('output', 'Joint-SMRGCN-tSNE-plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lib.plot.set_output_dir(output_dir)

    for rbp in RBP_LIST:
        if not os.path.exists('Data/GraphProt_CLIP_sequences/{}/train/positives/rnaplfold_True_150_prob_mat.obj'.format(rbp)):
            continue
        if os.path.exists('output/Joint-SMRGCN-tSNE-plots/{}-tSNE.png'.format(rbp)):
            continue

        print(rbp)
        dataset = \
            lib.graphprot_dataloader.load_clip_seq([rbp], use_embedding=False,
                                                   fold_algo='rnaplfold', force_folding=False,
                                                   probabilistic=True, w=150,
                                                   nucleotide_label=True)[0]  # load one at a time
        expr_path = expr_path_list[expr_name.index(rbp)]
        if not os.path.exists(os.path.join('output/Joint-SMRGCN-Graphprot', expr_path, 'splits.npy')):
            continue
        dataset['splits'] = np.load(os.path.join('output/Joint-SMRGCN-Graphprot', expr_path, 'splits.npy'),
                                    allow_pickle=True)

        plot_tSNE(rbp, dataset, tf.train.latest_checkpoint(os.path.join('output/Joint-SMRGCN-Graphprot', expr_path, 'fold0/checkpoints/')))
