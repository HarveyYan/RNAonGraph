import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import lib.graphprot_dataloader

lib.graphprot_dataloader._initialize()
RBP_LIST = lib.graphprot_dataloader.all_rbps


def plot_last_nucleotide():
    pos_token, neg_token = [], []
    for seq, raw_seq, label in zip(dataset['seq'], dataset['raw_seq'], dataset['label']):
        pseudo_label = (np.array(list(raw_seq)) <= 'Z').astype(np.int32)
        last_token = 'PACGT'[seq[np.where(pseudo_label == 1)[0][-1]]]
        if label == 1:
            pos_token.append(last_token)
        else:
            neg_token.append(last_token)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.hist(pos_token)
    ax2.hist(neg_token)
    plt.savefig(os.path.join(save_path, '%s.png' % (rbp)))


weblogo_opts = '-X NO --fineprint "" --resolution "350" --format "PNG"'
weblogo_opts += ' -C "#0C8040" A A'
weblogo_opts += ' -C "#34459C" C C'
weblogo_opts += ' -C "#FBB116" G G'
weblogo_opts += ' -C "#CB2026" U U'


def save_weblogo(save_path):
    pos_logo_start, pos_logo_end = [], []
    neg_logo_start, neg_logo_end = [], []
    for seq, raw_seq, label in zip(dataset['seq'], dataset['raw_seq'], dataset['label']):
        pseudo_label = (np.array(list(raw_seq)) <= 'Z').astype(np.int32)

        viewpoint_region = list(np.where(pseudo_label == 1)[0])

        if viewpoint_region[0] - 5 >= 0:
            starting_string = ''.join(['PACGU'[seq[c]] for c in list(range(viewpoint_region[0] - 5,
                                                                           viewpoint_region[0])) + viewpoint_region[
                                                                                                   :5]])
            if np.max(label) == 1:
                pos_logo_start.append(starting_string)
            else:
                neg_logo_start.append(starting_string)
        if viewpoint_region[-1] + 6 <= len(seq):
            ending_string = ''.join(['PACGU'[seq[c]] for c in viewpoint_region[-5:] +
                                     list(range(viewpoint_region[-1] + 1,
                                                viewpoint_region[-1] + 6))])
            if np.max(label) == 1:
                pos_logo_end.append(ending_string)
            else:
                neg_logo_end.append(ending_string)

    all_strings = [pos_logo_start, pos_logo_end, neg_logo_start, neg_logo_end]

    for i, name in enumerate(['pos_start', 'pos_end', 'neg_start', 'neg_end']):

        with open(os.path.join(save_path, '%s.fa' % (name)), 'w') as f:
            for j, seq in enumerate(all_strings[i]):
                f.write('>%d\n%s\n' % (j, seq))

        weblogo_cmd = 'weblogo %s < %s > %s' % (
            weblogo_opts, os.path.join(save_path, '%s.fa' % (name)),
            os.path.join(save_path, '%s.png' % (name)))
        subprocess.call(weblogo_cmd, shell=True)
        os.remove(os.path.join(save_path, '%s.fa'%(name)))


if __name__ == "__main__":

    save_path = 'Graph/dataleak_logo_modified'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    from tqdm import tqdm

    for rbp in tqdm(RBP_LIST):
        try:
            dataset = \
                lib.graphprot_dataloader.load_clip_seq(
                    [rbp], use_embedding=False,
                    load_mat=False, nucleotide_label=True, modify_leaks=True)[0]  # load one at a time
        except ValueError as e:
            print(e)
            continue
        rbp_path = os.path.join(save_path, rbp)
        if not os.path.exists(rbp_path):
            os.makedirs(rbp_path)
        save_weblogo(rbp_path)
