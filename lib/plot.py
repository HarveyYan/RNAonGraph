import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import forgi
from forgi.visual.mplotlib import _find_annot_pos_on_circle
import colorsys
import RNA
import matplotlib.colors as mc
import shutil
import subprocess as sp

weblogo_opts = '-X NO --fineprint "" --resolution "350" --format "PNG"'
weblogo_opts += ' -C "#CB2026" A A'
weblogo_opts += ' -C "#34459C" C C'
weblogo_opts += ' -C "#FBB116" G G'
weblogo_opts += ' -C "#0C8040" T T'
weblogo_opts += ' -C "#0C8040" U U'

weblogo_opts += ' -C "#CB2026" M M'
weblogo_opts += ' -C "#FBB116" I I'
weblogo_opts += ' -C "#0C8040" S S'
weblogo_opts += ' -C "#34459C" F F'
weblogo_opts += ' -C "#34459C" T T'
weblogo_opts += ' -C "#34459C" E E' # external regions that include F/T

plt.style.use('classic')
matplotlib.rcParams.update({'figure.figsize': [10.0, 10.0], 'font.family': 'Times New Roman', 'figure.dpi': 350})
matplotlib.rcParams['agg.path.chunksize'] = 1e10
import collections

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]

_output_dir = ''
_stdout = True


def suppress_stdout():
    global _stdout
    _stdout = False


def set_output_dir(output_dir):
    global _output_dir
    _output_dir = output_dir


def tick():
    _iter[0] += 1


def plot(name, value):
    if type(value) is tuple:
        _since_last_flush[name][_iter[0]] = np.array(value)
    else:
        _since_last_flush[name][_iter[0]] = value


def flush():
    prints = []

    for name, vals in _since_last_flush.items():
        prints.append("{}: {}\t".format(name, np.mean(np.array(list(vals.values())), axis=0)))
        _since_beginning[name].update(vals)

        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = np.array([_since_beginning[name][x] for x in x_vals])

        plt.clf()
        if len(y_vals.shape) == 1:
            plt.plot(x_vals, y_vals)
        else:  # with standard deviation
            plt.plot(x_vals, y_vals[:, 0])
            plt.fill_between(x_vals, y_vals[:, 0] - y_vals[:, 1], y_vals[:, 0] + y_vals[:, 1], alpha=0.5)
        plt.xlabel('epoch')
        plt.ylabel(name)
        plt.savefig(os.path.join(_output_dir, name.replace(' ', '_') + '.jpg'), dpi=350)

    if _stdout:
        print("epoch {}\t{}".format(_iter[0], "\t".join(prints)))
    _since_last_flush.clear()


def reset():
    global _since_beginning, _since_last_flush, _iter, _output_dir, _stdout
    _since_beginning = collections.defaultdict(lambda: {})
    _since_last_flush = collections.defaultdict(lambda: {})

    _iter = [0]

    _output_dir = ''
    _stdout = True


'''
The following motif logo visualization scripts are adapted from
https://github.com/kundajelab/deeplift/tree/master/deeplift/visualization
'''


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.2, 0.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.8, 0.0],
        ]),
        np.array([
            [0.225, 0.45],
            [0.775, 0.45],
            [0.85, 0.3],
            [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.825, base + 0.085 * height], width=0.174, height=0.415 * height,
                                     facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.625, base + 0.35 * height], width=0.374, height=0.15 * height,
                                     facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.4, base],
                                              width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.8 * height],
                                              width=1.0, height=0.2 * height, facecolor=color, edgecolor=color,
                                              fill=True))


def plot_u(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.5, base + 0.3 * height], width=1., height=0.6 * height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.5, base + 0.3 * height], width=0.7 * 1., height=0.7 * 0.6 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.3 * height], width=1.0, height=0.7 * height,
                                              facecolor='white', edgecolor='white', fill=True))

    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.29 * height], width=0.01, height=0.71 * height,
                                              facecolor=color, edgecolor=color, fill=True))

    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.95, base + 0.29 * height], width=0.05, height=0.71 * height,
                                     facecolor=color, edgecolor=color, fill=True))


default_colors = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red'}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_u}


def plot_weights_given_ax(ax, array,
                          height_padding_factor,
                          length_padding,
                          subticks_frequency,
                          highlight,
                          colors=default_colors,
                          plot_funcs=default_plot_funcs):
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if (array.shape[0] == 4 and array.shape[1] != 4):
        array = array.transpose(1, 0)
    assert array.shape[1] == 4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        # sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # now highlight any desired positions; the key of
    # the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos, min_depth],
                                             width=end_pos - start_pos,
                                             height=max_height - min_depth,
                                             edgecolor=color, fill=False))

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))
    height_padding = max(abs(min_neg_height) * (height_padding_factor),
                         abs(max_pos_height) * (height_padding_factor))
    ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)


def plot_weights(array,
                 figsize=(20, 2),
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={}, save_path=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plot_weights_given_ax(ax=ax, array=array,
                          height_padding_factor=height_padding_factor,
                          length_padding=length_padding,
                          subticks_frequency=subticks_frequency,
                          colors=colors,
                          plot_funcs=plot_funcs,
                          highlight=highlight)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=350)
    plt.close(fig)


def plot_rna_struct(seq, struct, ax=None, offset=(0, 0), text_kwargs={}, backbone_kwargs={},
                    basepair_kwargs={}, highlight_bp_idx=[], highlight_nt_idx=[], lighten=0.7, saveto='tmp.png'):
    with open('tmp.fa', 'w') as file:
        file.write('>tmp\n%s\n%s' % (seq, struct))
    cg = forgi.load_rna('tmp.fa', allow_many=False)

    RNA.cvar.rna_plot_type = 1

    fig = plt.figure(figsize=(30, 30))
    coords = []

    bp_string = cg.to_dotbracket_string()

    if ax is None:
        ax = plt.gca()

    if offset is None:
        offset = (0, 0)
    elif offset is True:
        offset = (ax.get_xlim()[1], ax.get_ylim()[1])
    else:
        pass

    vrna_coords = RNA.get_xy_coordinates(bp_string)
    # TODO Add option to rotate the plot
    for i, _ in enumerate(bp_string):
        coord = (offset[0] + vrna_coords.get(i).X,
                 offset[1] + vrna_coords.get(i).Y)
        coords.append(coord)
    coords = np.array(coords)
    # First plot backbone
    bkwargs = {"color": "grey", "zorder": 0, "linewidth": 0.5}
    bkwargs.update(backbone_kwargs)
    ax.plot(coords[:, 0], coords[:, 1], **bkwargs)
    # Now plot basepairs
    basepairs_hl, basepairs_nonhl = [], []
    for s in cg.stem_iterator():
        for p1, p2 in cg.stem_bp_iterator(s):
            if (p1 - 1, p2 - 1) in highlight_bp_idx:
                basepairs_hl.append([coords[p1 - 1], coords[p2 - 1]])
            else:
                basepairs_nonhl.append([coords[p1 - 1], coords[p2 - 1]])

    if len(basepairs_hl) > 0:
        basepairs_hl = np.array(basepairs_hl)
        bpkwargs_hl = {"color": 'red', "zorder": 0, "linewidth": 3}
        bpkwargs_hl.update(basepair_kwargs)
        ax.plot(basepairs_hl[:, :, 0].T, basepairs_hl[:, :, 1].T, **bpkwargs_hl)

    if len(basepairs_nonhl) > 0:
        basepairs_nonhl = np.array(basepairs_nonhl)
        bpkwargs_nonhl = {"color": 'black', "zorder": 0, "linewidth": 0.5}
        bpkwargs_nonhl.update(basepair_kwargs)
        ax.plot(basepairs_nonhl[:, :, 0].T, basepairs_nonhl[:, :, 1].T, **bpkwargs_nonhl)

    # Now plot circles
    for i, coord in enumerate(coords):

        if i in highlight_nt_idx:
            c = 'green'
            h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
            if lighten > 0:
                l += (1 - l) * min(1, lighten)
            else:
                l += l * max(-1, lighten)
            c = colorsys.hls_to_rgb(h, l, s)
            circle = plt.Circle((coord[0], coord[1]),
                                edgecolor="black", facecolor=c)
        else:
            circle = plt.Circle((coord[0], coord[1]),
                                edgecolor="black", facecolor="white")

        ax.add_artist(circle)
        if cg.seq:
            if "fontweight" not in text_kwargs:
                text_kwargs["fontweight"] = "bold"
            ax.annotate(cg.seq[i + 1], xy=coord, ha="center", va="center", **text_kwargs)

    all_coords = list(coords)
    ntnum_kwargs = {"color": "gray"}
    ntnum_kwargs.update(text_kwargs)
    for nt in range(10, cg.seq_length, 10):
        # We try different angles
        annot_pos = _find_annot_pos_on_circle(nt, all_coords, cg)
        if annot_pos is not None:
            ax.annotate(str(nt), xy=coords[nt - 1], xytext=annot_pos,
                        arrowprops={"width": 1, "headwidth": 1, "color": "gray"},
                        ha="center", va="center", zorder=0, **ntnum_kwargs)
            all_coords.append(annot_pos)

    datalim = ((min(list(coords[:, 0]) + [ax.get_xlim()[0]]),
                min(list(coords[:, 1]) + [ax.get_ylim()[0]])),
               (max(list(coords[:, 0]) + [ax.get_xlim()[1]]),
                max(list(coords[:, 1]) + [ax.get_ylim()[1]])))

    ax.set_aspect('equal', 'datalim')
    ax.update_datalim(datalim)
    ax.autoscale_view()
    ax.set_axis_off()

    plt.savefig(saveto, dpi=350)
    plt.close(fig)


def plot_weblogo(msa_file_path, save_path):
    if shutil.which("weblogo") is None:
        print('weblogo command is not available!')
        return

    weblogo_cmd = 'weblogo %s < %s > %s' % (
        weblogo_opts, msa_file_path, save_path)
    sp.call(weblogo_cmd, shell=True)