import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('ggplot')
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
        plt.savefig(os.path.join(_output_dir, name.replace(' ', '_') + '.jpg'))

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
