import locale
import tensorflow as tf

def _stats(name, grads_and_vars):
    # show all trainable weights
    print("{} Params:".format(name))
    total_param_count = 0
    for g, v in grads_and_vars:
        shape = v.get_shape()
        shape_str = ",".join([str(x) for x in v.get_shape()])

        param_count = 1
        for dim in shape:
            param_count *= int(dim)
        total_param_count += param_count

        if g == None:
            print("\t{} ({}) [no grad!]".format(v.name, shape_str))
        else:
            print("\t{} ({})".format(v.name, shape_str))
    print("Total param count: {}".format(
        locale.format("%d", total_param_count, grouping=True)
    ))

def _average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class Model(object):
    def __init__(self, **kwargs):
        pass

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError