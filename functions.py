from functools import reduce
import tensorflow as tf

def make_keys_grid(shape):
    """
    Create a tensor of shape `shape + [shape.size]`.
    Used to create a set of integer coordinates such that
    t[i_1, i_2, ..., i_k] = [i_1, i_2, ..., i_k].
    """
    num_axes = tf.size(shape)
    X = []
    for axis in tf.range(num_axes):
        s = shape[axis]
        x = tf.range(s, dtype=tf.int32)
        x = tf.reshape(x, [-1, 1])
        left_padding = axis
        right_padding = num_axes - 1 - axis
        x = tf.pad(x, [[0, 0], [left_padding, right_padding]])
        for _ in tf.range(axis):
            x = tf.expand_dims(x, 0)
        for _ in tf.range(num_axes - axis - 1):
            x = tf.expand_dims(x, -2)
        X.append(x)
    return reduce(lambda x, y: x + y, X)
