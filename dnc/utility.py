import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gen_state_ops

def pairwise_add(u, v=None, is_batch=False):
    """
    performs a pairwise summation between vectors (possibly the same)

    Parameters:
    ----------
    u: Tensor (n, ) | (n, 1)
    v: Tensor (n, ) | (n, 1) [optional]
    is_batch: bool
        a flag for whether the vectors come in a batch
        ie.: whether the vectors has a shape of (b,n) or (b,n,1)

    Returns: Tensor (n, n)
    Raises: ValueError
    """
    u_shape = u.get_shape().as_list()

    if len(u_shape) > 2 and not is_batch:
        raise ValueError("Expected at most 2D tensors, but got %dD" % len(u_shape))
    if len(u_shape) > 3 and is_batch:
        raise ValueError("Expected at most 2D tensor batches, but got %dD" % len(u_shape))

    if v is None:
        v = u
    else:
        v_shape = v.get_shape().as_list()
        if u_shape != v_shape:
            raise VauleError("Shapes %s and %s do not match" % (u_shape, v_shape))

    n = u_shape[0] if not is_batch else u_shape[1]

    column_u = tf.reshape(u, (-1, 1) if not is_batch else (-1, n, 1))
    U = tf.concat(1 if not is_batch else 2, [column_u] * n)

    if v is u:
        return U + tf.transpose(U, None if not is_batch else [0, 2, 1])
    else:
        row_v = tf.reshape(v, (1, -1) if not is_batch else (-1, 1, n))
        V = tf.concat(0 if not is_batch else 1, [row_v] * n)

        return U + V


def decaying_softmax(shape, axis):
    rank = len(shape)
    max_val = shape[axis]

    weights_vector = np.arange(1, max_val + 1, dtype=np.float32)
    weights_vector = weights_vector[::-1]
    weights_vector = np.exp(weights_vector) / np.sum(np.exp(weights_vector))

    container = np.zeros(shape, dtype=np.float32)
    broadcastable_shape = [1] * rank
    broadcastable_shape[axis] = max_val

    return container + np.reshape(weights_vector, broadcastable_shape)

def unpack_into_tensorarray(value, axis, size=None):
    """
    unpacks a given tensor along a given axis into a TensorArray

    Parameters:
    ----------
    value: Tensor
        the tensor to be unpacked
    axis: int
        the axis to unpack the tensor along
    size: int
        the size of the array to be used if shape inference resulted in None

    Returns: TensorArray
        the unpacked TensorArray
    """

    shape = value.get_shape().as_list()
    rank = len(shape)
    dtype = value.dtype
    array_size = shape[axis] if not shape[axis] is None else size

    if array_size is None:
        raise ValueError("Can't create TensorArray with size None")

    array = tf.TensorArray(dtype=dtype, size=array_size)
    dim_permutation = [axis] + range(1, axis) + [0] + range(axis + 1, rank)
    unpack_axis_major_value = tf.transpose(value, dim_permutation)
    full_array = array.unpack(unpack_axis_major_value)

    return full_array

def pack_into_tensor(array, axis):
    """
    packs a given TensorArray into a tensor along a given axis

    Parameters:
    ----------
    array: TensorArray
        the tensor array to pack
    axis: int
        the axis to pack the array along

    Returns: Tensor
        the packed tensor
    """

    packed_tensor = array.pack()
    shape = packed_tensor.get_shape()
    rank = len(shape)

    dim_permutation = [axis] + range(1, axis) + [0]  + range(axis + 1, rank)
    correct_shape_tensor = tf.transpose(packed_tensor, dim_permutation)

    return correct_shape_tensor
