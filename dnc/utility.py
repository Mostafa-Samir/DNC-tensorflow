import tensorflow as tf
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


def allocate(shape, dtype=tf.float32, name=None):
    """
    allocates a temporary variable with given shape, dtype,and name

    Parameters:
    ----------
    shape: iterable
        the desired shape of the variable
    dtype: tf.DType
        the variable's data type
    name: string
        the name of the variable

    Returns: Tuple
        tf.Variable: the allocated variable,
        string: the allocation op name (used for deallocating later)
    """

    var = gen_state_ops._temporary_variable(shape=shape, dtype=dtype)
    var_name = var.op.name

    return var, var_name


def read_and_deallocate(var, var_name):
    """
    deallocates a previously allocted temporary variable and returns its value

    Parameters:
    ----------
    var: tf.Variable
        the previously allocated variable
    var_name: string
        the previously allocated variable allocation's op name

    Returns: Tensor
        the variable's value
    """

    return gen_state_ops._destroy_temporary_variable(var, var_name=var_name)
