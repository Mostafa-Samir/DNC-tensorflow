import tensorflow as tf

def pairwise_add(u, v=None):
    """
    performs a pairwise summation between vectors (possibly the same)

    Parameters:
    ----------
    u: Tensor (n, ) | (n, 1)
    v: Tensor (n, ) | (n, 1) [optional]

    Returns: Tensor (n, n)
    Raises: ValueError
    """
    u_shape = u.get_shape().as_list()

    if len(u_shape) > 2:
        raise ValueError("Expected at most 2D tensors, but got %dD" % len(u_shape))
    if len(u_shape) == 2 and u_shape[1] != 1:
        raise ValueError("Expected a vector, but got a matrix")

    if v is None:
        v = u
    else:
        v_shape = v.get_shape().as_list()
        if u_shape != v_shape:
            raise VauleError("Shapes %s and %s do not match" % (u_shape, v_shape))

    n = u_shape[0]

    column_u = tf.reshape(u, (-1, 1))
    U = tf.concat(1, [column_u] * n)

    if v is u:
        return U + tf.transpose(U)
    else:
        row_v = tf.reshape(v, (1, -1))
        V = tf.concat(0, [row_v] * n)

        return U + V
