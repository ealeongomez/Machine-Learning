
# ======================================================================================
# Libraries 
# ======================================================================================
import tensorflow as tf

# ======================================================================================
# Functions 
# ======================================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
        Inputs:
            Q: Query matriz
            K: Key matriz
            V: Value matriz
            mask: Mask 
        Outputs:
            A: 
            H: Attention matrix
    """
    d_k = Q.shape[-1]
    dot_softmax = tf.matmul(Q, K, transpose_b=True) / math.sqrt(d_k)

    if mask is not None:
        dot_softmax = dot_softmax + mask

    A = tf.nn.softmax(dot_softmax, axis=-1)
    H = tf.matmul(A, V)
  
    return H, A
