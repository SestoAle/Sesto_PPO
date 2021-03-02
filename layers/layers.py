import tensorflow as tf
from utils import *

## Layers
def linear(inp, inner_size, name='linear', bias=True, activation=None, init=None):
    with tf.compat.v1.variable_scope(name):
        lin = tf.compat.v1.layers.dense(inp, inner_size, name=name, activation=activation, use_bias=bias,
                                        kernel_initializer=init)
        return lin

def transformer(input, n_head, hidden_size, mask_value=None, num_entities=None, mlp_layer=2, pooling='max',
                    residual=True, with_embeddings=True, with_ffn=True, post_norm=False,
                    pre_norm=False, name='transformer'):

    with tf.compat.v1.variable_scope(name):
        # Utility functions
        def qkv_embed(input, heads, n_embd):
            if pre_norm:
                input = layer_norm(input, axis=3)

            qk = linear(input, hidden_size*2, name='qk')
            qk = tf.reshape(qk, (bs, T, NE, heads, n_embd // heads, 2))

            # (bs, T, NE, heads, features)
            query, key = [tf.squeeze(x, -1) for x in tf.split(qk, 2, -1)]

            value = linear(input, hidden_size, name='v')
            value = tf.reshape(value, (bs, T, NE, heads, n_embd // heads))

            query = tf.transpose(query, (0, 1, 3, 2, 4),
                                 name="transpose_query")  # (bs, T, heads, NE, n_embd / heads)
            key = tf.transpose(key, (0, 1, 3, 4, 2),
                               name="transpose_key")  # (bs, T, heads, n_embd / heads, NE)
            value = tf.transpose(value, (0, 1, 3, 2, 4),
                                 name="transpose_value")  # (bs, T, heads, NE, n_embd / heads)

            return query, key, value

        def self_attention(input, mask, heads, n_embd):
            query, key, value = qkv_embed(input, heads, n_embd)
            logits = tf.matmul(query, key, name="matmul_qk_parallel")  # (bs, T, heads, NE, NE)
            logits /= np.sqrt(n_embd / heads)
            softmax = stable_masked_softmax(logits, mask)

            att_sum = tf.matmul(softmax, value, name="matmul_softmax_value")  # (bs, T, heads, NE, features)

            out = tf.transpose(att_sum, (0, 1, 3, 2, 4))  # (bs, T, n_output_entities, heads, features)
            n_output_entities = shape_list(out)[2]
            out = tf.reshape(out, (bs, T, n_output_entities, n_embd))  # (bs, T, n_output_entities, n_embd)

            return out

        def create_mask(input, value):
            '''
                Create mask from the input. If the first element is 99, then mask it.
                The mask must be 1 for the input and 0 for the
            '''

            # x = bs, NE, feature
            mask = 1 - tf.cast(tf.equal(input[:,:,:,0], value), tf.float32)
            return mask

        #Initialize
        input = input[:, tf.newaxis, :, :]

        bs, T, NE, features = shape_list(input)
        mask = None
        if mask_value != None:
            mask = create_mask(input, mask_value)
            assert np.all(np.array(mask.get_shape().as_list()) == np.array(input.get_shape().as_list()[:3])), \
                f"Mask and input should have the same first 3 dimensions. {shape_list(mask)} -- {shape_list(input)}"
            mask = tf.expand_dims(mask, -2)  # (BS, T, 1, NE)

        if with_embeddings:
            input = linear(input, hidden_size, activation=tf.nn.tanh, name='embs')
        a = self_attention(input, mask, n_head, hidden_size)

        if residual:
            a = input + a

        if with_ffn:
            for i in range(mlp_layer - 1):
                a = linear(a, hidden_size*2, name='mlp_{}'.format(i))
            a = linear(a, hidden_size, name='mlp_{}'.format(mlp_layer))

            if residual:
                a = a + input

        input = a

        if post_norm:
            input = layer_norm(input, axis=3)

        mask = tf.reshape(mask, (bs, T, NE))

        if pooling == 'avg':
            input = entity_avg_pooling_masked(input, mask)
            bs, T, features = shape_list(input)
            input = tf.reshape(input, (bs, features))
        elif pooling == 'max':
            input = entity_max_pooling_masked(input, mask)
            bs, T, features = shape_list(input)
            input = tf.reshape(input, (bs, features))


        print(input.shape)


    return input

def layer_norm(input_tensor, axis):
  """Run layer normalization on the axis dimension of the tensor."""
  layer_norma = tf.keras.layers.LayerNormalization(axis = axis)
  return layer_norma(input_tensor)
