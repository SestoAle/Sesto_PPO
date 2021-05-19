import tensorflow as tf
from layers.layers import *

def input_spec():
    input_length = 68
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    return [global_state]

def obs_to_state(obs):
    global_batch = np.stack([np.asarray(state['global_in']) for state in obs])
    return [global_batch]

def network_spec(states):
    input_length = 68
    with_circular = False

    global_state = states[0]
    if input_length > 0:
        global_state, rays, coins, obstacles = tf.split(global_state, [7, 12, 28, 21], axis=1)
        global_state = linear(global_state, 1024, name='embs', activation=tf.nn.relu)

        rays = tf.cast(tf.reshape(rays, [-1, 5, 5]), tf.int32)
        rays = embedding(rays, indices=4, size=32, name='rays_embs')
        rays = conv_layer_2d(rays, 32, [3, 3], name='conv_31', activation=tf.nn.relu)
        rays = conv_layer_2d(rays, 64, [3, 3], name='conv_32', activation=tf.nn.relu)
        rays = tf.reshape(rays, [-1, 5 * 5 * 64])

        BS, f = shape_list(coins)
        my_mask = tf.concat([tf.ones((BS, 3)), tf.zeros((BS, 1))], axis = 1)
        my_mask = my_mask[:, tf.newaxis, :]
        coins = tf.reshape(coins, [-1, 14, 2])
        coins, _, mask = transformer(coins, n_head=4, hidden_size=1024, mask_value=99, with_embeddings=True,
                               name='transformer_coins', mask=my_mask)
        mask = tf.compat.v1.Print(mask, [mask], 'mask ', summarize=1e5)
        coins = entity_max_pooling_masked(coins, mask)
        coins = tf.reshape(coins, [-1, 1024])

        obstacles = tf.reshape(obstacles, [-1, 7, 3])
        obstacles = linear(obstacles, 1024, name='embs_obs', activation=tf.nn.relu)
        obstacles, _, mask = transformer(obstacles, n_head=4, hidden_size=1024, mask_value=99, with_embeddings=False,
                                   name='transformer_global')
        obstacles = tf.math.reduce_max(obstacles, axis=2)
        obstacles = tf.reshape(obstacles, [-1, 1024])



        global_state = tf.concat([global_state, coins], axis=1)

    else:
        # agent, goal, rays, obs = tf.split(global_state, [4, 3, 12, 21], axis=1)
        # Jump
        agent, goal, rays, obs, points = tf.split(global_state, [4, 3, 12, 21, 12], axis=1)

        rays = tf.reshape(rays, [-1, 12, 1])
        rays = circ_conv1d(rays, activation='relu', kernel_size=3, filters=32)
        rays = tf.reshape(rays, [-1, 12 * 32])

        points = tf.reshape(points, [-1, 4, 3])
        points, _ = transformer(points, n_head=4, hidden_size=1024, mask_value=99, with_embeddings=True,
                                   name='transformer_global', pooling='max')
        points = tf.reshape(points, [-1, 1024])

        global_state = tf.concat([goal, rays], axis=1)

        global_state = linear(global_state, 1024, name='embs', activation=tf.nn.relu)

    return global_state
