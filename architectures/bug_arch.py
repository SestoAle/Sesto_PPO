import tensorflow as tf
from layers.layers import *

def input_spec():
    input_length = 44
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    return [global_state]

def obs_to_state(obs):
    global_batch = np.stack([np.asarray(state['global_in']) for state in obs])
    return [global_batch]

def network_spec(states):
    input_length = 44
    with_circular = False

    global_state = states[0]
    if input_length > 44:
        global_state, global_grid, rays, coins, obstacles = tf.split(global_state, [7, 225, 25, 28, 21], axis=1)
        global_state = linear(global_state, 1024, name='embs', activation=tf.nn.relu)

        if with_circular:
            # rays = tf.reshape(rays, [-1, 14, 5])
            # rays, _ = transformer(rays, n_head=4, hidden_size=1024, mask_value=99, with_embeddings=True,
            #                           name='transformer_local', pooling='max')
            # rays = tf.reshape(rays, [-1, 1024])
            # rays = tf.reshape(rays, [-1, 8, 5])
            # rays = circ_conv1d(rays, activation='relu', kernel_size=3, filters=32)
            # rays = tf.reshape(rays, [-1, 8 * 32])
            #

            global_grid = tf.cast(tf.reshape(global_grid, [-1, 15, 15]), tf.int32)
            global_grid = embedding(global_grid, indices=4, size=32, name='global_embs')
            global_grid = conv_layer_2d(global_grid, 32, [3, 3], name='conv_01', activation=tf.nn.relu)
            global_grid = conv_layer_2d(global_grid, 64, [3, 3], name='conv_02', activation=tf.nn.relu)
            global_grid = tf.reshape(global_grid, [-1, 15 * 15 * 64])

            rays = tf.cast(tf.reshape(rays, [-1, 5, 5]), tf.int32)
            rays = embedding(rays, indices=4, size=32, name='rays_embs')
            rays = conv_layer_2d(rays, 32, [3, 3], name='conv_31', activation=tf.nn.relu)
            rays = conv_layer_2d(rays, 64, [3, 3], name='conv_32', activation=tf.nn.relu)
            rays = tf.reshape(rays, [-1, 5 * 5 * 64])

            coins = tf.reshape(coins, [-1, 14, 2])
            coins, _ = transformer(coins, n_head=4, hidden_size=1024, mask_value=99, with_embeddings=True,
                                   name='transformer_coins', pooling='max')
            coins = tf.reshape(coins, [-1, 1024])

        obstacles = tf.reshape(obstacles, [-1, 7, 3])
        obstacles = linear(obstacles, 1024, name='embs_obs', activation=tf.nn.relu)
        obstacles, _ = transformer(obstacles, n_head=4, hidden_size=1024, mask_value=99, with_embeddings=False,
                                   name='transformer_global')
        obstacles = tf.math.reduce_max(obstacles, axis=2)
        obstacles = tf.reshape(obstacles, [-1, 1024])

        if with_circular:
            global_state = tf.concat([global_grid, rays], axis=1)
        else:
            global_state = tf.concat([global_state, obstacles], axis=1)

    else:
        # agent, goal, rays, obs = tf.split(global_state, [4, 3, 12, 21], axis=1)
        # Jump
        agent, goal, grid, rays = tf.split(global_state, [2, 5, 25, 12], axis=1)

        # points = tf.reshape(points, [-1, 1024])
        grid = tf.cast(tf.reshape(grid, [-1, 5, 5]), tf.int32)
        grid = embedding(grid, indices=4, size=32, name='global_embs')
        grid = conv_layer_2d(grid, 32, [3, 3], name='conv_01', activation=tf.nn.relu)
        grid = conv_layer_2d(grid, 64, [3, 3], name='conv_02', activation=tf.nn.relu)
        grid = tf.reshape(grid, [-1, 5 * 5 * 64])

        agent = linear(agent, 1024, name='agent_embs', activation=tf.nn.relu)

        global_state = tf.concat([agent, grid], axis=1)

        global_state = linear(global_state, 1024, name='embs', activation=tf.nn.relu)

    return global_state