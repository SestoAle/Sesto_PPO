import tensorflow as tf
from layers.layers import *

def input_spec():
    input_length = 1359
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    return [global_state]

def obs_to_state(obs):
    global_batch = np.stack([np.asarray(state['global_in']) for state in obs])
    return [global_batch]

def network_spec(states):
    input_length = 71
    with_circular = False

    global_state = states[0]

    # agent, goal, rays, obs = tf.split(global_state, [4, 3, 12, 21], axis=1)
    # Jump
    agent_plane_x, agent_plane_z, agent_jump, is_grounded, can_double_jump, target_distances, goal, threedgrid, rotation, rays, \
    inventory = \
        tf.split(global_state, [1, 1, 1, 1, 1, 3, 2, 1331, 4, 12, 2], axis=1)

    agent_plane_x = ((agent_plane_x + 1) / 2) * 220
    agent_plane_x = tf.cast(agent_plane_x, tf.int32)

    agent_plane_z = ((agent_plane_z + 1) / 2) * 280
    agent_plane_z = tf.cast(agent_plane_z, tf.int32)

    agent_jump = ((agent_jump + 1) / 2) * 40
    agent_jump = tf.cast(agent_jump, tf.int32)

    agent = tf.concat([agent_plane_x, agent_plane_z, agent_jump], axis=1)

    agent = embedding(agent, indices=280, size=32, name='agent_embs')
    agent = tf.reshape(agent, (-1, 3 * 32))
    agent = tf.concat([agent, is_grounded, can_double_jump], axis=1)
    agent = linear(agent, 1024, name='global_embs', activation=tf.nn.tanh)

    threedgrid = tf.cast(tf.reshape(threedgrid, [-1, 11, 11, 11]), tf.int32)
    threedgrid = embedding(threedgrid, indices=7, size=8, name='global_embs')
    threedgrid = conv_layer_3d(threedgrid, 32, [3, 3, 3], strides=(2, 2, 2), name='conv_01', activation=tf.nn.tanh)
    threedgrid = conv_layer_3d(threedgrid, 64, [3, 3, 3], strides=(2, 2, 2), name='conv_02', activation=tf.nn.tanh)
    threedgrid = tf.reshape(threedgrid, [-1, 3 * 3 * 3 * 64])


    # target_distances = linear(target_distances, 1024, name='target_distances_emb', activation=tf.nn.tanh)

    # inventory = linear(inventory, 1024, name='inventory_embs', activation=tf.nn.tanh)

    global_state = tf.concat([agent, threedgrid], axis=1)

    global_state = linear(global_state, 1024, name='embs', activation=tf.nn.tanh)

    return global_state

def obs_to_state_rnd(obs):
    global_batch = np.stack([state['global_in'] for state in obs])
    return [global_batch]

def network_spec_rnd(states):
    input_length = 1359
    with_circular = False

    global_state = states[0]

    # agent, goal, rays, obs = tf.split(global_state, [4, 3, 12, 21], axis=1)
    # Jump
    agent_plane_x, agent_plane_z, agent_jump, is_grounded, can_double_jump, target_distances, goal, threedgrid, rotation, rays, \
    inventory = \
        tf.split(global_state, [1, 1, 1, 1, 1, 3, 2, 1331, 4, 12, 2], axis=1)

    agent_plane_x = ((agent_plane_x + 1) / 2) * 220
    agent_plane_x = tf.cast(agent_plane_x, tf.int32)

    agent_plane_z = ((agent_plane_z + 1) / 2) * 280
    agent_plane_z = tf.cast(agent_plane_z, tf.int32)

    agent_jump = ((agent_jump + 1) / 2) * 40
    agent_jump = tf.cast(agent_jump, tf.int32)

    agent = tf.concat([agent_plane_x, agent_plane_z, agent_jump], axis=1)

    global_state = agent

    global_state = embedding(global_state, indices=280, size=32, name='embs')
    global_state = tf.reshape(global_state, (-1, 3 * 32))
    global_state = linear(global_state, 64, name='global_embs', activation=tf.nn.relu)

    inventory = linear(inventory, 32, name='inventory_embs', activation=tf.nn.tanh)
    inventory = linear(inventory, 64, name='inventory_latent', activation=tf.nn.relu)

    global_state = tf.concat([global_state, inventory], axis=1)

    global_state = linear(global_state, 512, name='latent_1', activation=tf.nn.relu,
                          init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
                                                                          dtype=tf.dtypes.float32)
                          )
    global_state = linear(global_state, 64, name='latent_2',
                          init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
                                                                          dtype=tf.dtypes.float32)
                          )


    return global_state



def input_spec_irl():
    input_length = 1359
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    global_state_n = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state_n')

    act = tf.compat.v1.placeholder(tf.int32, [None, 1], name='act')

    return [[global_state], act, [global_state_n]]

def obs_to_state_irl(obs):
    global_batch = np.stack([state['global_in'] for state in obs])
    return [global_batch]

def network_spec_irl(states, states_n, act, with_action, actions_size):

    global_state = states[0]
    global_state_n = states_n[0]
    action_state = tf.cast(act, tf.int32)

    # Jump
    agent_plane_x, agent_plane_z, agent_jump, is_grounded, can_double_jump, target_distances, goal, threedgrid, rotation, rays, \
    inventory = \
        tf.split(global_state, [1, 1, 1, 1, 1, 3, 2, 1331, 4, 12, 2], axis=1)

    agent_plane_x = ((agent_plane_x + 1) / 2) * 220
    agent_plane_x = tf.cast(agent_plane_x, tf.int32)

    agent_plane_z = ((agent_plane_z + 1) / 2) * 280
    agent_plane_z = tf.cast(agent_plane_z, tf.int32)

    agent_jump = ((agent_jump + 1) / 2) * 40
    agent_jump = tf.cast(agent_jump, tf.int32)

    agent = tf.concat([agent_plane_x, agent_plane_z, agent_jump], axis=1)
    global_state = agent

    # agent_n, goal_n, grid_n, vertical_grid_n, rays_n, inventory_n = tf.split(global_state_n, [4, 6, 49, 81, 12, 2], axis=1)
    #
    #
    # agent_n = tf.cast(agent_n, tf.int32)
    # global_state_n = agent_n

    global_state = embedding(global_state, indices=280, size=32, name='embs')
    global_state = tf.reshape(global_state, (-1, 3*32))
    global_state = linear(global_state, 64, name='latent_1', activation=tf.nn.relu,
                          init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
                                                                          dtype=tf.dtypes.float32)
                          )
    # threedgrid = tf.cast(tf.reshape(threedgrid, [-1, 11, 11, 11]), tf.int32)
    # threedgrid = embedding(threedgrid, indices=7, size=8, name='global_embs')
    # threedgrid = conv_layer_3d(threedgrid, 32, [3, 3, 3], strides=(2, 2, 2), name='conv_01', activation=tf.nn.tanh)
    # threedgrid = conv_layer_3d(threedgrid, 64, [3, 3, 3], strides=(2, 2, 2), name='conv_02', activation=tf.nn.tanh)
    # threedgrid = tf.reshape(threedgrid, [-1, 3 * 3 * 3 * 64])
    # global_state = tf.compat.v1.layers.dropout(global_state, rate=0.2)

    global_state_n = embedding(global_state_n, indices=280, size=32, name='embs')
    global_state_n = tf.reshape(global_state_n, (-1, 2 * 32))
    global_state_n = linear(global_state_n, 64, name='latent_1_n', activation=tf.nn.relu,
                          init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
                                                                          dtype=tf.dtypes.float32)
                         )

    # action_state = embedding(action_state, indices=10, size=32, name='action_embs')
    # action_state = tf.reshape(action_state, [-1, 32])
    # action_state = linear(action_state, 64, name='latent_action_n', activation=tf.nn.relu,
    #                       init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
    #                                                                       dtype=tf.dtypes.float32)
    #                      )
    # action_state = tf.compat.v1.layers.dropout(action_state, rate=0.2)

    # inventory = linear(inventory, 32, name='inventory_embs', activation=tf.nn.tanh)
    # inventory = linear(inventory, 64, name='latent_inventory_n', activation=tf.nn.relu,
    #                       init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
    #                                                                       dtype=tf.dtypes.float32)
    #                       )

    encoded = tf.concat([global_state, global_state_n], axis=1)

    global_state = linear(encoded, 512, name='latent_2', activation=tf.nn.relu,
                          init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
                                                                          dtype=tf.dtypes.float32)
                         )
    # global_state = tf.compat.v1.layers.dropout(global_state, rate=0.2)

    global_state = linear(global_state, 1, name='out',
                          init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
                                                                          dtype=tf.dtypes.float32)
                          )
    # global_state = tf.compat.v1.layers.dropout(global_state, rate=0.2)



    return global_state, encoded