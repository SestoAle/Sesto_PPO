import tensorflow as tf
import tensorflow_probability as tfp
import random
import numpy as np
from math import sqrt
import utils

import os

eps = 1e-5

# Actor-Critic PPO. The Actor is independent by the Critic.
class SAC:
    # PPO agent
    def __init__(self, sess, p_lr=0.0003, v_lr=0.0003, batch_fraction=32, p_num_itr=20, v_num_itr=20, action_size=3,
                 epsilon=0.2, c1=0.5, c2=0.01, discount=0.99, lmbda=1.0, name='ppo', memory=10, norm_reward=False,

                 alpha = 0.2, q_lr=0.0003,

                 model_name='agent',

                 # LSTM
                 recurrent = False, recurrent_length = 5,

                 **kwargs):

        # Model parameters
        self.sess = sess
        self.p_lr = p_lr
        self.v_lr = v_lr
        self.batch_fraction = batch_fraction
        self.p_num_itr = p_num_itr
        self.v_num_itr = v_num_itr
        self.name = name
        self.action_size = action_size
        self.norm_reward = norm_reward
        self.model_name = model_name

        # PPO hyper-parameters
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.discount = discount
        self.lmbda = lmbda

        # SAC hyper-parameters
        self.alpha = alpha
        self.q_lr = q_lr

        # Recurrent paramtere
        self.recurrent = recurrent
        self.recurrent_length = recurrent_length
        self.recurrent_size = 256

        self.buffer = dict()
        self.clear_buffer()
        self.memory = memory
        # Create the network
        with tf.compat.v1.variable_scope(name) as vs:
            # Input spefication (for DeepCrawl)
            # self.global_state = tf.compat.v1.placeholder(tf.float32, [None, 10, 10, 53], name='global_state')
            # self.local_state = tf.compat.v1.placeholder(tf.float32, [None, 5, 5, 53], name='local_state')
            # self.local_two_state = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 53], name='local_two_state')
            # self.agent_stats = tf.compat.v1.placeholder(tf.int32, [None, 16], name='agent_stats')
            # self.target_stats = tf.compat.v1.placeholder(tf.int32, [None, 15], name='target_stats')
            # self.previous_acts = tf.compat.v1.placeholder(tf.float32, [None, self.action_size], name='previous_acts')
            self.input = tf.compat.v1.placeholder(tf.float32, [None, 7*7], name='input')

            # Actor network
            with tf.compat.v1.variable_scope('policy'):
                # Previous prob, for training
                self.old_logprob = tf.compat.v1.placeholder(tf.float32, [None,], name='old_prob')
                self.baseline_values = tf.compat.v1.placeholder(tf.float32, [None,], name='baseline_values')
                self.reward = tf.compat.v1.placeholder(tf.float32, [None, ], name='rewards')

                # Network specification
                #self.conv_network = self.conv_net(self.global_state, self.local_state, self.local_two_state,
                #                               self.agent_stats, self.target_stats)

                self.conv_network = self.input

                # Final p_layers
                self.p_network = self.linear(self.conv_network, 256, name='p_fc1', activation=tf.nn.relu)
                #self.p_network = tf.concat([self.p_network, self.previous_acts], axis=1)

                if not self.recurrent:
                    self.p_network = self.linear(self.p_network, 256, name='p_fc2', activation=tf.nn.relu)
                else:
                    # The last FC layer will be replaced by an LSTM layer.
                    # Recurrent network needs more variables

                    # Get batch size and number of feature of the previous layer
                    bs, feature = utils.shape_list(self.p_network)
                    self.recurrent_train_length = tf.compat.v1.placeholder(tf.int32)
                    self.p_network = tf.reshape(self.p_network, [bs/self.recurrent_train_length, self.recurrent_train_length, feature])
                    # Define the RNN cell
                    self.rnn_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units = self.recurrent_size, state_is_tuple=True)
                    # Define state_in for the cell
                    self.state_in = self.rnn_cell.zero_state(bs, tf.float32)

                    # Apply rnn
                    self.rnn, self.rnn_state = tf.compat.v1.nn.dynamic_rnn(
                        inputs = self.p_network, cell=self.rnn_cell, dtype=tf.float32, initial_state=self.state_in
                    )
                    self.p_network = tf.reshape(self.rnn, [-1, self.recurrent_size])


                # Probability distribution
                self.probs = self.linear(self.p_network, action_size, activation=tf.nn.softmax, name='probs') + eps
                # Distribution to sample
                self.dist = tfp.distributions.Categorical(probs=self.probs, allow_nan_stats=False)

                # Sample action
                self.action = self.dist.sample(name='action')
                self.log_prob = self.dist.log_prob(self.action)

                # Get probability of a given action - useful for training
                with tf.compat.v1.variable_scope('eval_with_action'):
                    self.eval_action = tf.compat.v1.placeholder(tf.int32, [None,], name='eval_action')
                    self.log_prob_with_action = self.dist.log_prob(self.eval_action)

            self.action_idxs = tf.compat.v1.placeholder(tf.int32, [None, 2], name='action_idxs')
            # Q1 network
            with tf.compat.v1.variable_scope('q1'):
                self.q1 = self.linear(self.input, 256, name='q1_fc1', activation=tf.nn.relu)
                self.q1 = self.linear(self.q1, 256, name='q1_fc2', activation=tf.nn.relu)
                self.q1 = self.linear(self.q1, self.action_size, name='q1_values')
            self.q1_action = tf.gather_nd(self.q1, self.action_idxs)

            # Q2 network
            with tf.compat.v1.variable_scope('q2'):
                self.q2 = self.linear(self.input, 256, name='q2_fc1', activation=tf.nn.relu)
                self.q2 = self.linear(self.q2, 256, name='q2_fc2', activation=tf.nn.relu)
                self.q2 = self.linear(self.q2, self.action_size, name='q2_values')
            self.q2_action = tf.gather_nd(self.q2, self.action_idxs)

            # Q1 target network
            with tf.compat.v1.variable_scope('t_q1'):
                self.q1_t = self.linear(self.input, 256, name='q1_t_fc1', activation=tf.nn.relu)
                self.q1_t = self.linear(self.q1_t, 256, name='q1_t_fc2', activation=tf.nn.relu)
                self.q1_t = self.linear(self.q1_t, self.action_size, name='q1_t_values')

            # Q2 network
            with tf.compat.v1.variable_scope('t_q2'):
                self.q2_t = self.linear(self.input, 256, name='q2_t_fc1', activation=tf.nn.relu)
                self.q2_t = self.linear(self.q2_t, 256, name='q2_t_fc2', activation=tf.nn.relu)
                self.q2_t = self.linear(self.q2_t, self.action_size, name='q2_t_values')

            # Update the q-functions
            self.targets = tf.compat.v1.placeholder(tf.float32, [None,], name='targets')

            self.q1_loss = tf.reduce_mean(0.5 * tf.pow(self.q1_action - self.targets, 2))
            self.q2_loss = tf.reduce_mean(0.5 * tf.pow(self.q2_action - self.targets, 2))

            # Q Optimizers
            self.q1_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.q_lr).minimize(self.q1_loss)
            self.q2_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.q_lr).minimize(self.q2_loss)

            # Update policy
            self.q1_values = tf.compat.v1.placeholder(tf.float32, [None, action_size], name='q1_values')
            self.q2_values = tf.compat.v1.placeholder(tf.float32, [None, action_size], name='q2_values')
            self.p_loss = self.alpha * tf.math.log(self.probs) - tf.minimum(self.q1_values, self.q2_values)
            self.probs_mat = tf.expand_dims(self.probs, axis=1)
            self.p_loss = - tf.reduce_mean(tf.matmul(self.probs_mat, self.p_loss))

            # Policy Optimizer
            self.p_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.p_lr).minimize(self.p_loss)

        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

    # Update target q networks with hard update
    def update_target_q_net_hard(self):
        q1_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="ppo/q1")
        q1_t_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="ppo/t_q1")
        # Hard update
        self.sess.run([v_t.assign(v) for v_t, v in zip(q1_t_vars, q1_vars)])

        q2_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="ppo/q2")
        q2_t_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="ppo/t_q2")
        # Hard update
        self.sess.run([v_t.assign(v) for v_t, v in zip(q2_t_vars, q2_vars)])

    # Update target q networks with soft update
    def update_target_q_net_soft(self, tau=0.05):
        q1_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="ppo/q1")
        q1_t_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="ppo/t_q1")

        self.sess.run([v_t.assign(v_t * (1. - tau) + v * tau) for v_t, v in zip(q1_t_vars, q1_vars)])

        q2_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="ppo/q2")
        q2_t_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="ppo/t_q2")

        self.sess.run([v_t.assign(v_t * (1. - tau) + v * tau) for v_t, v in zip(q2_t_vars, q2_vars)])

    # Compute targets for policy update
    def compute_targets(self, rewards, dones, q1_t_values, q2_t_values, probs):

        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        q1_t_values = np.asarray(q1_t_values)
        q2_t_values = np.asarray(q2_t_values)
        probs = np.squeeze(np.asarray(probs))

        q1_t_values = np.expand_dims(q1_t_values, axis=2)
        q2_t_values = np.expand_dims(q2_t_values, axis=2)

        matmul = (np.matmul(np.expand_dims(probs, axis=1) ,
                            (np.minimum(q1_t_values, q2_t_values) - self.alpha*np.log(np.expand_dims(probs, axis=2)))))
        matmul = np.reshape(matmul, [-1,])

        targets = rewards + (1-dones)*self.discount*matmul

        return targets

    ## Layers
    def linear(self, inp, inner_size, name='linear', bias=True, activation=None, init=None):
        with tf.compat.v1.variable_scope(name):
            lin = tf.compat.v1.layers.dense(inp, inner_size, name=name, activation=activation, use_bias=bias,
                                            kernel_initializer=init)
            return lin

    def conv_layer_2d(self, input, filters, kernel_size, strides=(1, 1), padding="SAME", name='conv',
                      activation=None, bias=True):

        with tf.compat.v1.variable_scope(name):
            conv = tf.compat.v1.layers.conv2d(input, filters, kernel_size, strides, padding=padding, name=name,
                                              activation=activation, use_bias=bias)
            return conv

    def embedding(self, input, indices, size, name='embs'):
        with tf.compat.v1.variable_scope(name):
            shape = (indices, size)
            stddev = min(0.1, sqrt(2.0 / (utils.product(xs=shape[:-1]) + shape[-1])))
            initializer = tf.random.normal(shape=shape, stddev=stddev, dtype=tf.float32)
            W = tf.Variable(
                initial_value=initializer, trainable=True, validate_shape=True, name='W',
                dtype=tf.float32, shape=shape
            )
            return tf.nn.tanh(tf.compat.v1.nn.embedding_lookup(params=W, ids=input, max_norm=None))

    # Convolutional network, the same for both the networks
    def conv_net(self, global_state, local_state, local_two_state, agent_stats, target_stats, baseline=False):
        conv_10 = self.conv_layer_2d(global_state, 32, [1, 1], name='conv_10', activation=tf.nn.tanh, bias=False)
        conv_11 = self.conv_layer_2d(conv_10, 32, [3, 3], name='conv_11', activation=tf.nn.relu)
        conv_12 = self.conv_layer_2d(conv_11, 64, [3, 3], name='conv_12', activation=tf.nn.relu)
        flat_11 = tf.reshape(conv_12, [-1, 10 * 10 * 64])

        conv_20 = self.conv_layer_2d(local_state, 32, [1, 1], name='conv_20', activation=tf.nn.tanh, bias=False)
        conv_21 = self.conv_layer_2d(conv_20, 32, [3, 3], name='conv_21', activation=tf.nn.relu)
        conv_22 = self.conv_layer_2d(conv_21, 64, [3, 3], name='conv_22', activation=tf.nn.relu)
        flat_21 = tf.reshape(conv_22, [-1, 5 * 5 * 64])

        conv_30 = self.conv_layer_2d(local_two_state, 32, [1, 1], name='conv_30', activation=tf.nn.tanh, bias=False)
        conv_31 = self.conv_layer_2d(conv_30, 32, [3, 3], name='conv_31', activation=tf.nn.relu)
        conv_32 = self.conv_layer_2d(conv_31, 64, [3, 3], name='conv_32', activation=tf.nn.relu)
        flat_31 = tf.reshape(conv_32, [-1, 3 * 3 * 64])

        embs_41 = tf.nn.tanh(self.embedding(agent_stats, 129, 256, name='embs_41'))
        embs_41 = tf.reshape(embs_41, [-1, 16 * 256])
        if not baseline:
            flat_41 = self.linear(embs_41, 256, name='fc_41', activation=tf.nn.relu)
        else:
            flat_41 = self.linear(embs_41, 128, name='fc_41', activation=tf.nn.relu)

        embs_51 = self.embedding(target_stats, 125, 256, name='embs_51')
        embs_51 = tf.reshape(embs_51, [-1, 15 * 256])
        if not baseline:
            flat_51 = self.linear(embs_51, 256, name='fc_51', activation=tf.nn.relu)
        else:
            flat_51 = self.linear(embs_51, 128, name='fc_51', activation=tf.nn.relu)

        all_flat = tf.concat([flat_11, flat_21, flat_31, flat_41, flat_51], axis=1)

        return all_flat


    # Train loop
    def train(self):
        losses = []
        v_losses = []

        # Get batch size based on batch_fraction
        batch_size = self.batch_fraction

        # Train the value function
        for it in range(self.p_num_itr):
            # Take a mini-batch of batch_size experience
            mini_batch_idxs = random.sample(range(len(self.buffer['states'])), batch_size)

            states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
            states_n_mini_batch = [self.buffer['states_n'][id] for id in mini_batch_idxs]
            # Convert the observation to states
            states_n = self.obs_to_state(states_n_mini_batch)
            feed_dict = self.create_state_feed_dict(states_n)

            # Compute target values with target networks
            q1_t_values, q2_t_values = self.sess.run([self.q1_t, self.q2_t], feed_dict=feed_dict)

            states = self.obs_to_state(states_mini_batch)
            feed_dict = self.create_state_feed_dict(states)
            rewards_mini_batch = [self.buffer['rewards'][id] for id in mini_batch_idxs]
            terminals_mini_batch = [self.buffer['terminals'][id] for id in mini_batch_idxs]
            actions_mini_batch = [self.buffer['actions'][id] for id in mini_batch_idxs]

            actions_mini_batch = np.asarray(actions_mini_batch)
            actions_mini_batch = [[b,a] for b,a in zip(np.arange(batch_size), actions_mini_batch)]

            _, _, probs = self.eval(states_n_mini_batch)

            targets_mini_batch = self.compute_targets(rewards_mini_batch, terminals_mini_batch, q1_t_values, q2_t_values, probs)

            feed_dict[self.targets] = targets_mini_batch
            feed_dict[self.action_idxs] = actions_mini_batch

            # Update q functions
            q1_loss, q2_loss, q1_step, q2_step = self.sess.run([self.q1_loss, self.q2_loss, self.q1_step, self.q2_step],
                                                               feed_dict=feed_dict)

            # Get new Q values
            q1_values, q2_values = self.sess.run([self.q1, self.q2], feed_dict=feed_dict)
            feed_dict[self.q1_values] = q1_values
            feed_dict[self.q2_values] = q2_values

            # Update policy
            loss, step = self.sess.run([self.p_loss, self.p_step], feed_dict=feed_dict)

            q1_values, q2_values = self.sess.run([self.q1, self.q2], feed_dict=feed_dict)

            self.update_target_q_net_soft(0.05)
            
            losses.append(loss)

        print(np.mean(losses))

        return np.mean(losses)

    # Eval sampling the action (done by the net)
    def eval(self, state):

        state = self.obs_to_state(state)
        feed_dict = self.create_state_feed_dict(state)

        action, logprob, probs = self.sess.run([self.action, self.log_prob, self.probs], feed_dict=feed_dict)

        return action, logprob, probs

    # Eval sampling the action, but with recurrent: it needs the internal hidden state
    def eval_recurrent(self, state, internal):
        state = self.obs_to_state(state)
        feed_dict = self.create_state_feed_dict(state)

        # Pass the internal state
        feed_dict[self.state_in] = internal
        feed_dict[self.recurrent_train_length] = 1
        action, logprob, probs, internal = self.sess.run([self.action, self.log_prob, self.probs, self.rnn_state], feed_dict=feed_dict)

        # Return is equal to eval(), but with the new internal state
        return action, logprob, probs, internal

    # Eval with argmax
    def eval_max(self, state):

        state = self.obs_to_state(state)
        feed_dict = self.create_state_feed_dict(state)

        probs = self.sess.run([self.probs], feed_dict=feed_dict)
        return np.argmax(probs)

    # Eval with a given action
    def eval_action(self, states, actions):

        state = self.obs_to_state(states)
        feed_dict = self.create_state_feed_dict(state)
        feed_dict[self.eval_action] = actions

        logprobs = self.sess.run([self.log_prob_with_action], feed_dict=feed_dict)[0]

        logprobs = np.reshape(logprobs, [-1, 1])

        return logprobs

    # Transform an observation to a DeepCrawl state
    def obs_to_state(self, obs):

        input_batch = np.stack([np.asarray(state['input']) for state in obs])

        return input_batch

    # Create a state feed_dict from states
    def create_state_feed_dict(self, states):
        all_input = states

        feed_dict = {
            self.input: all_input,
        }

        return feed_dict

    # Clear the memory buffer
    def clear_buffer(self):

        self.buffer['states'] = []
        self.buffer['actions'] = []
        self.buffer['old_probs'] = []
        self.buffer['states_n'] = []
        self.buffer['rewards'] = []
        self.buffer['terminals'] = []
        self.buffer['probs'] = []

    # Add a transition to the buffer
    def add_to_buffer(self, state, state_n, action, reward, old_prob, terminals, probs):

        # If we store more than memory episodes, remove the last episode
        if len(self.buffer['states']) + 1 >= self.memory + 1:
            idxs_to_remove = np.random.randint(0,self.memory)
            del self.buffer['states'][idxs_to_remove]
            del self.buffer['actions'][idxs_to_remove]
            del self.buffer['old_probs'][idxs_to_remove]
            del self.buffer['states_n'][idxs_to_remove]
            del self.buffer['rewards'][idxs_to_remove]
            del self.buffer['terminals'][idxs_to_remove]
            del self.buffer['probs'][idxs_to_remove]

        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['old_probs'].append(old_prob)
        self.buffer['states_n'].append(state_n)
        self.buffer['rewards'].append(reward)
        self.buffer['terminals'].append(terminals)
        self.buffer['probs'].append(probs)


    # Change rewards in buffer to discounted rewards
    def compute_discounted_reward(self):

        discounted_rewards = []
        discounted_reward = 0
        # The discounted reward can be computed in reverse
        for (terminal, reward) in zip(reversed(self.buffer['terminals']), reversed(self.buffer['rewards'])):
            if terminal:
                discounted_reward = 0

            discounted_reward = reward + (self.discount*discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # Normalizing reward
        if self.norm_reward:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + eps)

        return discounted_rewards

    # Change rewards in buffer to discounted rewards or GAE rewards (if lambda == 1, gae == discounted)
    def compute_gae(self, v_values):

        rewards = []
        gae = 0

        # The gae rewards can be computed in reverse
        for i in reversed(range(len(self.buffer['rewards']))):
            terminal = self.buffer['terminals'][i]
            m = 1
            if terminal:
                m = 0

            delta = self.buffer['rewards'][i] + self.discount * v_values[i + 1] * m - v_values[i]
            gae = delta + self.discount * self.lmbda * m * gae
            reward = gae + v_values[i]

            rewards.insert(0, reward)

        # Normalizing
        if self.norm_reward:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + eps)

        return rewards

    # Save the entire model
    def save_model(self, name=None, folder='saved'):
        self.saver.save(self.sess, '{}/{}'.format(folder, name))

        if True:
            graph_def = self.sess.graph.as_graph_def()

            # freeze_graph clear_devices option
            for node in graph_def.node:
                node.device = ''

            graph_def = tf.compat.v1.graph_util.remove_training_nodes(input_graph=graph_def)
            output_node_names = [
                'ppo/actor/add',
                'ppo/actor/ppo_actor_Categorical/action/Reshape_2',
                'ppo/critic/Squeeze'
            ]

            # implies tf.compat.v1.graph_util.extract_sub_graph
            graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess=self.sess, input_graph_def=graph_def,
                output_node_names=output_node_names
            )
            graph_path = tf.io.write_graph(
                graph_or_graph_def=graph_def, logdir=folder,
                name=(name + '.pb'), as_text=False
            )

        return

    # Load entire model
    def load_model(self, name=None, folder='saved'):
        #self.saver = tf.compat.v1.train.import_meta_graph('{}/{}.meta'.format(folder, name))
        self.saver.restore(self.sess, '{}/{}'.format(folder, name))

        print('Model loaded correctly!')
        return
