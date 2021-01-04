import tensorflow as tf
import random
import numpy as np
from math import sqrt
from tensorforce import util
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

eps = 1e-5

class PPO:
    # PPO agent
    def __init__(self, sess, p_lr=5e-6, v_lr=5e-4, batch_fraction=0.33, num_itr=20, v_num_itr=10, action_size=19,
                 epsilon=0.2, c1=0.5, c2=0.01, discount=0.99, lmbda=1.0, name='ppo', memory=10, **kwargs):

        # Model parameters
        self.sess = sess
        self.p_lr = p_lr
        self.v_lr = v_lr
        self.batch_fraction = batch_fraction
        self.num_itr = num_itr
        self.v_num_itr = v_num_itr
        self.name = name
        self.action_size = action_size

        # PPO hyper-parameters
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.discount = discount
        self.lmbda = lmbda

        self.buffer = dict()
        self.clear_buffer()
        self.memory = memory

        # Create the network
        with tf.compat.v1.variable_scope(name) as vs:
            # Input spefication (for DeepCrawl)
            self.global_state = tf.compat.v1.placeholder(tf.float32, [None, 10, 10, 52], name='global_state')
            self.local_state = tf.compat.v1.placeholder(tf.float32, [None, 5, 5, 52], name='local_state')
            self.local_two_state = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 52], name='local_two_state')
            self.agent_stats = tf.compat.v1.placeholder(tf.int32, [None, 16], name='agent_stats')
            self.target_stats = tf.compat.v1.placeholder(tf.int32, [None, 15], name='target_stats')
            self.previous_acts = tf.compat.v1.placeholder(tf.float32, [None, self.action_size], name='previous_acts')

            # Actor network
            with tf.compat.v1.variable_scope('actor'):
                # Previous prob, for training
                self.old_logprob = tf.compat.v1.placeholder(tf.float32, [None,], name='old_prob')
                self.baseline_values = tf.compat.v1.placeholder(tf.float32, [None,], name='baseline_values')
                self.reward = tf.compat.v1.placeholder(tf.float32, [None, ], name='rewards')

                # Network specification
                self.conv_network = self.conv_net(self.global_state, self.local_state, self.local_two_state,
                                               self.agent_stats, self.target_stats, self.previous_acts)

                # Final p_layers
                self.p_network = self.linear(self.conv_network, 256, name='p_fc1', activation=tf.nn.relu)
                self.p_network = tf.concat([self.p_network, self.previous_acts], axis=1)
                self.p_network = self.linear(self.p_network, 256, name='p_fc2', activation=tf.nn.relu)

                # Probability distribution
                self.probs = self.linear(self.p_network, action_size, activation=tf.nn.softmax, name='probs')
                # Distribution to sample
                self.dist = tf.compat.v1.distributions.Categorical(probs=self.probs)

                # Sample action
                self.action = self.dist.sample()
                self.log_prob = self.dist.log_prob(self.action)

                # Get probability of a given action - useful for training
                with tf.compat.v1.variable_scope('eval_with_action'):
                    self.eval_action = tf.compat.v1.placeholder(tf.int32, [None,], name='eval_action')
                    self.log_prob_with_action = self.dist.log_prob(self.eval_action)

            # Critic network
            with tf.compat.v1.variable_scope('critic'):

                # V Network specification
                #self.v_network = self.conv_net(self.global_state, self.local_state, self.local_two_state,
                #                               self.agent_stats, self.target_stats, self.previous_acts)

                # Final p_layers
                self.v_network = self.linear(self.conv_network, 256, name='v_fc1', activation=tf.nn.relu)
                self.v_network = self.linear(self.v_network, 256, name='v_fc2', activation=tf.nn.relu)

                # Value function
                self.value = tf.squeeze(self.linear(self.p_network, 1))

            # Advantage
            # Advantage (reward - baseline)
            self.advantage = self.reward - self.baseline_values

            # L_clip loss
            self.ratio = tf.exp(self.log_prob_with_action - self.old_logprob)
            self.surr1 = self.ratio * self.advantage
            self.surr2 = tf.clip_by_value(self.ratio, 1 - self.epsilon, 1 + self.epsilon) * self.advantage
            self.clip_loss = tf.minimum(self.surr1, self.surr2)

            # Value function loss
            self.mse_loss = tf.compat.v1.squared_difference(self.reward, self.value)

            # Entropy bonus
            self.entr_loss = self.dist.entropy()

            # Total loss
            self.total_loss = - tf.reduce_mean(self.clip_loss - self.c1*self.mse_loss + self.c2*self.entr_loss)

            # Policy Optimizer
            self.p_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.p_lr).minimize(self.total_loss)
            # Value Optimizer
            # self.v_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.v_lr).minimize(self.mse_loss)

    ## Layers
    def linear(self, inp, inner_size, name='linear', bias=True, activation=None, init=None):
        with tf.compat.v1.variable_scope(name):
            lin = tf.compat.v1.layers.dense(inp, inner_size, name=name, activation=activation, use_bias=bias,
                                            kernel_initializer=init)
            return lin

    def conv_layer_2d(self, input, filters, kernel_size, strides=(1, 1), padding="SAME", name='conv',
                      activation=None,
                      bias=True):
        with tf.compat.v1.variable_scope(name):
            conv = tf.compat.v1.layers.conv2d(input, filters, kernel_size, strides, padding=padding, name=name,
                                              activation=activation, use_bias=bias)
            return conv

    def embedding(self, input, indices, size, name='embs'):
        with tf.compat.v1.variable_scope(name):
            shape = (indices, size)
            stddev = min(0.1, sqrt(2.0 / (util.product(xs=shape[:-1]) + shape[-1])))
            initializer = tf.random.normal(shape=shape, stddev=stddev, dtype=tf.float32)
            W = tf.Variable(
                initial_value=initializer, trainable=True, validate_shape=True, name='W',
                dtype=tf.float32, shape=shape
            )
            return tf.nn.tanh(tf.compat.v1.nn.embedding_lookup(params=W, ids=input, max_norm=None))

    # Convolutional network, the same for both the networks
    def conv_net(self, global_state, local_state, local_two_state, agent_stats, target_stats, prev_action):
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
        flat_41 = self.linear(embs_41, 256, name='fc_41', activation=tf.nn.relu)

        embs_51 = self.embedding(target_stats, 125, 256, name='embs_51')
        embs_51 = tf.reshape(embs_51, [-1, 15 * 256])
        flat_51 = self.linear(embs_51, 256, name='fc_51', activation=tf.nn.relu)

        all_flat = tf.concat([flat_11, flat_21, flat_31, flat_41, flat_51], axis=1)

        return all_flat

    # Train loop
    def train(self):
        losses = []
        v_losses = []

        # Before training, compute discounted reward
        # Compute GAE for rewards. If lambda == 1, they are discoutned rewards
        # Compute values for each state
        states = self.obs_to_state(self.buffer['states'])
        feed_dict = self.create_state_feed_dict(states)
        v_values = self.sess.run(self.value, feed_dict=feed_dict)
        v_values = np.append(v_values, 0)
        discounted_rewards = self.compute_gae(v_values)

        batch_size = int(len(self.buffer['states']) * self.batch_fraction)

        # Train the policy
        for it in range(self.num_itr):

            # Take a mini-batch of batch_size experience
            mini_batch_idxs = random.sample(range(len(self.buffer['states'])), batch_size)

            states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
            actions_mini_batch = [self.buffer['actions'][id] for id in mini_batch_idxs]
            old_probs_mini_batch = [self.buffer['old_probs'][id] for id in mini_batch_idxs]
            rewards_mini_batch = [discounted_rewards[id] for id in mini_batch_idxs]

            # Get DeepCrawl state
            # Convert the observation to states
            states = self.obs_to_state(states_mini_batch)
            feed_dict = self.create_state_feed_dict(states)

            # Get the baseline values
            v_values_mini_batch = self.sess.run(self.value, feed_dict=feed_dict)

            # Reshape problem, why?
            rewards_mini_batch = np.reshape(rewards_mini_batch, [-1, ])
            old_probs_mini_batch = np.reshape(old_probs_mini_batch, [-1, ])

            # Update feed dict for training
            feed_dict[self.reward] = rewards_mini_batch
            feed_dict[self.old_logprob] = old_probs_mini_batch
            feed_dict[self.eval_action] = actions_mini_batch
            feed_dict[self.baseline_values] = v_values_mini_batch

            loss, step = self.sess.run([self.total_loss, self.p_step], feed_dict=feed_dict)

            losses.append(loss)

        # # Train the value function
        # for it in range(self.v_num_itr):
        #     # Take a mini-batch of batch_size experience
        #     mini_batch_idxs = random.sample(range(len(self.buffer['states'])), batch_size)
        #
        #     states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
        #     rewards_mini_batch = [discounted_rewards[id] for id in mini_batch_idxs]
        #     # Reshape problem, why?
        #     rewards_mini_batch = np.reshape(rewards_mini_batch, [-1, ])
        #
        #     # Get DeepCrawl state
        #     # Convert the observation to states
        #     states = self.obs_to_state(states_mini_batch)
        #
        #     feed_dict = self.create_state_feed_dict(states)
        #
        #     # Update feed dict for training
        #     feed_dict[self.reward] = rewards_mini_batch
        #     v_loss, step = self.sess.run([self.mse_loss, self.v_step], feed_dict=feed_dict)
        #     v_losses.append(v_loss)

        return np.mean(losses)

    # Eval sampling the action (done by the net)
    def eval(self, state):

        state = self.obs_to_state(state)
        feed_dict = self.create_state_feed_dict(state)

        action, logprob, probs = self.sess.run([self.action, self.log_prob, self.probs], feed_dict=feed_dict)

        return action, logprob, probs

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
        global_batch = np.stack([np.asarray(state['global_in']) for state in obs])
        local_batch = np.stack([np.asarray(state['local_in']) for state in obs])
        local_two_batch = np.stack([np.asarray(state['local_in_two']) for state in obs])
        agent_stats_batch = np.stack([np.asarray(state['agent_stats']) for state in obs])
        target_stats_batch = np.stack([np.asarray(state['target_stats']) for state in obs])
        prev_act_batch = np.stack([np.asarray(state['prev_action']) for state in obs])

        return global_batch, local_batch, local_two_batch, agent_stats_batch, target_stats_batch, prev_act_batch

    # Create a state feed_dict from states
    def create_state_feed_dict(self, states):
        all_global = states[0]
        all_local = states[1]
        all_local_two = states[2]
        all_agent_stats = states[3]
        all_target_stats = states[4]
        all_prev_acts = states[5]

        feed_dict = {
            self.global_state: all_global,
            self.local_state: all_local,
            self.local_two_state: all_local_two,
            self.agent_stats: all_agent_stats,
            self.target_stats: all_target_stats,
            self.previous_acts: all_prev_acts
        }

        return feed_dict

    # Clear the memory buffer
    def clear_buffer(self):

        self.buffer['episode_lengths'] = []
        self.buffer['states'] = []
        self.buffer['actions'] = []
        self.buffer['old_probs'] = []
        self.buffer['states_n'] = []
        self.buffer['rewards'] = []
        self.buffer['terminals'] = []

    # Add a transition to the buffer
    def add_to_buffer(self, state, state_n, action, reward, old_prob, terminals):

        # If we store more than memory episodes, remove the last episode
        if len(self.buffer['episode_lengths']) + 1 >= self.memory + 1:
            idxs_to_remove = self.buffer['episode_lengths'][0]
            del self.buffer['states'][:idxs_to_remove]
            del self.buffer['actions'][:idxs_to_remove]
            del self.buffer['old_probs'][:idxs_to_remove]
            del self.buffer['states_n'][:idxs_to_remove]
            del self.buffer['rewards'][:idxs_to_remove]
            del self.buffer['terminals'][:idxs_to_remove]
            del self.buffer['episode_lengths'][0]

        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['old_probs'].append(old_prob)
        self.buffer['states_n'].append(state_n)
        self.buffer['rewards'].append(reward)
        self.buffer['terminals'].append(terminals)
        # If its terminal, update the episode length count (all states - sum(previous episode lengths)
        if terminals:
            self.buffer['episode_lengths'].append(int(len(self.buffer['states']) - np.sum(self.buffer['episode_lengths'])))


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
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + eps)

        return rewards

