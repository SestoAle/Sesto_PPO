import tensorflow as tf
import numpy as np
from layers.layers import *

from utils import DynamicRunningStat, LimitedRunningStat, RunningStat
import random


eps = 1e-12

class RND:
    # Random Network Distillation class
    def __init__(self, sess, input_spec, network_spec, obs_to_state, lr=1e-4, buffer_size=1e4, batch_size = 128,
                 num_itr = 3, name='rnd', **kwargs):

        # Used to normalize the intrinsic reward due to arbitrary scale
        self.r_norm = RunningStat()

        # The tensorflow session
        self.sess = sess

        # Model hyperparameters
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_itr = num_itr
        # Functions that define input and network specifications
        self.input_spec = input_spec
        self.network_spec = network_spec
        self.obs_to_state = obs_to_state

        # Buffer of experience
        self.buffer = []

        with tf.compat.v1.variable_scope(name) as vs:
            # Input placeholders, they depend on DeepCrawl
            self.inputs = self.input_spec()

            # For fixed target labels, use a placeholder in order to NOT update the target network
            self.target_labels = tf.compat.v1.placeholder(tf.float32, [None, 1], name='target_labels')

            # Target network, it must remain fixed during all the training
            with tf.compat.v1.variable_scope('target'):
                # Network specification from external function
                self.target = self.network_spec(self.inputs)
                # Latent space
                # TODO: move this to network specification
                self.target = linear(self.target, 32, name='latent_1', activation=tf.nn.relu)
                self.target = linear(self.target, 1, name='out')

            # Predictor network
            with tf.compat.v1.variable_scope('predictor'):
                # Network specification from external function
                self.predictor = self.network_spec(self.inputs)
                # Latent space
                # TODO: move this to network specification
                self.predictor = linear(self.predictor, 32, name='latent_1', activation=tf.nn.relu)
                self.predictor = linear(self.predictor, 1, name='out')


            self.reward_loss = tf.compat.v1.losses.mean_squared_error(self.target_labels, self.predictor)
            self.rewards = tf.math.squared_difference(self.target_labels, self.predictor)

            optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.reward_loss))
            gradients, _ = tf.compat.v1.clip_by_global_norm(gradients, 1.0)
            self.step = optimizer.apply_gradients(zip(gradients, variables))

        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

    # Fit function
    def train(self):
        losses = []

        for it in range(self.num_itr):

            # Take a mini-batch of batch_size experience
            mini_batch_idxs = random.sample(range(len(self.buffer)), self.batch_size)

            mini_batch = [self.buffer[id] for id in mini_batch_idxs]

            # Convert the observation to states
            states = self.obs_to_state(mini_batch)

            # Create the feed dict for the target network
            feed_target = self.create_state_feed_dict(states)

            # Get the target prediction (without training it)
            target_labels = self.sess.run([self.target], feed_target)[0]

            # Get the predictor estimation
            feed_predictor = self.create_state_feed_dict(states)
            feed_predictor[self.target_labels] = target_labels

            # Update the predictor networks
            loss, step, rews = self.sess.run([self.reward_loss, self.step, self.rewards], feed_predictor)

            losses.append(loss)

        # # Update the normalization statistics
        # for it in range(self.num_itr):
        #     # Take a mini-batch of batch_size experience
        #     mini_batch_idxs = random.sample(range(len(self.buffer)), self.batch_size)
        #
        #     mini_batch = [self.buffer[id] for id in mini_batch_idxs]
        #
        #     # Convert the observation to states
        #     states = self.obs_to_state(mini_batch)
        #
        #     # Create the feed dict for the target network
        #     feed_target = self.create_state_feed_dict(states)
        #
        #     # Get the target prediction (without training it)
        #     target_labels = self.sess.run([self.target], feed_target)[0]
        #
        #     # Get the predictor estimation
        #     feed_predictor = self.create_state_feed_dict(states)
        #     feed_predictor[self.target_labels] = target_labels
        #
        #     # Update the predictor networks
        #     rewards = self.sess.run([self.rewards], feed_predictor)
        #     rewards = np.squeeze(rewards)
        #     self.push_reward(rewards)

        # Update the normalization statistics

        # # Take a mini-batch of batch_size experience
        # mini_batch_idxs = random.sample(range(len(self.buffer)), len(self.buffer))
        #
        # mini_batch = [self.buffer[id] for id in mini_batch_idxs]
        #
        # # Convert the observation to states
        # states = self.obs_to_state(mini_batch)
        #
        # # Create the feed dict for the target network
        # feed_target = self.create_state_feed_dict(states)
        #
        # # Get the target prediction (without training it)
        # target_labels = self.sess.run([self.target], feed_target)[0]
        #
        # # Get the predictor estimation
        # feed_predictor = self.create_state_feed_dict(states)
        # feed_predictor[self.target_labels] = target_labels
        #
        # # Update the predictor networks
        # rewards = self.sess.run([self.rewards], feed_predictor)
        # rewards = np.squeeze(rewards)
        # self.push_reward(rewards)
        #
        # Update normalization statistics
        # Update Dynamic Running Stat
        if isinstance(self.r_norm, DynamicRunningStat):
            self.r_norm.reset()

        self.buffer = []

        # Return the mean losses of all the iterations
        return np.mean(losses)

    # Eval function
    def eval(self, obs):
        # Convert the observation to states
        states = self.obs_to_state(obs)

        # Create the feed dict for the target network
        feed_target = self.create_state_feed_dict(states)

        # Get the target prediction (without training it)
        target_labels = self.sess.run([self.target], feed_target)[0]

        # Get the predictor estimation
        feed_predictor = self.create_state_feed_dict(states)
        feed_predictor[self.target_labels] = target_labels

        # Compute the MSE to use as reward (after normalization)
        # Update the predictor networks
        rewards = self.sess.run([self.reward_loss], feed_predictor)

        # norm_rewards = self.normalize_rewards(rewards)
        # if norm_rewards[0] is None:
        #     rewards = norm_rewards
        if not isinstance(self.r_norm, DynamicRunningStat):
            self.r_norm.push(rewards[0])

        return rewards[0]

    # Create a state feed_dict from states
    def create_state_feed_dict(self, states):

        feed_dict = {}
        for i in range(len(states)):
            feed_dict[self.inputs[i]] = states[i]

        return feed_dict

    # Normalize the reward for each frame of the sequence.
    def push_reward(self, rewards):
        for r in rewards:
            self.r_norm.push(r)

    def normalize_rewards(self, rewards):
        rewards -= self.r_norm.mean
        rewards /= (self.r_norm.std + 1e-12)
        rewards *= 1

        return rewards

    def add_to_buffer(self, obs):

        if len(self.buffer) >= self.buffer_size:
            del self.buffer[0]

        self.buffer.append(obs)

    # Save the entire model
    def save_model(self, name=None, folder='saved'):
        tf.compat.v1.disable_eager_execution()
        self.saver.save(self.sess, '{}/{}_rnd'.format(folder, name))

        return

    # Load entire model
    def load_model(self, name=None, folder='saved'):
        # self.saver = tf.compat.v1.train.import_meta_graph('{}/{}.meta'.format(folder, name))
        tf.compat.v1.disable_eager_execution()
        self.saver.restore(self.sess, '{}/{}_rnd'.format(folder, name))

        print('RND loaded correctly!')
        return
