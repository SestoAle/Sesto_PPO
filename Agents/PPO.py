import tensorflow as tf
import random
import numpy as np

eps = 1e-5

class PPO:
    # PPO agent
    def __init__(self, sess, lr=0.0002, batch_size=2000, num_itr=4, action_size=4, epsilon=0.2, c1=0.5, c2=0.01,
                 discount=0.99, lmbda=1.0, name='ppo', **kwargs):

        # Model parameters
        self.sess = sess
        self.lr = lr
        self.batch_size = batch_size
        self.num_itr = num_itr
        self.name = name

        # PPO hyper-parameters
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.discount = discount
        self.lmbda = lmbda

        self.buffer = dict()
        self.clear_buffer()

        # Create the network
        with tf.compat.v1.variable_scope(name) as vs:
            # Input spefication (for LunarLander)
            self.state = tf.compat.v1.placeholder(tf.float32, [None, 8], name='state')

            # Critic network
            with tf.compat.v1.variable_scope('critic'):
                # Network specification
                self.reward = tf.compat.v1.placeholder(tf.float32, [None,], name='rewards')
                self.v_network = self.mlp(self.state)

                # Value function
                self.value = tf.squeeze(self.linear(self.v_network, 1))

            # Actor network
            with tf.compat.v1.variable_scope('actor'):
                # Previous prob
                self.old_logprob = tf.compat.v1.placeholder(tf.float32, [None,], name='old_prob')

                # Network specification
                self.p_network = self.mlp(self.state)

                # Probability distribution
                self.probs = self.linear(self.p_network, action_size, activation='softmax', name='probs')
                # Distribution to sample
                self.dist = tf.compat.v1.distributions.Categorical(probs=self.probs)

                # Sample action
                self.action = self.dist.sample()
                self.log_prob = self.dist.log_prob(self.action)

                # Get probability of a given action - useful for training
                with tf.compat.v1.variable_scope('eval_with_action'):
                    self.eval_action = tf.compat.v1.placeholder(tf.int32, [None,], name='eval_action')
                    self.log_prob_with_action = self.dist.log_prob(self.eval_action)

            # Advantage
            # Advantage (reward - baseline)
            self.advantage = self.reward - self.value

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

            # Optimizer
            self.step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss)

    # Layers
    def linear(self, inp, inner_size, name='linear', bias=True, activation=None, init=None):
        with tf.compat.v1.variable_scope(name):
            lin = tf.compat.v1.layers.dense(inp, inner_size, name=name, activation=activation, use_bias=bias,
                                            kernel_initializer=init)
            return lin

    # A simple MLP network
    def mlp(self, state, layers = 2, hidden_size = 64):

        for i in range(layers):
            state = self.linear(state, hidden_size, name='linear_{}'.format(i), activation='tanh')

        return state

    # Train loop
    def train(self):
        losses = []

        # Before training, compute discounted reward
        #self.compute_discounted_reward()

        for it in range(self.num_itr):
            # Compute GAE for rewards. If lambda == 1, they are discoutned rewards
            # Compute values for each state
            v_values = self.sess.run(self.value, feed_dict={
                self.state: self.buffer['states']
            })
            v_values = np.append(v_values, 0)
            discounted_rewards = self.compute_gae(v_values)

            # Take a mini-batch of batch_size experience
            mini_batch_idxs = random.sample(range(len(self.buffer['states'])), self.batch_size)

            states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
            actions_mini_batch = [self.buffer['actions'][id] for id in mini_batch_idxs]
            old_probs_mini_batch = [self.buffer['old_probs'][id] for id in mini_batch_idxs]
            rewards_mini_batch = [discounted_rewards[id] for id in mini_batch_idxs]

            # Reshape problem, why?
            rewards_mini_batch = np.reshape(rewards_mini_batch, [-1, ])
            old_probs_mini_batch = np.reshape(old_probs_mini_batch, [-1, ])

            feed_dict = {
                self.state: states_mini_batch,
                self.reward: rewards_mini_batch,
                self.old_logprob: old_probs_mini_batch,
                self.eval_action: actions_mini_batch
            }

            loss, step = self.sess.run([self.total_loss, self.step], feed_dict=feed_dict)

            losses.append(loss)

        return np.mean(losses)

    # Eval sampling the action (done by the net)
    def eval(self, state):

        feed_dict = {
            self.state: state
        }

        action, logprob, probs = self.sess.run([self.action, self.log_prob, self.probs], feed_dict=feed_dict)

        return action, logprob, probs

    # Eval with argmax
    def eval_max(self, state):

        feed_dict = {
            self.state: state
        }

        probs = self.sess.run([self.probs], feed_dict=feed_dict)
        return np.argmax(probs)

    # Eval with a given action
    def eval_action(self, states, actions):

        feed_dict = {
            self.state: states,
            self.eval_action: actions
        }

        logprobs = self.sess.run([self.log_prob_with_action], feed_dict=feed_dict)[0]

        logprobs = np.reshape(logprobs, [-1, 1])

        return logprobs


    # Clear the memory buffer
    def clear_buffer(self):
        self.buffer['states'] = []
        self.buffer['actions'] = []
        self.buffer['old_probs'] = []
        self.buffer['states_n'] = []
        self.buffer['rewards'] = []
        self.buffer['terminals'] = []

    # Add a transition to the buffer
    def add_to_buffer(self, state, state_n, action, reward, old_prob, terminals):
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['old_probs'].append(old_prob)
        self.buffer['states_n'].append(state_n)
        self.buffer['rewards'].append(reward)
        self.buffer['terminals'].append(terminals)

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

