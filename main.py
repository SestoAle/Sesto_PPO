from Agents.PPO import PPO
import tensorflow as tf
import gym


import numpy as np

class Environment:
    # Class for environment
    def __init__(self, name="LunarLander-v2", no_graphic=True, **kwargs):

        self.game_name = name
        self.env = gym.make(self.game_name)
        self.no_graphic = no_graphic

        self.max_timestep = 300


    def get_state(self, obs):
        state = np.reshape(obs, (8))
        return state

    def reset(self):
        obs = self.env.reset()
        if not self.no_graphic:
            self.env.render('human')
        state = self.get_state(obs)
        return state

    def execute(self, action):
        obs, reward, done, _ = self.env.step(action)
        if not self.no_graphic:
            self.env.render('human')
        state = self.get_state(obs)

        return state, reward, done


if __name__ == "__main__":

    # Total episode of training
    total_episode = 50000
    # Frequency of training (in episode)
    frequency = 2000
    # Frequency of logging
    logging = 100

    # Create environment
    env = Environment()

    # Create agent
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    agent = PPO(sess=sess)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # Training loop
    ep = 0
    total_step = 0
    # Cumulative rewards
    episode_rewards = []
    # Policy losses
    policy_losses = []
    # Value losses
    value_losses = []
    while ep <= total_episode:
        ep += 1
        step = 0
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:

            # Evaluation - Execute step
            action, logprob = agent.eval([state])
            action = action[0]
            state_n, reward, done = env.execute(action)
            step += 1
            total_step += 1
            episode_reward += reward

            # If step >= max_timestep, end the episode
            if step >= env.max_timestep:
                done = True

            # Update PPO memory
            agent.add_to_buffer(state, state_n, action, reward, logprob, done)
            state = state_n

            # If frequency episodes are passed, update the policy
            if total_step > 0 and total_step % frequency == 0:
                total_loss, value_loss, policy_loss = agent.train()
                agent.clear_buffer()
                value_losses.append(value_loss)
                policy_losses.append(policy_loss)

            # If done, end the episode
            if done:
                episode_rewards.append(episode_reward)
                break



        # Logging information
        if ep > 0 and ep % logging == 0:
            print('Mean of {} episode reward after {} episodes: {}'.
                  format(logging, ep, np.mean(episode_rewards[-logging:])))
            print('Policy loss {}'.
                  format(np.mean(policy_losses[-logging:])))
            print('Value loss {}'.
                  format(np.mean(value_losses[-logging:])))


    # Testing phase
    # Re-Create environment with graphics
    env = Environment(no_graphic=False)
    while True:

        state = env.reset()
        step = 0
        done = False
        episode_reward = 0

        while not done:

            # Evaluation - Execute step
            action = agent.eval([state])
            state_n, reward, done = env.execute(action)
            step += 1
            episode_reward += reward

            # If step >= max_timestep, end the episode
            if step >= env.max_timestep:
                done = True

            state = state_n

            # If done, end the episode
            if done:
                episode_rewards.append(episode_reward)
                break
