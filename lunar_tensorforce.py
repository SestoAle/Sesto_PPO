import tensorforce
import gym
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.environments import Environment
import numpy as np

def callback(r,p):
    episode_rewards = r.episode_rewards
    if len(episode_rewards) % 100 == 0:
        print('Mean episode reward of the last 100 episodes at episode {}: {}'.format(len(episode_rewards), np.mean(episode_rewards[-100:])))

class LunarEnvironment(Environment):
    def __init__(self, with_graphics = False, max_episode_timesteps=400, name='LunarLanderContinuous-v2'):
        self.env = gym.make(name)
        self.with_graphics = with_graphics
        self._max_episode_timesteps=max_episode_timesteps

    def reset(self):
        if self.with_graphics:
            self.env.render('human')
        state = self.env.reset()
        return dict(global_in=state)

    def execute(self, actions):
        if self.with_graphics:
            self.env.render('human')
        state, reward, done, _ = self.env.step(actions)
        state = dict(global_in=state)
        return state, done, reward

    def set_config(self, config):
        return

    def entropy(self, probs):
        return 0

    def states(self):
        return {
            'global_in': {'shape': (8), 'type': 'float'},
        }

    def actions(self):
        return {
            'type': 'float',
            'shape': (2),
            'min_value': -1.0,
            'max_value': 1.0
        }

if __name__ == "__main__":

    agent = Agent.create(
        # Agent type
        agent='ppo',
        # Inputs structure
        states={
            'global_in': {'shape': (8), 'type': 'float'},
        },
        # Actions structure
        actions={
            'type': 'float',
            'shape': (2),
            'min_value': -1.0,
            'max_value': 1.0
        },
        network=[
            dict(type='retrieve', tensors=['global_in']),
            dict(type='dense', size=256, activation='relu'),
            dict(type='dense', size=256, activation='relu')
        ],
        # MemoryModel

        # 10 episodes per update
        batch_size=5,
        # Every 10 episodes
        update_frequency=5,
        max_episode_timesteps=400,

        # DistributionModel

        discount=0.99,
        entropy_regularization=0.01,
        likelihood_ratio_clipping=0.2,

        critic_network=[
            dict(type='retrieve', tensors=['global_in']),
            dict(type='dense', size=256, activation='relu'),
            dict(type='dense', size=256, activation='relu')
        ],

        critic_optimizer=dict(
            type='multi_step',
            optimizer=dict(
                type='subsampling_step',
                fraction=0.33,
                optimizer=dict(
                    type='adam',
                    learning_rate=5e-3
                )
            ),
            num_steps=10
        ),

        # PPOAgent

        learning_rate=1e-5,

        subsampling_fraction=0.33,
        optimization_steps=20,
        # TensorFlow etc
        name='agent', device=None, parallel_interactions=1, seed=None, saver=None, #tracking=['distribution'],
        summarizer=None, recorder=None
    )

    env = LunarEnvironment()
    runner = Runner(agent, env, 400)
    total_episode = 30000
    runner.run(total_episode, callback=callback, use_tqdm=False)


    # ep_rewards = []
    # for e in range(1, int(total_episode)):
    #     state = env.reset()
    #     ep_reward = 0
    #     done = False
    #     step = 0
    #     while not done:
    #         actions = agent.act(state)
    #         state_n, done, reward = env.execute(actions)
    #         step += 1
    #         ep_reward += reward
    #         if step % 400 == 0:
    #             done = True
    #         agent.observe(reward, done)
    #         state=state_n
    #     ep_rewards.append(ep_reward)
    #     if e % 100 == 0:
    #         print('Mean episode reward of the last 100 episodes at episode {}: {}'.format(e, np.mean(ep_rewards[-100:])))



