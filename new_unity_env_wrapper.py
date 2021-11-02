from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
import logging as logs
import matplotlib.pyplot as plt

class BugEnvironment:
    def __init__(self, game_name, no_graphics, worker_id, max_episode_timesteps, pos_already_normed=True):

        self.no_graphics = no_graphics
        # Channel for passing the parameters
        self.channel = EnvironmentParametersChannel()
        self.configuration_channel = EngineConfigurationChannel()
        self.unity_env = UnityEnvironment(game_name, no_graphics=no_graphics, seed=worker_id, worker_id=worker_id,
                                    side_channels=[self.channel, self.configuration_channel])
        self.behavior_name = 'BugBehavior?team=0'
        self.unity_env.reset()
        self.configuration_channel.set_configuration_parameters(time_scale=500, quality_level=0)
        self._max_episode_timesteps = max_episode_timesteps
        self.config = None
        self.actions_eps = 0.1
        self.previous_action = [0, 0]
        # Table where we save the position for intrisic reward and spawn position
        self.pos_buffer = dict()
        self.pos_already_normed = pos_already_normed
        self.r_max = 0.5
        self.max_counter = 500
        self.tau = 1 / 40
        self.standard_position = [14, 14, 1]
        self.coverage_of_points = []

        # Dict to store the trajectories at each episode
        self.trajectories_for_episode = dict()
        # Dict to store the actions at each episode
        self.actions_for_episode = dict()
        self.episode = -1

        self.dems = None

        # Defined the values to sample for goal-conditioned policy
        self.reward_weights = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    def reset(self):
        # Sample a motivation reward weight
        # self.reward_weights = self.config['reward_weights']
        # self.win_weight = self.config['win_weight']
        # self.sample_weights = self.reward_weights[np.random.randint(len(self.reward_weights))]
        # self.sample_win = self.win_weight[np.random.randint(len(self.win_weight))]
        # self.sample_weights = [self.sample_win, self.sample_weights, 1-self.sample_weights]

        if self.dems is not None:
            self.sample_position_from_dems()

        # Change config to be fed to Unity (no list)
        unity_config = dict()
        if self.config is not None:
            for key in self.config.keys():
                if key != "reward_weights" and key != 'win_weight':
                    unity_config[key] = self.config[key]

        self.previous_action = [0, 0]
        logs.getLogger("mlagents.envs").setLevel(logs.WARNING)
        self.coverage_of_points.append(len(self.pos_buffer.keys()))
        self.episode += 1
        self.trajectories_for_episode[self.episode] = []
        self.actions_for_episode[self.episode] = []
        # self.set_spawn_position()

        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        state = dict(global_in=decision_steps.obs[0][0, :])
        # Append the value of the motivation weight
        state['global_in'] = np.concatenate([state['global_in']])
        position = state['global_in'][:5]
        self.trajectories_for_episode[self.episode].append(np.concatenate([position, state['global_in'][-2:]]))
        # print(np.reshape(state['global_in'][7:7 + 225], [15, 15]))

        return state

    def execute(self, actions, visualize=False):

        self.actions_for_episode[self.episode].append(actions)

        self.previous_action = actions
        actions = np.asarray(actions)
        actions = np.reshape(actions, [1, 1])
        actionsAT = ActionTuple()
        actionsAT.add_discrete(actions)
        self.unity_env.set_actions(self.behavior_name, actionsAT)
        self.unity_env.step()
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

        reward = None

        if(len(terminal_steps.interrupted) > 0):
            state = dict(global_in=terminal_steps.obs[0][0, :])
            done = True
            reward = terminal_steps.reward
        else:
            state = dict(global_in=decision_steps.obs[0][0, :])
            done = False
            reward = decision_steps.reward

        # Append the value of the motivation weight
        state['global_in'] = np.concatenate([state['global_in']])

        # Get the agent position from the state to compute reward
        position = state['global_in'][:5]
        self.trajectories_for_episode[self.episode].append(np.concatenate([position, state['global_in'][-2:]]))
        # Get the counter of that position and compute reward
        # counter = self.insert_to_pos_table(position[:2])
        # reward = self.compute_intrinsic_reward(counter)
        # reward = 0

        # print(np.flip(np.transpose(np.reshape(state['global_in'][10:10+225], [15, 15])), 0))
        # print(np.flip(np.transpose(np.reshape(state['global_in'][10+225:10+225 + 225], [15, 15])), 0))
        # Visualize 3D boxcast
        if visualize:
            threedgrid = np.reshape(state['global_in'][4:4 + 9261], [21, 21, 21])
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            filled = (1 - (threedgrid == 0))
            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(threedgrid.min(), threedgrid.max())
            ax.voxels(filled, facecolors=cmap(norm(threedgrid)), edgecolor="black")
            plt.show()

        # print(state['global_in'][-2:])
        # print(state['global_in'][7:7+12])
        # print(reward)
        # print(done)

        return state, done, reward

    def entropy(self, probs):
        entr = 0
        for p in probs:
            entr += (p * np.log(p))
        return -entr

    def set_config(self, config):
        self.config = config

    def print_observation(self, state):
        sum = state[:, :, 0] * 0
        for i in range(1,7):
            sum += state[:,:,i]*(i)
        print(sum)

    def set_demonstrations(self, demonstrations):

        self.dems = demonstrations

    def close(self):
        self.unity_env.close()

    def multidimensional_shifting(self, num_samples, sample_size, elements, probabilities):
        # replicate probabilities as many times as `num_samples`
        replicated_probabilities = np.tile(probabilities, (num_samples, 1))
        # get random shifting numbers & scale them correctly
        random_shifts = np.random.random(replicated_probabilities.shape)
        random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
        # shift by numbers & find largest (by finding the smallest of the negative)
        shifted_probabilities = random_shifts - replicated_probabilities
        return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]

    # Spawn a position from the buffer
    # If the buffer is empty, spawn at standard position
    def set_spawn_position(self):
        values = self.pos_buffer.values()
        if len(values) > 0:
            values = np.asarray(list(values))

            probs = 1 / values

            for i in range(len(values)):
                pos_key = list(self.pos_buffer.keys())[i]
                pos = np.asarray(list(map(float, pos_key.split(" "))))
                if pos[2] == 0:
                    probs[i] = 0

            probs = probs / np.sum(probs)

            index = self.multidimensional_shifting(1, 1, np.arange(len(probs)), probs)[0][0]

            pos_key = list(self.pos_buffer.keys())[index]
            pos = np.asarray(list(map(float, pos_key.split(" "))))
            # re-normalize pos to world coordinates
            pos = (((pos + 1) / 2) * 40) - 20

            self.config['agent_spawn_x'] = pos[0]
            self.config['agent_spawn_z'] = pos[1]
        else:
            self.config['agent_spawn_x'] = self.standard_position[0]
            self.config['agent_spawn_z'] = self.standard_position[1]

    def sample_position_from_dems(self):
        sample_index = np.random.randint(len(self.dems['obs']))
        spawn_position = deepcopy(self.dems['obs'][sample_index]['global_in'][:3])
        spawn_position[0] = (((spawn_position[0] + 1) / 2) * 500) - 250
        spawn_position[1] = (((spawn_position[1] + 1) / 2) * 500) - 250
        spawn_position[2] = (((spawn_position[2] + 1) / 2) * 59) + 1
        self.config['agent_spawn_x'] = spawn_position[0]
        self.config['agent_spawn_z'] = spawn_position[1]
        self.config['agent_spawn_y'] = spawn_position[2]

    # Insert to the table. Position must be a 2 element vector
    # Return the counter of that position
    def insert_to_pos_table(self, position):

        # Check if the position is already in the buffer
        for k in self.pos_buffer.keys():
            # If position - k < tau, then the position is already in the buffer
            # Add its counter to one and return it

            # The position are already normalized by the environment
            k_value = list(map(float, k.split(" ")))
            k_value = np.asarray(k_value)
            position = np.asarray(position)

            distance = np.linalg.norm(k_value - position)
            if distance < self.tau:
                self.pos_buffer[k] += 1
                return self.pos_buffer[k]

        pos_key = ' '.join(map(str, position))
        self.pos_buffer[pos_key] = 1
        return self.pos_buffer[pos_key]

    def clear_buffers(self):
        self.trajectories_for_episode = dict()
        self.actions_for_episode = dict()

    def command_to_action(self, command):

        if command == 'w':
            return 3
        if command == 'a':
            return 2
        if command == 's':
            return 4
        if command == 'd':
            return 1

        if command == 'e':
            return 5
        if command == 'c':
            return 7
        if command == 'z':
            return 6
        if command == 'q':
            return 8

        if command == 'r':
            return 0

        if command == ' ':
            return 9

        return 99

    # Compute the intrinsic reward based on the counter
    def compute_intrinsic_reward(self, counter):
        return self.r_max * (1 - (counter / self.max_counter))