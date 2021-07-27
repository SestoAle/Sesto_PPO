import os
import numpy as np
import json
from utils import NumpyEncoder
import time
from threading import Thread

# Act thread
# Act thread
class ActThreaded(Thread):
    def __init__(self, agent, env, parallel_buffer, index, config, num_steps, states, recurrent=False, internals=None,
                 v_internals=None):
        self.env = env
        self.parallel_buffer = parallel_buffer
        self.index = index
        self.env.set_config(config)
        self.num_steps = num_steps
        self.agent = agent
        self.states = states
        self.internals = internals
        self.v_internals = v_internals
        self.recurrent = recurrent
        super().__init__()


    def run(self) -> None:
        state = self.states[self.index]
        if self.recurrent:
            internal = self.internals[self.index]
            v_internal = self.v_internals[self.index]
        total_reward = 0
        step = 0
        entropies = []
        for i in range(self.num_steps):
            # Execute the environment with the action
            if not self.recurrent:
                actions, logprobs, probs = self.agent.eval([state])
            else:
                actions, logprobs, probs, internal_n, v_internal_n = self.agent.eval_recurrent([state], internal,
                                                                                               v_internal)
            entropies.append(self.env.entropy(probs[0]))
            actions = actions[0]
            state_n, done, reward = self.env.execute(actions)
            step += 1
            total_reward += reward

            self.parallel_buffer['states'][self.index].append(state)
            self.parallel_buffer['states_n'][self.index].append(state_n)
            self.parallel_buffer['done'][self.index].append(done)
            self.parallel_buffer['reward'][self.index].append(reward)
            self.parallel_buffer['action'][self.index].append(actions)
            self.parallel_buffer['logprob'][self.index].append(logprobs)
            if self.recurrent:
                self.parallel_buffer['internal'][self.index].append(internal)
                self.parallel_buffer['v_internal'][self.index].append(v_internal)
                internal = internal_n
                v_internal = v_internal_n


            state = state_n
            if done:
                state = self.env.reset()
                if self.recurrent:
                    internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))
                    v_internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))

        if not done:
            self.parallel_buffer['done'][self.index][-1] = 2

        self.parallel_buffer['episode_rewards'][self.index].append(total_reward)
        self.parallel_buffer['episode_timesteps'][self.index].append(self.num_steps)
        self.parallel_buffer['mean_entropies'][self.index].append(0)
        self.parallel_buffer['std_entropies'][self.index].append(0)
        self.states[self.index] = state
        if self.recurrent:
            self.internals[self.index] = internal
            self.v_internals[self.index] = v_internal

# Epsiode thread
class EpisodeThreaded(Thread):
    def __init__(self, env, parallel_buffer, agent_imitation, agent_motivation, index, config, num_episode=1, recurrent=False, motivation=False,
                 reward_model=False, phase='imitation'):
        self.env = env
        self.parallel_buffer = parallel_buffer
        self.agent_imitation = agent_imitation
        self.agent_motivation = agent_motivation
        self.index = index
        self.num_episode = num_episode
        self.recurrent = recurrent
        self.env.set_config(config)
        self.motivation = motivation
        self.reward_model = reward_model
        self.phase = phase
        self.no_fusion = False
        super().__init__()

    # Entropy-weighted mixture policy for policy fusion
    def entropy_weighted_fusion(self, main_probs, sub_probs):
        # Entropy of the main policy
        sub_entr = self.entropy(sub_probs)
        total_probs = (sub_entr) * main_probs + (1 - sub_entr) * sub_probs
        total_probs = self.boltzmann(total_probs[0])
        actions = np.argmax(np.random.multinomial(1, total_probs))
        print("oh")
        return actions

    def boltzmann(self, probs, temperature=1.):
        sum = np.sum(np.power(probs, 1 / temperature))
        new_probs = []
        for p in probs:
            new_probs.append(np.power(p, 1 / temperature) / sum)

        return np.asarray(new_probs)

    def entropy(self, probs):
        entr = 0
        for p in probs:
            entr += p * np.log(p)
        return np.clip(-entr, 0, 1)

    def run(self) -> None:
        # Run each thread for num_episode episodes

        for i in range(self.num_episode):
            done = False
            step = 0
            # Reset the environment
            state = self.env.reset()

            # Total episode reward
            episode_reward = 0

            # Local entropies of the episode
            local_entropies = []

            # If recurrent, initialize hidden state
            if self.recurrent:
                internal_imitation = (np.zeros([1, self.agent_imitation.recurrent_size]), np.zeros([1, self.agent_imitation.recurrent_size]))
                v_internal_imitation = (np.zeros([1, self.agent_imitation.recurrent_size]), np.zeros([1, self.agent_imitation.recurrent_size]))
                internal_motivation = (np.zeros([1, self.agent_motivation.recurrent_size]), np.zeros([1, self.agent_motivation.recurrent_size]))
                v_internal_motivation = (np.zeros([1, self.agent_motivation.recurrent_size]), np.zeros([1, self.agent_motivation.recurrent_size]))

            while not done:
                # Evaluation - Execute step
                if not self.recurrent:
                    actions_imitation, logprobs, probs_imitation = self.agent_imitation.eval([state])
                    actions_motivation, logprobs, probs_motivation = self.agent_motivation.eval([state])
                else:
                    actions_imitation, logprobs, probs_imitation, internal_n_imitation, v_internal_n_imitation = \
                        self.agent_imitation.eval_recurrent([state], internal_imitation, v_internal_imitation)
                    internal_n_motivation = internal_n_imitation
                    v_internal_n_motivation = v_internal_n_imitation
                    #actions_motivation, logprobs, probs_motivation, internal_n_motivation, v_internal_n_motivation \
                    #    = self.agent_motivation.eval_recurrent([state], internal_motivation, v_internal_motivation)

                if self.phase == 'imitation':
                    actions = actions_imitation[0]
                else:
                    #actions = actions_motivation[0]
                    actions = self.entropy_weighted_fusion(probs_imitation, probs_motivation)

                probs = probs_imitation
                state_n, done, reward = self.env.execute(actions)

                #reward = reward[0]
                #done = done[0]

                episode_reward += reward
                local_entropies.append(self.env.entropy(probs[0]))
                # If step is equal than max timesteps, terminate the episode
                if step >= self.env._max_episode_timesteps - 1:
                    done = True
                self.parallel_buffer['states'][self.index].append(state)
                self.parallel_buffer['states_n'][self.index].append(state_n)
                self.parallel_buffer['done'][self.index].append(done)
                self.parallel_buffer['reward'][self.index].append(reward)
                self.parallel_buffer['action'][self.index].append(actions)
                self.parallel_buffer['logprob'][self.index].append(logprobs)

                if self.recurrent:
                    self.parallel_buffer['internal_imitation'][self.index].append(internal_imitation)
                    self.parallel_buffer['v_internal_imitation'][self.index].append(v_internal_imitation)
                    internal_imitation = internal_n_imitation
                    v_internal_imitation = v_internal_n_imitation
                    self.parallel_buffer['internal_motivation'][self.index].append(internal_motivation)
                    self.parallel_buffer['v_internal_motivation'][self.index].append(v_internal_motivation)
                    internal_motivation = internal_n_motivation
                    v_internal_motivation = v_internal_n_motivation

                if self.motivation:
                    self.parallel_buffer['motivation'][self.index]['state_n'].append(state_n)

                if self.reward_model:
                    self.parallel_buffer['reward_model'][self.index]['state'].append(state)
                    self.parallel_buffer['reward_model'][self.index]['state_n'].append(state_n)
                    self.parallel_buffer['reward_model'][self.index]['action'].append(actions)

                state = state_n
                step += 1

            # History statistics
            self.parallel_buffer['episode_rewards'][self.index].append(episode_reward)
            self.parallel_buffer['episode_timesteps'][self.index].append(step)
            self.parallel_buffer['mean_entropies'][self.index].append(np.mean(local_entropies))
            self.parallel_buffer['std_entropies'][self.index].append(np.std(local_entropies))


class Runner:
    def __init__(self, agent_il, agent_im, frequency, envs, save_frequency=3000, logging=100, total_episode=1e10, curriculum=None,
                 frequency_mode='episodes', random_actions=None, curriculum_mode='steps', evaluation=False,
                 callback_function=None, motivation=None,
                 # IRL
                 reward_model=None, fixed_reward_model=False, dems_name='', reward_frequency=30,
                 # Adversarial Play
                 adversarial_play=False, double_agent=None,
                 # Bug detection fusion
                 fusion_frequency=1000,
                 **kwargs):

        # Runner objects and parameters
        self.agent_imitation = agent_il
        self.agent_motivation = agent_im
        self.curriculum = curriculum
        self.total_episode = total_episode
        self.frequency = frequency
        self.frequency_mode = frequency_mode
        self.random_actions = random_actions
        self.logging = logging
        self.save_frequency = save_frequency
        self.envs = envs
        self.curriculum_mode = curriculum_mode
        self.evaluation = evaluation

        # TODO: pass this as an argument
        self.motivation_frequency = 5

        # For alternating between motivation and imitation reward
        self.alternate_frequency = 0
        self.alternate_count = 0
        self.alternate_turn = 0

        # Fusion detection
        self.fusion_frequency = fusion_frequency
        self.phase = 'motivation'

        # If we want to use intrinsic motivation
        # Right now only RND is available
        self.motivation = motivation

        # Function to call at the end of each episode.
        # It takes the agent, the runner and the env as input arguments
        self.callback_function = callback_function

        # Recurrent
        self.recurrent = self.agent_imitation.recurrent

        # Objects and parameters for IRL
        self.reward_model = reward_model
        self.fixed_reward_model = fixed_reward_model
        self.dems_name = dems_name
        self.reward_frequency = reward_frequency

        # Adversarial play
        self.adversarial_play = adversarial_play
        self.double_agent = double_agent
        # If adversarial play, save the first version of the main agent and load it to the double agent
        if self.adversarial_play:
            self.agent.save_model(name=self.agent.model_name + '_0', folder='saved/adversarial')
            self.double_agent.load_model(name=self.agent.model_name + '_0', folder='saved/adversarial')

        # Global runner statistics
        # total episode
        self.ep = 0
        # total steps
        self.total_step = 0
        # Initialize history
        # History to save model statistics
        self.history = {
            "episode_rewards": [],
            "episode_timesteps": [],
            "mean_entropies": [],
            "std_entropies": [],
            "reward_model_loss": [],
            "env_rewards": []
        }

        # Initialize parallel buffer for savig experience of each thread without race conditions
        self.parallel_buffer = None
        self.parallel_buffer = self.clear_parallel_buffer()

        # Initialize reward model
        if self.reward_model is not None:
            if not self.fixed_reward_model:
                # Ask for demonstrations
                answer = None
                while answer != 'y' and answer != 'n':
                    answer = input('Do you want to create new demonstrations? [y/n] ')
                # Before asking for demonstrations, set the curriculum of the environment
                config = self.set_curriculum(self.curriculum, self.history, self.curriculum_mode)
                self.envs[0].set_config(config)
                if answer == 'y':
                    dems, vals = self.reward_model.create_demonstrations(env=self.envs[0])
                elif answer == 'p':
                    dems, vals = self.reward_model.create_demonstrations(env=self.envs[0], with_policy=True)
                else:
                    print('Loading demonstrations...')
                    dems, vals = self.reward_model.load_demonstrations(self.dems_name)

                print('Demonstrations loaded! We have ' + str(len(dems['obs'])) + " timesteps in these demonstrations")
                # print('and ' + str(len(vals['obs'])) + " timesteps in these validations.")

                # Getting initial experience from the environment to do the first training epoch of the reward model
                self.get_experience(self.envs[0], self.reward_frequency, random=True)
                self.reward_model.train()

        # For curriculum training
        self.start_training = 0
        self.current_curriculum_step = 0

        # If a saved model with the model_name already exists, load it (and the history attached to it)
        if os.path.exists('{}/{}.meta'.format('saved', self.agent_imitation.model_name)):
            answer = None
            while answer != 'y' and answer != 'n':
                answer = input("There's already an agent saved with name {}, "
                               "do you want to continue training? [y/n] ".format(agent_im.model_name))

            if answer == 'y':
                self.history = self.load_model(agent_im.model_name, agent_im)
                self.ep = len(self.history['episode_timesteps'])
                self.total_step = np.sum(self.history['episode_timesteps'])

        # Decaying weight of the motivation/inverse reinforcement learning model
        self.last_episode_for_decaying = 0
        # if self.motivation is not None:
        #     self.motivation.motivation_weight = 0.8
        #     self.min_motivation_weight = 0.2

    # Return a list of thread, that will save the experience in the shared buffer
    # The thread will run for 1 episode
    def create_episode_threads(self, parallel_buffer, agent_imitation, agent_motivation, config, phase):
        # The number of thread will be equal to the number of environments
        threads = []
        for i, e in enumerate(self.envs):
            # Create a thread
            threads.append(EpisodeThreaded(env=e, index=i, agent_imitation=agent_imitation, agent_motivation=agent_motivation,
                                           parallel_buffer=parallel_buffer, config=config,
                                           recurrent=self.recurrent, motivation=(self.motivation is not None),
                                           reward_model=(self.reward_model is not None), phase=phase))

        # Return threads
        return threads

    # Return a list of thread, that will save the experience in the shared buffer
    # The thread will run for 1 step of the environment
    def create_act_threds(self, agent, parallel_buffer, config, states, num_steps, internals=None, v_internals=None):
        # The number of thread will be equal to the number of environments
        threads = []
        for i, e in enumerate(self.envs):
            # Create a thread
            threads.append(ActThreaded(agent=agent, env=e, index=i, parallel_buffer=parallel_buffer, config=config,
                                       states=states, num_steps=num_steps, recurrent=self.recurrent,
                                       internals=internals, v_internals=v_internals))

        # Return threads
        return threads

    # Clear parallel buffer to avoid memory leak
    def clear_parallel_buffer(self):
        # Manually delete parallel buffer
        if self.parallel_buffer is not None:
            del self.parallel_buffer
        # Initialize parallel buffer for savig experience of each thread without race conditions
        parallel_buffer = {
            'states': [],
            'states_n': [],
            'done': [],
            'reward': [],
            'action': [],
            'logprob': [],
            'internal_imitation': [],
            'v_internal_imitation': [],
            'internal_motivation': [],
            'v_internal_motivation': [],
            # Motivation
            'motivation': [],
            # Reward model
            'reward_model': [],
            # History
            'episode_rewards': [],
            'episode_timesteps': [],
            'mean_entropies': [],
            'std_entropies': [],
        }

        for i in range(len(self.envs)):
            parallel_buffer['states'].append([])
            parallel_buffer['states_n'].append([])
            parallel_buffer['done'].append([])
            parallel_buffer['reward'].append([])
            parallel_buffer['action'].append([])
            parallel_buffer['logprob'].append([])
            parallel_buffer['internal_imitation'].append([])
            parallel_buffer['v_internal_imitation'].append([])
            parallel_buffer['internal_motivation'].append([])
            parallel_buffer['v_internal_motivation'].append([])
            # Motivation
            parallel_buffer['motivation'].append(
                dict(state_n=[])
            )
            # Reward Model
            parallel_buffer['reward_model'].append(
                dict(state=[], state_n=[], action=[])
            )

            # History
            parallel_buffer['episode_rewards'].append([])
            parallel_buffer['episode_timesteps'].append([])
            parallel_buffer['mean_entropies'].append([])
            parallel_buffer['std_entropies'].append([])

        return parallel_buffer

    def run(self):

        # Trainin loop
        # Start training
        start_time = time.time()
        # If parallel act is in use, reset all environments at beginning of training

        if self.frequency_mode == 'timesteps':
            states = []
            if self.recurrent:
                internals = []
                v_internals = []
            for env in self.envs:
                config = self.set_curriculum(self.curriculum, self.history, self.curriculum_mode)
                env.set_config(config)
                states.append(env.reset())
                if self.recurrent:
                    internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))
                    v_internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))
                    internals.append(internal)
                    v_internals.append(v_internal)

        while self.ep <= self.total_episode:
            # Reset the episode
            # Set actual curriculum
            config = self.set_curriculum(self.curriculum, self.history, self.curriculum_mode)
            if self.start_training == 0:
                print(config)
            self.start_training = 1

            # Episode loop
            if self.frequency_mode=='episodes':
            # If frequency is episode, run the episodes in parallel
                # Check the phase
                if self.ep % self.fusion_frequency == 0:
                    if self.phase == 'imitation':
                        self.phase = 'motivation'
                    else:
                        self.phase = 'imitation'

                # Create threads
                threads = self.create_episode_threads(agent_imitation=self.agent_imitation, agent_motivation=self.agent_motivation,
                                                      parallel_buffer=self.parallel_buffer, config=config, phase=self.phase)

                # Run the threads
                for t in threads:
                    t.start()

                # Wait for the threads to finish
                for t in threads:
                    t.join()

                self.ep += len(threads)
            else:
            # If frequency is timesteps, run only the 'execute' in parallel for horizon steps
                # Create threads
                if self.recurrent:
                    threads = self.create_act_threds(agent=self.agent, parallel_buffer=self.parallel_buffer, config=config,
                                                 states=states, num_steps=self.frequency, internals=internals, v_internals=v_internals)
                else:
                    threads = self.create_act_threds(agent=self.agent, parallel_buffer=self.parallel_buffer, config=config,
                                                 states=states, num_steps=self.frequency)

                for t in threads:
                    t.start()

                for t in threads:
                    t.join()

                # Delete threads from memory
                del threads[:]

                # Get how many episodes and steps are passed within threads
                self.ep += np.sum(np.asarray(self.parallel_buffer['done'][:]) == 1)
                self.total_step = len(threads) * self.frequency

            # Add the overall experience to the buffer
            # Update the history
            for i in range(len(self.envs)):

                if not self.recurrent:
                    # Add to the agent experience in order of execution
                    for state, state_n, action, reward, logprob, done in zip(
                            self.parallel_buffer['states'][i],
                            self.parallel_buffer['states_n'][i],
                            self.parallel_buffer['action'][i],
                            self.parallel_buffer['reward'][i],
                            self.parallel_buffer['logprob'][i],
                            self.parallel_buffer['done'][i]
                    ):
                        self.agent_imitation.add_to_buffer(state, state_n, action, reward, logprob, done)
                        self.agent_motivation.add_to_buffer(state, state_n, action, reward, logprob, done)
                else:
                    # Add to the agent experience in order of execution
                    for state, state_n, action, reward, logprob, done, internal_imitation, v_internal_imitation,\
                            internal_motivation, v_internal_motivation in zip(
                            self.parallel_buffer['states'][i],
                            self.parallel_buffer['states_n'][i],
                            self.parallel_buffer['action'][i],
                            self.parallel_buffer['reward'][i],
                            self.parallel_buffer['logprob'][i],
                            self.parallel_buffer['done'][i],
                            self.parallel_buffer['internal_imitation'][i],
                            self.parallel_buffer['v_internal_imitation'][i],
                            self.parallel_buffer['internal_motivation'][i],
                            self.parallel_buffer['v_internal_motivation'][i],
                    ):
                        try:
                            self.agent_imitation.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                     internal_imitation.c[0], internal_imitation.h[0], v_internal_imitation.c[0], v_internal_imitation.h[0])
                            self.agent_motivation.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                     internal_motivation.c[0], internal_motivation.h[0], v_internal_motivation.c[0], v_internal_motivation.h[0])
                        except Exception as e:
                            zero_state = np.reshape(internal_imitation[0], [-1, ])
                            self.agent_imitation.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                     zero_state, zero_state, zero_state, zero_state)
                            self.agent_motivation.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                     zero_state, zero_state, zero_state, zero_state)

                # For motivation, add the agents experience to the motivation buffer
                for state_n in self.parallel_buffer['motivation'][i]['state_n']:
                    self.motivation.add_to_buffer(state_n)

                # For reward model, add the agents experience to the reward model buffer
                for state, state_n, action in zip(self.parallel_buffer['reward_model'][i]['state'],
                                                  self.parallel_buffer['reward_model'][i]['state_n'],
                                                  self.parallel_buffer['reward_model'][i]['action']):
                    self.reward_model.add_to_policy_buffer(state, state_n, action)

                # Upadte the hisotry in order of execution
                for episode_reward, step, mean_entropies, std_entropies in zip(
                        self.parallel_buffer['episode_rewards'][i],
                        self.parallel_buffer['episode_timesteps'][i],
                        self.parallel_buffer['mean_entropies'][i],
                        self.parallel_buffer['std_entropies'][i],

                ):
                    self.history['episode_rewards'].append(episode_reward)
                    self.history['episode_timesteps'].append(step)
                    self.history['mean_entropies'].append(mean_entropies)
                    self.history['std_entropies'].append(std_entropies)

            # Clear parallel buffer
            self.parallel_buffer = self.clear_parallel_buffer()
            if self.frequency_mode == 'timesteps':
                self.agent.train()

            # Logging information
            if self.ep > 0 and self.ep % self.logging == 0:
                print('Mean of {} episode reward after {} episodes: {}'.
                      format(self.logging, self.ep, np.mean(self.history['episode_rewards'][-self.logging:])))

                if self.reward_model is not None:
                    print('Mean of {} environment episode reward after {} episodes: {}'.
                            format(self.logging, self.ep, np.mean(self.history['env_rewards'][-self.logging:])))

                print('The agent made a total of {} steps'.format(np.sum(self.history['episode_timesteps'])))

                if self.callback_function is not None:
                    self.callback_function(self.envs, self)

                self.timer(start_time, time.time())

            # If frequency episodes are passed, update the policy
            if not self.evaluation and self.frequency_mode == 'episodes' and \
                    self.ep > 0 and self.ep % self.motivation_frequency == 0:

                # If we use intrinsic motivation, update also intrinsic motivation
                if self.motivation is not None:
                    self.update_motivation()

            # If frequency episodes are passed, update the policy
            if not self.evaluation and self.frequency_mode == 'episodes' and \
                    self.ep > 0 and self.ep % self.frequency == 0:

                if self.random_actions is not None:
                    if self.total_step <= self.random_actions:
                        self.motivation.clear_buffer()
                        continue

                if self.motivation is not None:
                    # Normalize observation of the motivation buffer
                    # self.motivation.normalize_buffer()
                    # Compute intrinsic rewards
                    intrinsic_rews = self.motivation.eval(self.agent_motivation.buffer['states_n'])

                    # Normalize rewards
                    # intrinsic_rews -= self.motivation.r_norm.mean
                    # intrinsic_rews /= self.motivation.r_norm.std
                    intrinsic_rews -= np.mean(intrinsic_rews)
                    intrinsic_rews /= np.std(intrinsic_rews)
                    intrinsic_rews *= self.motivation.motivation_weight
                    self.agent_motivation.buffer['rewards'] = list(intrinsic_rews)

                if self.reward_model is not None:

                    # Compute intrinsic rewards
                    intrinsic_rews = self.reward_model.eval(self.agent_imitation.buffer['states'],
                                                            self.agent_imitation.buffer['states_n'],
                                                            self.agent_imitation.buffer['actions'])

                    # Normalize rewards
                    # intrinsic_rews -= self.reward_model.r_norm.mean
                    # intrinsic_rews /= self.reward_model.r_norm.std

                    #intrinsic_rews = (intrinsic_rews - np.min(intrinsic_rews)) / (np.max(intrinsic_rews) - np.min(intrinsic_rews))
                    intrinsic_rews -= np.mean(intrinsic_rews)
                    intrinsic_rews /= np.std(intrinsic_rews)
                    intrinsic_rews *= self.reward_model.reward_model_weight

                    self.agent_imitation.buffer['rewards'] = list(intrinsic_rews)

                self.agent_imitation.train()
                self.agent_motivation.train()
                # For alternating between motivation and imitation learning
                if self.alternate_frequency > 0:
                    self.alternate_count += 1
                    if self.alternate_count % self.alternate_frequency == 0:
                        self.alternate_turn = (self.alternate_turn + 1) % 2

            # If frequency episodes are passed, update the policy
            if not self.evaluation and self.frequency_mode == 'episodes' and \
                    self.ep > 0 and self.ep % self.reward_frequency == 0:

                # If we use intrinsic motivation, update also intrinsic motivation
                if self.reward_model is not None and not self.fixed_reward_model:
                    self.update_reward_model()

            # Save model and statistics
            if self.ep > 0 and self.ep % self.save_frequency == 0:
                self.save_model(self.history, self.agent_imitation.model_name, self.curriculum)

    def save_model(self, history, model_name, curriculum):

        # Save statistics as json
        json_str = json.dumps(history, cls=NumpyEncoder)
        f = open("arrays/{}.json".format(model_name), "w")
        f.write(json_str)
        f.close()

        # Save curriculum as json
        json_str = json.dumps(curriculum, cls=NumpyEncoder)
        f = open("arrays/{}_curriculum.json".format(model_name), "w")
        f.write(json_str)
        f.close()

        # Save the tf model
        #agent.save_model(name=model_name, folder='saved')

        # If we use intrinsic motivation, save the motivation model
        if self.motivation is not None:
            self.motivation.save_model(name=model_name, folder='saved')

        # If we use IRL, save the reward model
        if self.reward_model is not None and not self.fixed_reward_model:
            self.reward_model.save_model('{}_{}'.format(model_name, self.ep))

        print('Model saved with name: {}'.format(model_name))

    def load_model(self, model_name, agent):
        agent.load_model(name=model_name, folder='saved')

        # Load intrinsic motivation for testing
        if self.motivation is not None:
            self.motivation.load_model(name=model_name, folder='saved')

        # # Load reward motivation for testing
        # if self.reward_model is not None:
        #     self.reward_model.load_model(name=model_name, folder='saved')

        with open("arrays/{}.json".format(model_name)) as f:
            history = json.load(f)

        return history

    # Update curriculum for DeepCrawl
    def set_curriculum(self, curriculum, history, mode='steps'):

        total_timesteps = np.sum(history['episode_timesteps'])
        total_episodes = len(history['episode_timesteps'])

        if curriculum == None:
            return None

        if mode == 'episodes':
            lessons = np.cumsum(curriculum['thresholds'])
            curriculum_step = 0

            for (index, l) in enumerate(lessons):

                if total_episodes > l:
                    curriculum_step = index + 1

        if mode == 'steps':
            lessons = np.cumsum(curriculum['thresholds'])

            curriculum_step = 0

            for (index, l) in enumerate(lessons):

                if total_timesteps > l:
                    curriculum_step = index + 1

        parameters = curriculum['parameters']
        config = {}

        for (par, value) in parameters.items():
            config[par] = value[curriculum_step]

        # If Adversarial play
        if self.adversarial_play:
            if curriculum_step > self.current_curriculum_step:
                # Save the current version of the main agent
                self.agent.save_model(name=self.agent.model_name + '_' + str(curriculum_step),
                                      folder='saved/adversarial')
                # Load the weights of the current version of the main agent to the double agent
                self.double_agent.load_model(name=self.agent.model_name + '_' + str(curriculum_step),
                                             folder='saved/adversarial')

        self.current_curriculum_step = curriculum_step

        return config

    # For IRL, get initial experience from environment, the agent act in the env without update itself
    def get_experience(self, env, num_discriminator_exp=None, verbose=False, random=False):

        if num_discriminator_exp == None:
            num_discriminator_exp = self.frequency

        # For policy update number
        for ep in range(num_discriminator_exp):
            states = []
            state = env.reset()
            step = 0
            # While the episode si not finished
            reward = 0
            while True:
                step += 1
                if random:
                    num_actions = self.agent_imitation.action_size
                    action = np.random.randint(0, num_actions)
                else:
                    action, _, c_probs = self.agent_imitation.eval([state])
                state_n, terminal, step_reward = env.execute(actions=action)
                self.reward_model.add_to_policy_buffer(state, state_n, action)

                state = state_n
                reward += step_reward
                if terminal or step >= env._max_episode_timesteps:
                    break

            if verbose:
                print("Reward at the end of episode " + str(ep + 1) + ": " + str(reward))

    # Update intrinsic motivation
    # Update its statistics AND train the model. We print also the model loss
    def update_motivation(self):
        loss = self.motivation.train()
        # print('Mean motivation loss = {}'.format(loss))

    # Update reward model
    # Update its statistics AND train the model. We print also the model loss
    def update_reward_model(self):
        loss, _ = self.reward_model.train()
        # print('Mean reward loss = {}'.format(loss))

    # Method for count time after each episode
    def timer(self, start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
