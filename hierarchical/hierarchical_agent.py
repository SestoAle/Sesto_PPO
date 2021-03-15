from agents.PPO_gridworld import PPO
from tensorforce.execution import Runner

class HierarchicalAgent:

    def __init__(self, sess, manager_lr, workers_lr, num_workers):

        # Neural networks parameters
        self.sess = sess
        self.manager_lr = manager_lr
        self.workers_lr = workers_lr
        self.model_name = 'hierarchical'

        # Manager and Workers parameters
        self.memory_manager = 10
        self.memory_workers = 10

        self.manager = None
        self.workers = []

        # Instantiate the manager
        self.manager = PPO(
            sess, memory=self.memory_manager, p_lr=self.manager_lr,  name='manager', action_size=num_workers
        )

        # Instantiate the workers
        for i, w in enumerate(range(num_workers)):
            self.workers.append(
                PPO(sess, memory=self.memory_manager, p_lr=self.manager_lr,  name='worker_{}'.format(i))
            )

    def train(self):
        # Train the manager with its memory
        self.manager.train()
        # Train each of the workers with their memory
        for w in self.workers:
            w.train()

    # Add a transition to buffers
    def add_to_buffer(self, state, state_n, manager_action, worker_action,
                      reward, manager_old_prob, worker_old_prob, terminals):

        # Add experience to manager
        self.manager.add_to_buffer(state, state_n, manager_action, reward, manager_old_prob, terminals)

        # Add experience to worker
        self.workers[manager_action].add_to_buffer(state, state_n, worker_action, reward, worker_old_prob, terminals)


    def eval(self, state):

        # Make an action from manager
        man_action, man_logprob, man_probs = self.manager.eval(state)

        # Make an action from worker
        work_action, work_logprob, work_probs = self.workers[man_action[0]].eval(state)

        # Return all statistics from both manager and worker
        return work_action, man_action, man_logprob, work_logprob, man_probs, work_probs