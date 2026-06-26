import numpy as np


class ReplayBuffer:
    """Shared replay over JOINT transitions; each independent agent slices its own
    observation / action-index / per-task rewards at learn time. Discrete actions are
    stored as integer indices (one per platoon)."""

    def __init__(self, max_size, input_shape, n_agents):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.state_memory = np.zeros((max_size, input_shape * n_agents), dtype=np.float32)
        self.new_state_memory = np.zeros((max_size, input_shape * n_agents), dtype=np.float32)
        self.action_memory = np.zeros((max_size, n_agents), dtype=np.int64)      # discrete action index / agent
        self.reward_task1 = np.zeros((max_size, n_agents), dtype=np.float32)
        self.reward_task2 = np.zeros((max_size, n_agents), dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=bool)

    def store_transition(self, state, action_idx, reward_t1, reward_t2, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action_idx
        self.reward_task1[index] = reward_t1
        self.reward_task2[index] = reward_t2
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        return (self.state_memory[batch], self.action_memory[batch],
                self.reward_task1[batch], self.reward_task2[batch],
                self.new_state_memory[batch], self.terminal_memory[batch])
