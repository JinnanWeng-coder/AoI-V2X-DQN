import numpy as np
import torch as T
import torch.nn.functional as F
from Classes.networks import QNetwork


class IndependentDQNAgent:
    """One platoon leader = one independent Double-DQN agent.

    Task-decomposed Q-heads (Parvini's TDec lineage): a separate Q for task-1
    (V2V/CAM-delivery) and task-2 (V2I-revenue + AoI soft penalty). The greedy
    policy ranks discrete actions by the SUM Q1+Q2. This is the discrete analog of
    the MADDPG-TDec actor objective (-Q1 - Q2), and leaves a clean slot for the
    later CMDP step: add a Q^c head and select argmax_a [Q1 + Q2 - lambda * Q^c].
    """

    def __init__(self, lr, input_dims, n_actions, gamma, fc1_dims, fc2_dims,
                 batch_size, agent_label, tau=0.005):
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.tau = tau
        self.agent_label = agent_label

        self.q_task1 = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'q_task1', agent_label)
        self.q_task2 = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'q_task2', agent_label)
        self.target_q_task1 = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'target_q_task1', agent_label)
        self.target_q_task2 = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'target_q_task2', agent_label)
        self.update_target(tau=1.0)

    def choose_action(self, observation, epsilon):
        if np.random.random() < epsilon:
            return int(np.random.randint(self.n_actions))
        self.q_task1.eval()
        self.q_task2.eval()
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.q_task1.device)
        with T.no_grad():
            q = self.q_task1.forward(state) + self.q_task2.forward(state)
        return int(T.argmax(q, dim=1).item())

    def learn(self, states, actions, r1, r2, states_, dones):
        dev = self.q_task1.device
        states = T.tensor(states, dtype=T.float).to(dev)
        states_ = T.tensor(states_, dtype=T.float).to(dev)
        actions = T.tensor(actions, dtype=T.long).to(dev)            # (batch,)
        r1 = T.tensor(r1, dtype=T.float).to(dev)
        r2 = T.tensor(r2, dtype=T.float).to(dev)
        dones = T.tensor(dones, dtype=T.bool).to(dev)

        self.q_task1.train()
        self.q_task2.train()
        self.target_q_task1.eval()
        self.target_q_task2.eval()

        # Double-DQN: pick the bootstrap action with the ONLINE summed Q at s',
        # evaluate it with the per-task TARGET nets.
        with T.no_grad():
            a_star = T.argmax(self.q_task1.forward(states_) + self.q_task2.forward(states_), dim=1)
            idx_next = a_star.unsqueeze(1)
            q1_next = self.target_q_task1.forward(states_).gather(1, idx_next).squeeze(1)
            q2_next = self.target_q_task2.forward(states_).gather(1, idx_next).squeeze(1)
            q1_next[dones] = 0.0
            q2_next[dones] = 0.0
            y1 = r1 + self.gamma * q1_next
            y2 = r2 + self.gamma * q2_next

        idx = actions.unsqueeze(1)
        q1_pred = self.q_task1.forward(states).gather(1, idx).squeeze(1)
        q2_pred = self.q_task2.forward(states).gather(1, idx).squeeze(1)

        self.q_task1.optimizer.zero_grad()
        loss1 = F.mse_loss(q1_pred, y1)
        loss1.backward()
        self.q_task1.optimizer.step()

        self.q_task2.optimizer.zero_grad()
        loss2 = F.mse_loss(q2_pred, y2)
        loss2.backward()
        self.q_task2.optimizer.step()

        self.update_target()
        return float(loss1.detach().cpu()), float(loss2.detach().cpu())

    def update_target(self, tau=None):
        tau = self.tau if tau is None else tau
        for online, target in ((self.q_task1, self.target_q_task1),
                               (self.q_task2, self.target_q_task2)):
            for op, tp in zip(online.parameters(), target.parameters()):
                tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)

    def save_models(self):
        self.q_task1.save_checkpoint()
        self.q_task2.save_checkpoint()
        self.target_q_task1.save_checkpoint()
        self.target_q_task2.save_checkpoint()

    def load_models(self):
        self.q_task1.load_checkpoint()
        self.q_task2.load_checkpoint()
        self.target_q_task1.load_checkpoint()
        self.target_q_task2.load_checkpoint()
