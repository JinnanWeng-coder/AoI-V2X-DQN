import numpy as np
import torch as T
import torch.nn.functional as F
from Classes.networks import QNetwork


class IndependentDQNAgent:
    """One platoon leader = one independent Double-DQN agent (task-decomposed Q-heads).

    soft mode:  greedy = argmax_a [Q1 + Q2]            (AoI as the env's -AoI/20 penalty)
    hard mode [RQ1-CMDP]: add a per-platoon cost head Q^c (Bellman on 1{AoI>tau}) and a
                multiplier lambda_j; greedy = argmax_a [Q1 + Q2 - lambda_j * Q^c]. lambda_j
                is updated once per episode by the dual (Main.py). The cost head is built
                ONLY in hard mode, so the soft path is byte-identical to the validated
                baseline (no extra RNG draw at construction).
    """

    def __init__(self, lr, input_dims, n_actions, gamma, fc1_dims, fc2_dims,
                 batch_size, agent_label, tau=0.005, constraint_mode='soft'):
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.tau = tau
        self.agent_label = agent_label
        self.constraint_mode = constraint_mode
        self.lam = 0.0                                            # per-platoon Lagrange multiplier

        self.q_task1 = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'q_task1', agent_label)
        self.q_task2 = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'q_task2', agent_label)
        self.target_q_task1 = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'target_q_task1', agent_label)
        self.target_q_task2 = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'target_q_task2', agent_label)
        if self.constraint_mode == 'hard':                       # [RQ1-CMDP] cost head — hard mode only
            self.q_cost = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'q_cost', agent_label)
            self.target_q_cost = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'target_q_cost', agent_label)
        else:
            self.q_cost = self.target_q_cost = None
        self.update_target(tau=1.0)

    def _greedy_q(self, q1, q2, qc):
        # constraint-adjusted action value: maximise reward, minimise discounted cost
        if self.constraint_mode == 'hard':
            return q1 + q2 - self.lam * qc
        return q1 + q2

    def choose_action(self, observation, epsilon):
        if np.random.random() < epsilon:
            return int(np.random.randint(self.n_actions))
        self.q_task1.eval(); self.q_task2.eval()
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.q_task1.device)
        with T.no_grad():
            q1 = self.q_task1.forward(state)
            q2 = self.q_task2.forward(state)
            qc = self.q_cost.forward(state) if self.constraint_mode == 'hard' else 0.0
            q = self._greedy_q(q1, q2, qc)
        return int(T.argmax(q, dim=1).item())

    def learn(self, states, actions, r1, r2, r_cost, states_, dones):
        dev = self.q_task1.device
        hard = (self.constraint_mode == 'hard')
        states = T.tensor(states, dtype=T.float).to(dev)
        states_ = T.tensor(states_, dtype=T.float).to(dev)
        actions = T.tensor(actions, dtype=T.long).to(dev)
        r1 = T.tensor(r1, dtype=T.float).to(dev)
        r2 = T.tensor(r2, dtype=T.float).to(dev)
        dones = T.tensor(dones, dtype=T.bool).to(dev)
        if hard:
            r_cost = T.tensor(r_cost, dtype=T.float).to(dev)

        self.q_task1.train(); self.q_task2.train()
        self.target_q_task1.eval(); self.target_q_task2.eval()
        if hard:
            self.q_cost.train(); self.target_q_cost.eval()

        # Double-DQN: bootstrap action from the ONLINE constraint-adjusted Q at s'
        with T.no_grad():
            q1n = self.q_task1.forward(states_)
            q2n = self.q_task2.forward(states_)
            qcn = self.q_cost.forward(states_) if hard else 0.0
            a_star = T.argmax(self._greedy_q(q1n, q2n, qcn), dim=1)
            ix = a_star.unsqueeze(1)
            q1_next = self.target_q_task1.forward(states_).gather(1, ix).squeeze(1)
            q2_next = self.target_q_task2.forward(states_).gather(1, ix).squeeze(1)
            q1_next[dones] = 0.0
            q2_next[dones] = 0.0
            y1 = r1 + self.gamma * q1_next
            y2 = r2 + self.gamma * q2_next
            if hard:
                qc_next = self.target_q_cost.forward(states_).gather(1, ix).squeeze(1)
                qc_next[dones] = 0.0
                yc = r_cost + self.gamma * qc_next

        idx = actions.unsqueeze(1)
        q1_pred = self.q_task1.forward(states).gather(1, idx).squeeze(1)
        q2_pred = self.q_task2.forward(states).gather(1, idx).squeeze(1)
        self.q_task1.optimizer.zero_grad(); F.mse_loss(q1_pred, y1).backward(); self.q_task1.optimizer.step()
        self.q_task2.optimizer.zero_grad(); F.mse_loss(q2_pred, y2).backward(); self.q_task2.optimizer.step()
        if hard:
            qc_pred = self.q_cost.forward(states).gather(1, idx).squeeze(1)
            self.q_cost.optimizer.zero_grad(); F.mse_loss(qc_pred, yc).backward(); self.q_cost.optimizer.step()

        self.update_target()

    def update_target(self, tau=None):
        tau = self.tau if tau is None else tau
        pairs = [(self.q_task1, self.target_q_task1), (self.q_task2, self.target_q_task2)]
        if self.constraint_mode == 'hard':
            pairs.append((self.q_cost, self.target_q_cost))
        for online, target in pairs:
            for op, tp in zip(online.parameters(), target.parameters()):
                tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)

    def save_models(self):
        for net in (self.q_task1, self.q_task2, self.target_q_task1, self.target_q_task2):
            net.save_checkpoint()
        if self.constraint_mode == 'hard':
            self.q_cost.save_checkpoint(); self.target_q_cost.save_checkpoint()

    def load_models(self):
        for net in (self.q_task1, self.q_task2, self.target_q_task1, self.target_q_task2):
            net.load_checkpoint()
        if self.constraint_mode == 'hard':
            self.q_cost.load_checkpoint(); self.target_q_cost.load_checkpoint()
