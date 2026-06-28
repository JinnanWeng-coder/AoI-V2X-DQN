import numpy as np
import torch as T
import torch.nn.functional as F
from Classes.networks import QNetwork


class IndependentDQNAgent:
    """Independent Double-DQN with task-decomposed Q-heads (Q1=V2V/CAM, Q2=V2I+AoI).

    soft : greedy argmax[Q1+Q2]; AoI enters via the env's -AoI/20 reward penalty.
    hard + cost_source='raw' (RCPO, DEFAULT): fold the per-step constraint penalty
        -lambda_j*c (c=1{AoI>tau}) into the task-2 reward TARGET
        (y2 = r2 - lambda_j*c + gamma*Q2'); greedy stays argmax[Q1+Q2] (the constraint
        is baked into Q2). lambda multiplies a 0/1 cost so it stays on the reward scale
        -> no value-scale domination, unlike the argmax-critic variant below.
    hard + cost_source='critic': separate cost head Q^c (Double-DQN Bellman on c);
        greedy = argmax[Q1+Q2 - lambda_j*Q^c]. Combining raw Q-VALUES is value-scale
        sensitive (Q^c >> the action-discriminating part of Q1+Q2), so keep lam_max small.
    lambda_j is updated once per episode by the dual (Main.py). Extra nets are built ONLY
    when used, so the soft path is byte-identical to the validated baseline.
    """

    def __init__(self, lr, input_dims, n_actions, gamma, fc1_dims, fc2_dims,
                 batch_size, agent_label, tau=0.005, constraint_mode='soft', cost_source='raw'):
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.tau = tau
        self.agent_label = agent_label
        self.constraint_mode = constraint_mode
        self.cost_source = cost_source
        self.hard = (constraint_mode == 'hard')
        self.use_cost_critic = (self.hard and cost_source == 'critic')
        self.lam = 0.0                                            # per-platoon Lagrange multiplier

        self.q_task1 = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'q_task1', agent_label)
        self.q_task2 = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'q_task2', agent_label)
        self.target_q_task1 = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'target_q_task1', agent_label)
        self.target_q_task2 = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'target_q_task2', agent_label)
        if self.use_cost_critic:                                 # cost head: critic mode only
            self.q_cost = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'q_cost', agent_label)
            self.target_q_cost = QNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions, 'target_q_cost', agent_label)
        else:
            self.q_cost = self.target_q_cost = None
        self.update_target(tau=1.0)

    def _greedy_q(self, q1, q2, qc):
        if self.use_cost_critic:
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
            qc = self.q_cost.forward(state) if self.use_cost_critic else 0.0
            q = self._greedy_q(q1, q2, qc)
        return int(T.argmax(q, dim=1).item())

    def learn(self, states, actions, r1, r2, r_cost, states_, dones):
        dev = self.q_task1.device
        states = T.tensor(states, dtype=T.float).to(dev)
        states_ = T.tensor(states_, dtype=T.float).to(dev)
        actions = T.tensor(actions, dtype=T.long).to(dev)
        r1 = T.tensor(r1, dtype=T.float).to(dev)
        r2 = T.tensor(r2, dtype=T.float).to(dev)
        dones = T.tensor(dones, dtype=T.bool).to(dev)
        if self.hard:
            r_cost = T.tensor(r_cost, dtype=T.float).to(dev)

        self.q_task1.train(); self.q_task2.train()
        self.target_q_task1.eval(); self.target_q_task2.eval()
        if self.use_cost_critic:
            self.q_cost.train(); self.target_q_cost.eval()

        with T.no_grad():
            q1n = self.q_task1.forward(states_)
            q2n = self.q_task2.forward(states_)
            qcn = self.q_cost.forward(states_) if self.use_cost_critic else 0.0
            a_star = T.argmax(self._greedy_q(q1n, q2n, qcn), dim=1)
            ix = a_star.unsqueeze(1)
            q1_next = self.target_q_task1.forward(states_).gather(1, ix).squeeze(1)
            q2_next = self.target_q_task2.forward(states_).gather(1, ix).squeeze(1)
            q1_next[dones] = 0.0
            q2_next[dones] = 0.0
            y1 = r1 + self.gamma * q1_next
            # [RQ1-CMDP] RCPO (raw): fold -lambda*c into the task-2 reward target (reward scale).
            r2_eff = r2 - self.lam * r_cost if (self.hard and self.cost_source == 'raw') else r2
            y2 = r2_eff + self.gamma * q2_next
            if self.use_cost_critic:
                qc_next = self.target_q_cost.forward(states_).gather(1, ix).squeeze(1)
                qc_next[dones] = 0.0
                yc = r_cost + self.gamma * qc_next

        idx = actions.unsqueeze(1)
        q1_pred = self.q_task1.forward(states).gather(1, idx).squeeze(1)
        q2_pred = self.q_task2.forward(states).gather(1, idx).squeeze(1)
        self.q_task1.optimizer.zero_grad(); F.mse_loss(q1_pred, y1).backward(); self.q_task1.optimizer.step()
        self.q_task2.optimizer.zero_grad(); F.mse_loss(q2_pred, y2).backward(); self.q_task2.optimizer.step()
        if self.use_cost_critic:
            qc_pred = self.q_cost.forward(states).gather(1, idx).squeeze(1)
            self.q_cost.optimizer.zero_grad(); F.mse_loss(qc_pred, yc).backward(); self.q_cost.optimizer.step()

        self.update_target()

    def update_target(self, tau=None):
        tau = self.tau if tau is None else tau
        pairs = [(self.q_task1, self.target_q_task1), (self.q_task2, self.target_q_task2)]
        if self.use_cost_critic:
            pairs.append((self.q_cost, self.target_q_cost))
        for online, target in pairs:
            for op, tp in zip(online.parameters(), target.parameters()):
                tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)

    def save_models(self):
        for net in (self.q_task1, self.q_task2, self.target_q_task1, self.target_q_task2):
            net.save_checkpoint()
        if self.use_cost_critic:
            self.q_cost.save_checkpoint(); self.target_q_cost.save_checkpoint()

    def load_models(self):
        for net in (self.q_task1, self.q_task2, self.target_q_task1, self.target_q_task2):
            net.load_checkpoint()
        if self.use_cost_critic:
            self.q_cost.load_checkpoint(); self.target_q_cost.load_checkpoint()
