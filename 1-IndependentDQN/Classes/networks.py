import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    """State -> Q-values over the discrete action set (one head per task).
    Small MLP (LayerNorm), sized for the ~19-d platoon observation: Parvini's
    1024/512 critic was ~54x over-parameterised for this input."""

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, agent_label,
                 chkpt_dir='tmp/dqn'):
        super().__init__()
        self.name = name + '_' + str(agent_label)
        # per-run checkpoint dir via env var so concurrent runs don't race (mirrors the
        # CMDP repo's RQ1_CKPT_SUBDIR convention).
        chkpt_dir = os.environ.get('DQN_CKPT_SUBDIR', chkpt_dir)
        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), chkpt_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_dqn')

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        self.q = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.q(x)                       # (batch, n_actions)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
