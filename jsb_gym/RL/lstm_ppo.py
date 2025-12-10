import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.lstm_hx = []
        self.lstm_cx = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.lstm_hx[:]
        del self.lstm_cx[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, NN_conf, use_gpu=False, lstm_hidden_dim=64, lstm_layers=1):
        super(ActorCritic, self).__init__()

        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        # LSTM layer
        self.lstm = nn.LSTM(state_dim, lstm_hidden_dim, lstm_layers, batch_first=True)

        # Actor network
        if NN_conf == 'tanh':
            self.actor = nn.Sequential(
                nn.Linear(lstm_hidden_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        elif NN_conf == 'relu':
            print('ReLU')
            self.actor = nn.Sequential(
                nn.Linear(lstm_hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.ReLU()
            )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.set_device(use_gpu)
        self.action_var = torch.full((action_dim,), action_std * action_std).to(self.device)

    def set_device(self, use_gpu=False):
        if use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory, gready, hx=None, cx=None):
        # Reshape state for LSTM (batch_size, sequence_length, features)
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # Add sequence dimension

        # LSTM forward pass
        if hx is None or cx is None:
            # Initialize hidden states
            batch_size = state.size(0)
            hx = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(self.device)
            cx = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(self.device)

        lstm_out, (hx, cx) = self.lstm(state, (hx, cx))

        # Use the last output from LSTM
        lstm_output = lstm_out[:, -1, :]  # Take the last time step

        action_mean = self.actor(lstm_output)

        if not gready:
            cov_mat = torch.diag(self.action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            action = dist.sample()
            action_logprob = dist.log_prob(action)

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            memory.lstm_hx.append(hx)
            memory.lstm_cx.append(cx)
            return action.detach(), hx, cx
        else:
            return action_mean.detach(), hx, cx

    def evaluate(self, state, action, hx=None, cx=None):
        # Reshape state for LSTM
        if len(state.shape) == 2:
            state = state.unsqueeze(1)

        # LSTM forward pass
        if hx is None or cx is None:
            batch_size = state.size(0)
            hx = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(self.device)
            cx = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(self.device)

        lstm_out, (hx, cx) = self.lstm(state, (hx, cx))
        lstm_output = lstm_out[:, -1, :]

        # Actor forward pass
        action_mean = self.actor(lstm_output)

        # Create covariance matrix
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)

        dist_entropy = dist.entropy()
        state_value = self.critic(lstm_output)

        return action_logprobs, torch.squeeze(state_value), dist_entropy, hx, cx


class PPO:
    def __init__(self, state_dim, action_dim, conf_ppo, use_gpu=False):
        self.lr = conf_ppo['lr']
        self.betas = conf_ppo['betas']
        self.gamma = conf_ppo['gamma']
        self.eps_clip = conf_ppo['eps_clip']
        self.K_epochs = conf_ppo['K_epochs']
        action_std = conf_ppo['action_std']

        self.set_device(use_gpu)

        # Initialize LSTM parameters
        self.lstm_hidden_dim = conf_ppo.get('lstm_hidden_dim', 64)
        self.lstm_layers = conf_ppo.get('lstm_layers', 1)

        self.policy = ActorCritic(
            state_dim, action_dim, action_std,
            NN_conf=conf_ppo['nn_type'],
            use_gpu=use_gpu,
            lstm_hidden_dim=self.lstm_hidden_dim,
            lstm_layers=self.lstm_layers
        ).to(self.device)

        self.policy_old = ActorCritic(
            state_dim, action_dim, action_std,
            NN_conf=conf_ppo['nn_type'],
            use_gpu=use_gpu,
            lstm_hidden_dim=self.lstm_hidden_dim,
            lstm_layers=self.lstm_layers
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.lam_a = conf_ppo['lam_a']
        self.normalize_rewards = conf_ppo['normalize_rewards']
        self.loss_a = 0.0
        self.loss_max = 0.0
        self.loss_min = 0.0

    def set_device(self, use_gpu=True, set_policy=False):
        if use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        if set_policy:
            self.policy.actor.to(self.device)
            self.policy.critic.to(self.device)
            self.policy.action_var.to(self.device)
            self.policy.set_device(self.device)

            self.policy_old.actor.to(self.device)
            self.policy_old.critic.to(self.device)
            self.policy_old.action_var.to(self.device)
            self.policy_old.set_device(self.device)

    def select_action(self, state, memory, gready=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, hx, cx = self.policy_old.act(state, memory, gready)
        return action.cpu().data.numpy().flatten(), hx, cx

    def estimate_action(self, state, action, hx=None, cx=None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
        return self.policy_old.evaluate(state, action, hx, cx)

    def update(self, memory, to_tensor=False, use_gpu=True):
        self.set_device(use_gpu, set_policy=True)

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        if to_tensor:
            memory.states = [torch.FloatTensor(i.reshape(1, -1)).to(self.device) for i in memory.states]
            memory.actions = [torch.FloatTensor(i.reshape(1, -1)).to(self.device) for i in memory.actions]
            memory.logprobs = [torch.FloatTensor(i.reshape(1, -1)).to(self.device) for i in memory.logprobs]

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if self.normalize_rewards:
            rewards = ((rewards - rewards.mean()) / (rewards.std() + 1e-7)).to(self.device)

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy, _, _ = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            MseLoss = 0.5 * self.MseLoss(state_values, rewards)
            loss = (-torch.min(surr1, surr2) + MseLoss - 0.01 * dist_entropy).mean()

            if self.lam_a != 0:
                # Add additional loss terms if needed
                pass

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.loss_a = MseLoss.item()
        self.loss_max = torch.max(advantages).item()
        self.loss_min = torch.min(advantages).item()
