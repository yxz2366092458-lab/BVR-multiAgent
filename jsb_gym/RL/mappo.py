# MAPPO implementation based on the provided PPO code
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, NN_conf, use_gpu=True):
        super(ActorCritic, self).__init__()
        # Action network (individual policy for each agent)
        if NN_conf == 'tanh':
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
            # Critic network (shared value function using global state)
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        elif NN_conf == 'relu':
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh()  # Actions still need to be in [-1, 1]
            )
            # Critic network
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
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

    def act(self, state, memory, greedy=False):
        action_mean = self.actor(state)
        if not greedy:
            cov_mat = torch.diag(self.action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            return action.detach()
        else:
            return action_mean.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class CentralizedCritic(nn.Module):
    """Centralized critic that takes global state as input"""

    def __init__(self, global_state_dim, NN_conf, use_gpu=True):
        super(CentralizedCritic, self).__init__()
        if NN_conf == 'tanh':
            self.critic = nn.Sequential(
                nn.Linear(global_state_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        elif NN_conf == 'relu':
            self.critic = nn.Sequential(
                nn.Linear(global_state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        self.set_device(use_gpu)

    def set_device(self, use_gpu=False):
        if use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

    def forward(self, global_state):
        return self.critic(global_state)


class MAPPO:
    def __init__(self, state_dim, action_dim, n_agents, conf_ppo, use_gpu=False):
        """
        MAPPO implementation for multi-agent environments

        Args:
            state_dim: Dimension of individual agent's observation
            action_dim: Dimension of individual agent's action
            n_agents: Number of agents in the environment
            conf_ppo: Configuration dictionary for PPO parameters
            use_gpu: Whether to use GPU for training
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = conf_ppo['lr']
        self.betas = conf_ppo['betas']
        self.gamma = conf_ppo['gamma']
        self.eps_clip = conf_ppo['eps_clip']
        self.K_epochs = conf_ppo['K_epochs']
        action_std = conf_ppo['action_std']
        self.use_gpu = use_gpu

        self.set_device(use_gpu)

        # Individual policies for each agent
        self.agents = []
        self.agent_optimizers = []

        for i in range(n_agents):
            agent = ActorCritic(state_dim, action_dim, action_std,
                                NN_conf=conf_ppo['nn_type'], use_gpu=use_gpu).to(self.device)
            self.agents.append(agent)

            optimizer = torch.optim.Adam(agent.parameters(), lr=self.lr, betas=self.betas)
            self.agent_optimizers.append(optimizer)

        # Shared centralized critic
        self.global_state_dim = state_dim * n_agents  # Global state is concatenation of all agent states
        self.centralized_critic = CentralizedCritic(self.global_state_dim,
                                                    NN_conf=conf_ppo['nn_type'],
                                                    use_gpu=use_gpu).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.centralized_critic.parameters(),
                                                 lr=self.lr, betas=self.betas)

        # Create copies for old policies
        self.agents_old = []
        for i in range(n_agents):
            agent_old = ActorCritic(state_dim, action_dim, action_std,
                                    NN_conf=conf_ppo['nn_type'], use_gpu=use_gpu).to(self.device)
            agent_old.load_state_dict(self.agents[i].state_dict())
            self.agents_old.append(agent_old)

        self.MseLoss = nn.MSELoss()
        self.lam_a = conf_ppo['lam_a']
        self.normalize_rewards = conf_ppo['normalize_rewards']

        self.loss_a = 0.0
        self.loss_max = 0.0
        self.loss_min = 0.0

    def set_device(self, use_gpu=True):
        if use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

    def select_action(self, agent_id, state, memory, greedy=False):
        """
        Select action for a specific agent

        Args:
            agent_id: ID of the agent
            state: Local state observation for the agent
            memory: Memory buffer to store experience
            greedy: Whether to sample greedily (for evaluation)
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.agents_old[agent_id].act(state, memory, greedy).cpu().data.numpy().flatten()

    def update(self, memories, global_states, shared_rewards, dones, use_gpu=True):
        """
        Update policies and critic using collected experiences

        Args:
            memories: List of Memory objects, one for each agent
            global_states: List of global states (concatenated observations of all agents)
            shared_rewards: Shared reward for all agents (common reward setting)
            dones: Terminal flags for each timestep
            use_gpu: Whether to use GPU
        """
        self.set_device(use_gpu, set_policy=True)

        n_timesteps = len(shared_rewards)

        # Calculate discounted returns for each agent
        discounted_returns = []
        for i in range(self.n_agents):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(shared_rewards), reversed(dones)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            discounted_returns.append(torch.tensor(rewards, dtype=torch.float32).to(self.device))

        # Normalize rewards if required
        if self.normalize_rewards:
            for i in range(self.n_agents):
                returns = discounted_returns[i]
                discounted_returns[i] = ((returns - returns.mean()) / (returns.std() + 1e-7)).to(self.device)

        # Convert global states to tensor
        global_states_tensor = torch.FloatTensor(np.array(global_states)).to(self.device)
        global_states_tensor = global_states_tensor.view(n_timesteps, -1)

        # Process each agent's memory
        agent_states = []
        agent_actions = []
        agent_logprobs = []

        for i in range(self.n_agents):
            memory = memories[i]
            # Convert to tensors
            states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
            actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
            logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()

            agent_states.append(states)
            agent_actions.append(actions)
            agent_logprobs.append(logprobs)

        # Get state values from centralized critic
        state_values = self.centralized_critic(global_states_tensor).squeeze()

        # Update centralized critic
        critic_loss = self.MseLoss(state_values, discounted_returns[0])  # Use first agent's returns (they're all same)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update each agent's policy
        for agent_id in range(self.n_agents):
            # Evaluate actions using current policy
            logprobs, _, dist_entropy = self.agents[agent_id].evaluate(
                agent_states[agent_id], agent_actions[agent_id])

            # Calculate ratios
            ratios = torch.exp(logprobs - agent_logprobs[agent_id].detach())

            # Calculate advantages using centralized critic
            advantages = discounted_returns[agent_id] - state_values.detach()

            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy

            # Additional action regularization if needed
            if self.lam_a != 0 and len(agent_actions[agent_id]) > 1:
                mu = agent_actions[agent_id][:-1]
                mu_nxt = agent_actions[agent_id][1:]
                action_reg_loss = 0.5 * self.MseLoss(mu_nxt, mu) * self.lam_a
                actor_loss += action_reg_loss

            # Update agent policy
            self.agent_optimizers[agent_id].zero_grad()
            actor_loss.mean().backward()
            self.agent_optimizers[agent_id].step()

            # Update old policy
            self.agents_old[agent_id].load_state_dict(self.agents[agent_id].state_dict())

        # Store loss metrics
        self.loss_a = critic_loss.cpu().data.numpy().flatten()[0]
        self.loss_max = advantages.max().cpu().data.numpy().flatten()[0]
        self.loss_min = advantages.min().cpu().data.numpy().flatten()[0]

    def save_models(self, checkpoint_path):
        """Save all agent models and centralized critic"""
        save_dict = {
            'centralized_critic': self.centralized_critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }

        for i in range(self.n_agents):
            save_dict[f'agent_{i}'] = self.agents[i].state_dict()
            save_dict[f'agent_optimizer_{i}'] = self.agent_optimizers[i].state_dict()
            save_dict[f'agent_old_{i}'] = self.agents_old[i].state_dict()

        torch.save(save_dict, checkpoint_path)

    def load_models(self, checkpoint_path):
        """Load all agent models and centralized critic"""
        checkpoint = torch.load(checkpoint_path)

        self.centralized_critic.load_state_dict(checkpoint['centralized_critic'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        for i in range(self.n_agents):
            self.agents[i].load_state_dict(checkpoint[f'agent_{i}'])
            self.agent_optimizers[i].load_state_dict(checkpoint[f'agent_optimizer_{i}'])
            self.agents_old[i].load_state_dict(checkpoint[f'agent_old_{i}'])
