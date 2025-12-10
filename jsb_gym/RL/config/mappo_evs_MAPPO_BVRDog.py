# MAPPO training configuration based on PPO template
conf_mappo = {
    'max_episodes': 300000,
    'update_timestep': 60 * 10,   # max time env_evs 180 sec, 60 -> one episode of actions
    'action_std': 0.1,
    'K_epochs': 80,               # update policy for K epochs
    'eps_clip': 0.2,              # clip parameter for PPO
    'gamma': 1.0,                 # discount factor
    'lr': 1e-5,                   # parameters for Adam optimizer
    'critic_lr': 1e-5,            # learning rate for centralized critic
    'betas': (0.9, 0.999),
    'random_seed': None,
    'lam_a': 0,
    'normalize_rewards': True,
    'nn_type': 'tanh',
    'entropy_coef': 0.01,         # entropy coefficient for exploration
    'value_loss_coef': 0.5,       # value loss coefficient
    'max_grad_norm': 0.5,         # gradient clipping norm
    'share_policy': False,        # whether to share policy among agents
    'use_centralized_critic': True,  # use centralized critic for training
    'buffer_size': 5000,          # replay buffer size
    'batch_size': 32,             # training batch size
    'mini_batch_size': 16,        # mini batch size for updates
    'episode_length': 200,        # maximum length of episode
    'num_env_steps': 10000000,    # total number of environment steps
    'train_interval': 5,          # update interval
}
