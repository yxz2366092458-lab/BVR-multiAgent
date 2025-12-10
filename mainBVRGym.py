import argparse, time
from jsb_gym.TAU.config import aim_dog_BVRGym
from jsb_gym.environmets import evasive, bvrdog
import numpy as np
from enum import Enum
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import random, os

class Maneuvers(Enum):
    Evasive = 0
    Crank = 1

def fig1(args):
    # plot figure in the article
    # Import the configuration file for the environment
    # Import Tactical units config files
    from jsb_gym.environmets.config import BVRGym_fig1
    from jsb_gym.TAU.config import aim_evs_BVRGym
    from jsb_gym.TAU.config import f16_evs_BVRGym
    from jsb_gym.utils.tb_logs import Env_logs
    BVRGym_fig1.logs['log_path'] = os.getcwd() + '/jsb_gym/logs/BVRGym'
    BVRGym_fig1.logs['save_to'] = os.getcwd() + '/jsb_gym/plots/BVRGym'

    env = evasive.Evasive(BVRGym_fig1, args, aim_evs_BVRGym, f16_evs_BVRGym)

    logs = Env_logs(BVRGym_fig1)
    # Actions
    # Choose heading , altitude and thrust
    # action[0]: heading [-1, 1]
    # action[1]: altitude [-1, 1]
    # action[2]: thrust [-1, 1]

    action = np.zeros(3)
    Eps = int(2)
    maneuver = Maneuvers.Evasive

    for i_episode in tqdm(range(1, Eps)):
        # state block contains states with respect to each missile
        state_block = env.reset(rand_state_f16 = False, rand_state_aim = False)
        # get the state with respect tp aim1 missile
        state = state_block['aim1']
        # initiate some initail parameters
        done = False
        time_step = 0
        action[0] = args['head']
        action[1] = args['alt']
        action[2] = args['thr']

        # run the simulation
        while not done:
            state_block, reward, done, _ = env.step(action, action_type= maneuver.value)
            state = state_block['aim1']
            logs.record(env)
            time_step +=1

    # Plot the results
    # for the BVR gym
    from jsb_gym.utils.tb_plots import Env_plots

    ap = Env_plots(BVRGym_fig1)
    ap.plot_f16()
    ap.plot_missile()
    ap.plot_env()

def train_BVRGym_PPO1(args):
    # import configuration files for the PPO 
    from jsb_gym.RL.config.ppo_evs_PPO1 import conf_ppo
    from jsb_gym.RL.lstm_ppo import Memory, PPO
    # import conf files for the environment, and tachtical units 
    from jsb_gym.environmets.config import BVRGym_PPO1
    from jsb_gym.TAU.config import aim_evs_BVRGym
    from jsb_gym.TAU.config import f16_evs_BVRGym

    # create the environment 
    env = evasive.Evasive(BVRGym_PPO1, args, aim_evs_BVRGym, f16_evs_BVRGym)
    
    writer = SummaryWriter('runs/' + args['track'] +str('_') +str(args['seed']))
    maneuver = Maneuvers.Evasive

    # some inital values for PPO     
    memory = Memory()
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    ppo = PPO(state_dim, action_dim, conf_ppo)
    
    time_step = 0

    for i_episode in tqdm(range(1, conf_ppo['max_episodes']+1)):
        running_reward = 0.0
        # env.reset(True. True) -> set f16 at random state and missile at random state 
        state_block = env.reset(True, True)
        state = state_block['aim1']
        done = False
        action = np.zeros(3) 
        # max thrust 
        # evasive manuevers usualy are performend with active afterburner
        action[2] = 1
        while not done:
            # action inputs from PPO 
            # heading [-1, 1] altitude [-1, 1] thrust [-1, 1]
            act = ppo.select_action(state, memory)
            action[0] = act[0][0]
            action[1] = act[0][1]
            # use simulation time 
            t1 = env.f16.get_sim_time_sec()
            # get the next state after action 
            state_block, reward, done, _ = env.step(action, action_type= maneuver.value)
            state = state_block['aim1']

            # store rewards 
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            time_step +=1            
            if time_step % conf_ppo['update_timestep'] == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
                        
        writer.add_scalar("Reward", reward, i_episode)
        writer.add_scalar("F16 dead", env.reward_f16_dead, i_episode)
        writer.add_scalar("AIM ground", env.reward_aim_hit_ground, i_episode)
        writer.add_scalar("F16 ground", env.reward_f16_hit_ground, i_episode)
        writer.add_scalar("Lost all", env.reward_all_lost, i_episode)
        writer.add_scalar("Max time", env.reward_max_time, i_episode)
        writer.add_scalar("loss_mse", ppo.loss_a, i_episode)
        writer.add_scalar("loss_adv_max", ppo.loss_max, i_episode)
        writer.add_scalar("loss_adv_min", ppo.loss_min, i_episode)
    
    # save model after training 
    torch.save(ppo.policy.state_dict(), 'jsb_gym/logs/trainM1.pth')


'''def train_BVRGym_PPO1(args):
    # import configuration files for the PPO
    from jsb_gym.RL.config.ppo_evs_PPO1 import conf_ppo
    from jsb_gym.RL.ppo import Memory, PPO
    # import conf files for the environment, and tactical units
    from jsb_gym.environmets.config import BVRGym_PPO1
    from jsb_gym.TAU.config import aim_evs_BVRGym
    from jsb_gym.TAU.config import f16_evs_BVRGym

    # 设置日志路径用于可视化
    if args['vizualize']:
        BVRGym_PPO1.logs['log_path'] = os.getcwd() + '/jsb_gym/logs/BVRGym_train'
        BVRGym_PPO1.logs['save_to'] = os.getcwd() + '/jsb_gym/plots/BVRGym_train'
        # 创建必要的目录
        os.makedirs(BVRGym_PPO1.logs['log_path'], exist_ok=True)
        os.makedirs(BVRGym_PPO1.logs['save_to'], exist_ok=True)

    # create the environment
    env = evasive.Evasive(BVRGym_PPO1, args, aim_evs_BVRGym, f16_evs_BVRGym)

    writer = SummaryWriter('runs/' + args['track'] + str('_') + str(args['seed']))
    maneuver = Maneuvers.Evasive

    # 如果启用可视化，创建日志记录器
    if args['vizualize']:
        from jsb_gym.utils.tb_logs import Env_logs
        logs = Env_logs(BVRGym_PPO1)

    # some initial values for PPO
    memory = Memory()
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    ppo = PPO(state_dim, action_dim, conf_ppo)

    time_step = 0

    for i_episode in tqdm(range(1, conf_ppo['max_episodes'] + 1)):
        running_reward = 0.0
        # env.reset(True, True) -> set f16 at random state and missile at random state
        state_block = env.reset(True, True)
        state = state_block['aim1']
        done = False
        action = np.zeros(3)
        # max thrust
        # evasive maneuvers usually are performed with active afterburner
        action[2] = 1

        # 每隔N个episode进行一次可视化（避免每个episode都可视化影响训练速度）
        visualize_this_episode = args['vizualize'] and (i_episode % 10 == 0)  # 每10个episode可视化一次

        while not done:
            # action inputs from PPO
            # heading [-1, 1] altitude [-1, 1] thrust [-1, 1]
            act = ppo.select_action(state, memory)
            action[0] = act[0]
            action[1] = act[1]

            # get the next state after action
            state_block, reward, done, _ = env.step(action, action_type=maneuver.value)
            state = state_block['aim1']

            # 如果启用可视化且当前episode需要可视化，记录日志
            if visualize_this_episode:
                logs.record(env)

            # store rewards
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            time_step += 1
            if time_step % conf_ppo['update_timestep'] == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward

        # 如果启用了可视化且这是最后一个episode，生成可视化图表
        if args['vizualize'] and i_episode == conf_ppo['max_episodes']:
            from jsb_gym.utils.tb_plots import Env_plots
            ap = Env_plots(BVRGym_PPO1)
            ap.plot_f16()
            ap.plot_missile()
            ap.plot_env()

        writer.add_scalar("Reward", reward, i_episode)
        writer.add_scalar("F16 dead", env.reward_f16_dead, i_episode)
        writer.add_scalar("AIM ground", env.reward_aim_hit_ground, i_episode)
        writer.add_scalar("F16 ground", env.reward_f16_hit_ground, i_episode)
        writer.add_scalar("Lost all", env.reward_all_lost, i_episode)
        writer.add_scalar("Max time", env.reward_max_time, i_episode)
        writer.add_scalar("loss_mse", ppo.loss_a, i_episode)
        writer.add_scalar("loss_adv_max", ppo.loss_max, i_episode)
        writer.add_scalar("loss_adv_min", ppo.loss_min, i_episode)

    # save model after training
    torch.save(ppo.policy.state_dict(), 'jsb_gym/logs/trainM1.pth')'''


def train_BVRGym_PPO2(args):
    # import configuration files for the PPO 
    from jsb_gym.RL.config.ppo_evs_PPO2 import conf_ppo
    from jsb_gym.RL.ppo import Memory, PPO
    # import conf files for the environment, and tachtical units
    from jsb_gym.environmets.config import BVRGym_PPO2
    from jsb_gym.TAU.config import aim_evs_BVRGym
    from jsb_gym.TAU.config import f16_evs_BVRGym

    # create the environment 
    env = evasive.Evasive(BVRGym_PPO2, args, aim_evs_BVRGym, f16_evs_BVRGym)
 
    writer = SummaryWriter('runs/' + args['track'] +str('_') +str(args['seed']))
    
    maneuver = Maneuvers.Evasive   
    memory = Memory()
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    ppo = PPO(state_dim*2, action_dim, conf_ppo)
    
    time_step = 0
    t1 = time.time()

    for i_episode in tqdm(range(1, conf_ppo['max_episodes']+1)):
        running_reward = 0.0
        # random initial position for f16 and missile 
        state_block = env.reset(True, True)
        # merge states from aim1 and aim2 into a single array
        state = np.concatenate((state_block['aim1'][0], state_block['aim2'][0]))
        done = False
        action = np.zeros(3) 
        # max thrust 
        action[2] = 1
        while not done:
            # action[0]: heading [-1, 1]
            # action[1]: altitude [-1, 1]
            # action[2]: thrust [-1, 1]

            act = ppo.select_action(state, memory)
            action[0] = act[0]
            action[1] = act[1]
            # integrate envirnment 
            state_block, reward, done, _ = env.step(action, action_type= maneuver.value)
            state = np.concatenate((state_block['aim1'], state_block['aim2']))

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            time_step +=1            
            if time_step % conf_ppo['update_timestep'] == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
                        
        writer.add_scalar("Reward", reward, i_episode)
        writer.add_scalar("F16 dead", env.reward_f16_dead, i_episode)
        writer.add_scalar("AIM ground", env.reward_aim_hit_ground, i_episode)
        writer.add_scalar("F16 ground", env.reward_f16_hit_ground, i_episode)
        writer.add_scalar("Lost all", env.reward_all_lost, i_episode)
        writer.add_scalar("Max time", env.reward_max_time, i_episode)
        writer.add_scalar("loss_mse", ppo.loss_a, i_episode)
        writer.add_scalar("loss_adv_max", ppo.loss_max, i_episode)
        writer.add_scalar("loss_adv_min", ppo.loss_min, i_episode)

    torch.save(ppo.policy.state_dict(), 'jsb_gym/logs/trainM2.pth')

def train_BVRDog(args):
    # import configuration files for the PPO 
    from jsb_gym.RL.config.ppo_evs_PPO_BVRDog import conf_ppo
    from jsb_gym.RL.lstm_ppo import Memory, PPO
    # import conf files for the environment, and tachtical units
    from jsb_gym.environmets.config import BVRGym_PPODog
    from jsb_gym.TAU.config import f16_dog_BVRGym, aim_dog_BVRGym
    from jsb_gym.utils.tb_logs import Dog_logs

    env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
    logs = Dog_logs(BVRGym_PPODog)
    writer = SummaryWriter('runs/' + args['track'] +str('_') +str(args['seed']))
    
    maneuver = Maneuvers.Evasive
    
    memory = Memory()
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    ppo = PPO(state_dim, action_dim, conf_ppo)
    
    time_step = 0
    t1 = time.time()
    Eps = int(100e3)
    for i_episode in tqdm(range(1, Eps)):
        
        running_reward = 0.0
        state = env.reset()
        done = False
        action = np.zeros(3) 
        # thrust 
        action[2] = 0.0
        print('-'*10)
        while not done:
        
            act = ppo.select_action(state, memory)
            action[0] = act[0]
            action[1] = act[1]
            #random.uniform(-1, 1)
            #action[0] = -0.5
            #action[1] = 0.3

            state, reward, done, _ = env.step(action, action_type= maneuver.value, blue_armed= False, red_armed= False)
            #print(reward)
            logs.record(env)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            time_step +=1            
            if time_step % conf_ppo['update_timestep'] == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward

        #from jsb_gym.utils.tb_plots import Dog_3D_plots
        #ap = Dog_3D_plots(BVRGym_PPODog)
        #ap.show_3D()

        writer.add_scalar("Reward", reward, i_episode)
        writer.add_scalar("F16 dead", env.reward_f16_dead, i_episode)
        writer.add_scalar("AIM ground", env.reward_aim_hit_ground, i_episode)
        writer.add_scalar("F16 ground", env.reward_f16_hit_ground, i_episode)
        writer.add_scalar("Lost all", env.reward_all_lost, i_episode)
        writer.add_scalar("Max time", env.reward_max_time, i_episode)
        writer.add_scalar("loss_mse", ppo.loss_a, i_episode)
        writer.add_scalar("loss_adv_max", ppo.loss_max, i_episode)
        writer.add_scalar("loss_adv_min", ppo.loss_min, i_episode)

    torch.save(ppo.policy.state_dict(), 'jsb_gym/logs/trainDog.pth')


def train_BVRDogV2(args):
    # import configuration files for the MAPPO
    from jsb_gym.RL.config.mappo_evs_MAPPO_BVRDog import conf_mappo
    from jsb_gym.RL.mappo import Memory, MAPPO
    # import conf files for the environment, and tactical units
    from jsb_gym.environmets.config import BVRGym_MAPPODog
    from jsb_gym.TAU.config import f16_dog_BVRGym, aim_dog_BVRGym
    from jsb_gym.utils.tb_logs import Dog_logsV2

    env = bvrdog.BVRDogV2(BVRGym_MAPPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
    logs = Dog_logsV2(BVRGym_MAPPODog)
    writer = SummaryWriter('runs/' + args['track'] + str('_') + str(args['seed']))

    maneuver = Maneuvers.Evasive

    # Create separate memories for each agent (2 agents: blue and red)
    n_agents = 2
    memories = [Memory() for _ in range(n_agents)]

    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]

    # Initialize MAPPO with 2 agents
    mappo = MAPPO(state_dim, action_dim, n_agents, conf_mappo)

    time_step = 0
    t1 = time.time()
    max_episodes = conf_mappo['max_episodes']

    for i_episode in tqdm(range(1, max_episodes + 1)):
        running_reward = 0.0
        state = env.reset()
        done = False

        # Initialize actions for both agents
        blue_action = np.zeros(3)
        red_action = np.zeros(3)
        # Set thrust for both agents
        blue_action[2] = 0.0
        red_action[2] = 0.0

        print('-' * 10)

        while not done:
            # Collect actions from both agents
            global_states = []  # For centralized critic

            # Get action for blue agent (agent 0)
            blue_act = mappo.select_action(0, state, memories[0])
            blue_action[0] = blue_act[0]
            blue_action[1] = blue_act[1]
            global_states.append(state)

            # Get action for red agent (agent 1)
            red_act = mappo.select_action(1, state, memories[1])
            red_action[0] = red_act[0]
            red_action[1] = red_act[1]
            global_states.append(state)

            # Execute actions in environment
            # Blue agent action
            state, reward, done, _ = env.step(blue_action, action_type=maneuver.value,
                                              blue_armed=True, red_armed=True)

            # Record environment state
            logs.record(env)

            # Store experiences for both agents (using shared reward)
            for memory in memories:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

            time_step += 1

            # Update MAPPO at specified intervals
            if time_step % conf_mappo['update_timestep'] == 0:
                # Prepare global states for centralized critic
                global_state_flat = np.concatenate(global_states)
                global_states_list = [global_state_flat for _ in range(len(memories[0].rewards))]
                dones = []
                for memory in memories:
                    dones.extend(memory.is_terminals)

                # Update MAPPO with all agents' memories
                mappo.update(memories, global_states_list,
                             [reward] * len(memories[0].rewards),
                             dones)

                # Clear all memories
                for memory in memories:
                    memory.clear_memory()

                time_step = 0

            running_reward += reward

        # Log training metrics
        writer.add_scalar("Reward", reward, i_episode)
        writer.add_scalar("F16 dead", env.reward_f16_dead, i_episode)
        writer.add_scalar("AIM ground", env.reward_aim_hit_ground, i_episode)
        writer.add_scalar("F16 ground", env.reward_f16_hit_ground, i_episode)
        writer.add_scalar("Lost all", env.reward_all_lost, i_episode)
        writer.add_scalar("Max time", env.reward_max_time, i_episode)
        writer.add_scalar("loss_mse", mappo.loss_a, i_episode)
        writer.add_scalar("loss_adv_max", mappo.loss_max, i_episode)
        writer.add_scalar("loss_adv_min", mappo.loss_min, i_episode)

    # Save trained model
    mappo.save_models('jsb_gym/logs/trainDog_MAPPO.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vizualize", action='store_true', help="Render in FG")
    parser.add_argument("-track", "--track", type = str, help="Tracks: train, test, plot", default=' ')
    parser.add_argument("-head", "--head", type = float, help="f16 action for training", default= 1.0)
    parser.add_argument("-alt", "--alt", type = float, help="f16 action for training", default= 0.2)
    parser.add_argument("-thr", "--thr", type = float, help="f16 action for training", default= 0.0)
    parser.add_argument("-seed", "--seed", type = int, help="radnom seed", default= None)

    args = vars(parser.parse_args())
    if args['seed'] != None:
        torch.manual_seed(args['seed'])
        np.random.seed(args['seed'])

    if args['track'] == 'f1':
        fig1(args)
    elif args['track'] == 't1':
        train_BVRGym_PPO1(args)
    elif args['track'] == 't2':
        train_BVRGym_PPO2(args)
    elif args['track'] == 'dog':
        train_BVRDog(args)
    elif args['track'] == 'dogv2':
        train_BVRDogV2(args)
    else:
        print('Env not defined')

# Generate figure 1-3
# Generate data for f16, aim and env by setting rec to true for the corresponding unit 
# python mainBVRGym.py -track f1 -head 0.0 -alt -1.0 -thr 1.0

# - if you have flightgear installed, you can run following (note addtitional fg directories needed) 
# - terminal 1: fgfs --fdm=null --native-fdm=socket,in,60,,5550,udp --aircraft=gripen --airport=ESSA --multiplay=out,10,127.0.0.1,5000 --multiplay=in,10,127.0.0.1,5001
# - terminal 2: fgfs --fdm=null --native-fdm=socket,in,60,,5551,udp --aircraft=ogel --airport=ESSA --multiplay=out,10,127.0.0.1,5001 --multiplay=in,10,127.0.0.1,5000
# - terminal 3: python mainBVRGym.py -track f1 -head 0.0 -alt -1.0 -thr 1.0 -v

# Generate figure 4 
# training: 
# python mainBVRGym.py -track t1 -seed 1
# plot:
# 
# Generate figure 5 
# training:
# python mainBVRGym.py -track t2 -seed 1
#
#
# Generate figure 6 
# training:
# python mainBVRGym.py -track dog
#
#
