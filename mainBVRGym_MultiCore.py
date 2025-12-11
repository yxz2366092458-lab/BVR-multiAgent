import argparse, time
from jsb_gym.environmets import evasive, bvrdog
import numpy as np
from enum import Enum
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import multiprocessing
from jsb_gym.RL.ppo import Memory, PPO
from jsb_gym.RL.mappo import Memory, MAPPO
from jsb_gym.environmets.config import BVRGym_PPO1, BVRGym_PPO2, BVRGym_PPODog, BVRGym_MAPPODog
from numpy.random import seed
from jsb_gym.TAU.config import aim_evs_BVRGym, f16_evs_BVRGym, aim_dog_BVRGym, f16_dog_BVRGym

def init_pool():
    seed()

class Maneuvers(Enum):
    Evasive = 0
    Crank = 1

def runPPO(args):
    if args['track'] == 'M1':
        from jsb_gym.RL.config.ppo_evs_PPO1 import conf_ppo
        
        env = evasive.Evasive(BVRGym_PPO1, args, aim_evs_BVRGym, f16_evs_BVRGym)
        torch_save = 'jsb_gym/logs/RL/M1.pth'
        state_scale = 1
    elif args['track'] == 'M2':
        from jsb_gym.RL.config.ppo_evs_PPO2 import conf_ppo
        env = evasive.Evasive(BVRGym_PPO2, args, aim_evs_BVRGym, f16_evs_BVRGym)
        torch_save = 'jsb_gym/logs/RL/M2.pth'
        state_scale = 2
    elif args['track'] == 'Dog':
        from jsb_gym.RL.config.ppo_evs_PPO_BVRDog import conf_ppo
        env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
        torch_save = 'jsb_gym/logs/RL/Dog/'
        state_scale = 1
    elif args['track'] == 'DogR':
        from jsb_gym.RL.config.ppo_evs_PPO_BVRDog import conf_ppo
        env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
        torch_save = 'jsb_gym/logs/RL/DogR.pth'
        state_scale = 1

    
    writer = SummaryWriter('runs/' + args['track'] )
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    memory = Memory()
    ppo = PPO(state_dim*state_scale, action_dim, conf_ppo, use_gpu = False)    
    pool = multiprocessing.Pool(processes=int(args['cpu_cores']), initializer=init_pool)
    
    for i_episode in range(1, args['Eps']+1):
        ppo_policy = ppo.policy.state_dict()    
        ppo_policy_old = ppo.policy_old.state_dict()
        input_data = [(args, ppo_policy, ppo_policy_old, conf_ppo, state_scale )for _ in range(args['cpu_cores'])]
        running_rewards = []
        tb_obs = []
        
        results = pool.map(train, input_data)
        for idx, tmp in enumerate(results):
            #print(tmp[5])
            #writer.add_scalar("running_rewards" + str(idx), tmp[5], i_episode)
            memory.actions.extend(tmp[0])
            memory.states.extend(tmp[1])
            memory.logprobs.extend(tmp[2])
            memory.rewards.extend(tmp[3])
            memory.is_terminals.extend(tmp[4])
            running_rewards.append(tmp[5])
            tb_obs.append(tmp[6])
            
        ppo.set_device(use_gpu=True)
        ppo.update(memory, to_tensor=True, use_gpu= True)
        memory.clear_memory()
        ppo.set_device(use_gpu=False)
        torch.cuda.empty_cache()
        
        writer.add_scalar("running_rewards", sum(running_rewards)/len(running_rewards), i_episode)
        tb_obs0 = None
        for i in tb_obs:
            #print(i)
            if tb_obs0 == None:
                tb_obs0 = i
            else:
                for key in tb_obs0:
                    tb_obs0[key] += i[key]

        nr = len(tb_obs)
        for key in tb_obs0:
            tb_obs0[key] = tb_obs0[key]/nr
            writer.add_scalar(key, tb_obs0[key], i_episode)
        if i_episode % 500 == 0:
            # save 
            torch.save(ppo.policy.state_dict(), torch_save + 'Dog'+str(i_episode) + '.pth')

    pool.close()
    pool.join()
    #torch.save(ppo.policy.state_dict(), torch_save)

def train(args):
    if args[0]['track'] == 'M1':
        env = evasive.Evasive(BVRGym_PPO1, args[0], aim_evs_BVRGym, f16_evs_BVRGym)
    elif args[0]['track'] == 'M2':
        env = evasive.Evasive(BVRGym_PPO2, args[0], aim_evs_BVRGym, f16_evs_BVRGym)
    elif args[0]['track'] == 'Dog':
        env = bvrdog.BVRDog(BVRGym_PPODog, args[0], aim_dog_BVRGym, f16_dog_BVRGym)
    elif args[0]['track'] == 'DogR':
        env = bvrdog.BVRDog(BVRGym_PPODog, args[0], aim_dog_BVRGym, f16_dog_BVRGym)


    maneuver = Maneuvers.Evasive
    memory = Memory()
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    ppo = PPO(state_dim* args[4], action_dim, args[3], use_gpu=False)

    ppo.policy.load_state_dict(args[1])
    ppo.policy_old.load_state_dict(args[2])

    ppo.policy.eval()
    ppo.policy_old.eval()
    running_reward = 0.0

    for i_episode in range(1, args[0]['eps']+1):
        action = np.zeros(3)
        # using string comparison, not the best, that is why I am keeping it short for now
        if args[0]['track'] == 'M1':
            state_block = env.reset(True, True)
            state = state_block['aim1']
            # max thrust 
            action[2] = 1

        elif args[0]['track'] == 'M2':
            state_block = env.reset(True, True)
            state = np.concatenate((state_block['aim1'][0], state_block['aim2'][0]))
            # max thrust 
            action[2] = 1
        
        elif args[0]['track'] == 'Dog':
            state = env.reset()
            # If you activate the afterburner, both aircraft will fall from the sky after 10 min  
            action[2] = 0.0
        
        elif args[0]['track'] == 'DogR':
            state = env.reset()
            # If you activate the afterburner, both aircraft will fall from the sky after 10 min  
            action[2] = 0.0
        
        done = False
        while not done:
            # heading [-1, 1] altitude [-1, 1] thrust [-1, 1]
            act = ppo.select_action(state, memory)
            action[0] = act[0][0]
            action[1] = act[0][1]

            if args[0]['track'] == 'M1':
                state_block, reward, done, _ = env.step(action, action_type= maneuver.value)
                state = state_block['aim1']
            
            elif args[0]['track'] == 'M2':
                state_block, reward, done, _ = env.step(action, action_type= maneuver.value)
                state = np.concatenate((state_block['aim1'], state_block['aim2']))
            
            elif args[0]['track'] == 'Dog':
                state, reward, done, _ = env.step(action, action_type= maneuver.value, blue_armed= True, red_armed= True)

            elif args[0]['track'] == 'DogR':
                state, reward, done, _ = env.step(action, action_type= maneuver.value, blue_armed= False, red_armed= True)


            memory.rewards.append(reward)
            memory.is_terminals.append(done)

        running_reward += reward 
    

    running_reward = running_reward/args[0]['eps']
    # tensorboard 
    if args[0]['track'] == 'M1':
        tb_obs = {}
    elif args[0]['track'] == 'M2':
        tb_obs = {}            
    elif args[0]['track'] == 'Dog':
        tb_obs = get_tb_obs_dog(env)
    elif args[0]['track'] == 'DogR':
        tb_obs = get_tb_obs_dog(env)




    actions = [i.detach().numpy() for i in memory.actions]
    states = [i.detach().numpy() for i in memory.states]
    logprobs = [i.detach().numpy() for i in memory.logprobs]
    rewards = [i for i in memory.rewards]
    #print(rewards)
    is_terminals = [i for i in memory.is_terminals]     
    return [actions, states, logprobs, rewards, is_terminals, running_reward, tb_obs]


def get_tb_obs_dog(env):
    tb_obs = {}
    tb_obs['Blue_ground'] = env.reward_f16_hit_ground
    tb_obs['Red_ground'] = env.reward_f16r_hit_ground
    tb_obs['maxTime'] = env.reward_max_time

    tb_obs['Blue_alive'] = env.f16_alive
    tb_obs['Red_alive'] = env.f16r_alive

    tb_obs['aim1_active'] = env.aim_block['aim1'].active
    tb_obs['aim1_alive'] = env.aim_block['aim1'].alive
    tb_obs['aim1_target_lost'] = env.aim_block['aim1'].target_lost
    tb_obs['aim1_target_hit'] = env.aim_block['aim1'].target_hit

    tb_obs['aim2_active'] = env.aim_block['aim2'].active
    tb_obs['aim2_alive'] = env.aim_block['aim2'].alive
    tb_obs['aim2_target_lost'] = env.aim_block['aim2'].target_lost
    tb_obs['aim2_target_hit'] = env.aim_block['aim2'].target_hit

    tb_obs['aim1r_active'] = env.aimr_block['aim1r'].active
    tb_obs['aim1r_alive'] = env.aimr_block['aim1r'].alive
    tb_obs['aim1r_target_lost'] = env.aimr_block['aim1r'].target_lost
    tb_obs['aim1r_target_hit'] = env.aimr_block['aim1r'].target_hit

    tb_obs['aim2r_active'] = env.aimr_block['aim2r'].active
    tb_obs['aim2r_alive'] = env.aimr_block['aim2r'].alive
    tb_obs['aim2r_target_lost'] = env.aimr_block['aim2r'].target_lost
    tb_obs['aim2r_target_hit'] = env.aimr_block['aim2r'].target_hit

    if env.aim_block['aim1'].target_lost:
        tb_obs['aim1_MD'] = env.aim_block['aim1'].position_tgt_NED_norm

    if env.aim_block['aim2'].target_lost:
        tb_obs['aim2_lost'] = 1
        tb_obs['aim2_MD'] = env.aim_block['aim2'].position_tgt_NED_norm

    if env.aimr_block['aim1r'].target_lost:
        tb_obs['aim1r_lost'] = 1
        tb_obs['aim1r_MD'] = env.aimr_block['aim1r'].position_tgt_NED_norm

    if env.aimr_block['aim2r'].target_lost:
        tb_obs['aim2r_lost'] = 1
        tb_obs['aim2r_MD'] = env.aimr_block['aim2r'].position_tgt_NED_norm

    return tb_obs

def get_tb_obs_dogv2(env):
    tb_obs = {}
    tb_obs['Blue_ground'] = env.reward_f16_hit_ground
    tb_obs['Red_ground'] = env.reward_f16r_hit_ground
    tb_obs['maxTime'] = env.reward_max_time

    tb_obs['f16_1_alive'] = env.f16_1_alive
    tb_obs['f16_2_alive'] = env.f16_2_alive
    tb_obs['f16r_1_alive'] = env.f16r_1_alive
    tb_obs['f16r_2_alive'] = env.f16r_2_alive


    tb_obs['aim1_1_active'] = env.aim_block_1['aim1_f16_1'].active
    tb_obs['aim1_1_alive'] = env.aim_block_1['aim1_f16_1'].alive
    tb_obs['aim1_1_target_lost'] = env.aim_block_1['aim1_f16_1'].target_lost
    tb_obs['aim1_1_target_hit'] = env.aim_block_1['aim1_f16_1'].target_hit

    tb_obs['aim2_1_active'] = env.aim_block_1['aim2_f16_1'].active
    tb_obs['aim2_1_alive'] = env.aim_block_1['aim2_f16_1'].alive
    tb_obs['aim2_1_target_lost'] = env.aim_block_1['aim2_f16_1'].target_lost
    tb_obs['aim2_1_target_hit'] = env.aim_block_1['aim2_f16_1'].target_hit

    tb_obs['aim1_2_active'] = env.aim_block_2['aim1_f16_2'].active
    tb_obs['aim1_2_alive'] = env.aim_block_2['aim1_f16_1'].alive
    tb_obs['aim1_2_target_lost'] = env.aim_block_2['aim1_f16_1'].target_lost
    tb_obs['aim1_2_target_hit'] = env.aim_block_2['aim1_f16_1'].target_hit

    tb_obs['aim2_2_active'] = env.aim_block_2['aim2_f16_2'].active
    tb_obs['aim2_2_alive'] = env.aim_block_2['aim2_f16_2'].alive
    tb_obs['aim2_2_target_lost'] = env.aim_block_2['aim2_f16_2'].target_lost
    tb_obs['aim2_2_target_hit'] = env.aim_block_2['aim2_f16_2'].target_hit

    tb_obs['aim1r_1_active'] = env.aimr_block_1['aim1_f16r_1'].active
    tb_obs['aim1r_alive'] = env.aimr_block_1['aim1_f16r_1'].alive
    tb_obs['aim1r_target_lost'] = env.aimr_block_1['aim1_f16r_1'].target_lost
    tb_obs['aim1r_target_hit'] = env.aimr_block_1['aim1_f16r_1'].target_hit

    tb_obs['aim2r_1_active'] = env.aimr_block_1['aim2_f16r_1'].active
    tb_obs['aim2r_alive'] = env.aimr_block_1['aim2_f16r_1'].alive
    tb_obs['aim2r_target_lost'] = env.aimr_block_1['aim2_f16r_1'].target_lost
    tb_obs['aim2r_target_hit'] = env.aimr_block_1['aim2_f16r_1'].target_hit

    tb_obs['aim1r_2_active'] = env.aimr_block_2['aim1_f16r_2'].active
    tb_obs['aim1r_2_alive'] = env.aimr_block_2['aim1_f16r_2'].alive
    tb_obs['aim1r_2_target_lost'] = env.aimr_block_2['aim1_f16r_2'].target_lost
    tb_obs['aim1r_2_target_hit'] = env.aimr_block_2['aim1_f16r_2'].target_hit

    tb_obs['aim2r_2_active'] = env.aimr_block_2['aim2_f16r_2'].active
    tb_obs['aim2r_2_alive'] = env.aimr_block_2['aim2_f16r_2'].alive
    tb_obs['aim2r_2_target_lost'] = env.aimr_block_2['aim2_f16r_2'].target_lost
    tb_obs['aim2r_2_target_hit'] = env.aimr_block_2['aim2_f16r_2'].target_hit

    if env.aim_block_1['aim1_f16_1'].target_lost:
        tb_obs['aim1_f16_1_MD'] = env.aim_block_1['aim1_f16_1'].position_tgt_NED_norm

    if env.aim_block_1['aim2_f16_1'].target_lost:
        tb_obs['aim2_f16_1_MD'] = env.aim_block_1['aim2_f16_1'].position_tgt_NED_norm

    if env.aim_block_2['aim1_f16_2'].target_lost:
        tb_obs['aim1_f16_2_MD'] = env.aim_block_2['aim1_f16_2'].position_tgt_NED_norm

    if env.aim_block_2['aim2_f16_2'].target_lost:
        tb_obs['aim2_f16_2_MD'] = env.aim_block_2['aim2_f16_2'].position_tgt_NED_norm
    
    if env.aimr_block_1['aim1_f16r_1'].target_lost:
        tb_obs['aim1_f16r_1_lost'] = 1
        tb_obs['aim1_f16r_1_MD'] = env.aimr_block_1['aim1_f16r_1'].position_tgt_NED_norm
    
    if env.aimr_block_1['aim2_f16r_1'].target_lost:
        tb_obs['aim2_f16r_1_lost'] = 1
        tb_obs['aim2_f16r_1_MD'] = env.aimr_block_1['aim2_f16r_1'].position_tgt_NED_norm

    if env.aimr_block_2['aim1_f16r_2'].target_lost:
        tb_obs['aim1_f16r_2_lost'] = 1
        tb_obs['aim1_f16r_2_MD'] = env.aimr_block_2['aim1_f16r_2'].position_tgt_NED_norm

    if env.aimr_block_2['aim2_f16r_2'].target_lost:
        tb_obs['aim2_f16r_2_lost'] = 1
        tb_obs['aim2_f16r_2_MD'] = env.aimr_block['aim2_f16r_2'].position_tgt_NED_norm

    return tb_obs

def runMAPPO(args):
    from jsb_gym.RL.config.mappo_evs_MAPPO_BVRDog import conf_mappo
    env = bvrdog.BVRDogV2(BVRGym_MAPPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
    torch_save = 'jsb_gym/logs/RL/DogMR.pth'
    state_scale = 2

    writer = SummaryWriter('runs/' + args['track'])
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    #memory = Memory()
    #mappo = MAPPO(state_dim * state_scale, action_dim, conf_mappo, use_gpu=False)
    #pool = multiprocessing.Pool(processes=int(args['cpu_cores']), initializer=init_pool)

    for i_episode in range(1, args['Eps']+1):
        # 初始化MAPPO算法
        mappo = MAPPO(state_dim, action_dim, n_agents=2, conf_ppo=conf_mappo, use_gpu=False)
        pool = multiprocessing.Pool(processes=int(args['cpu_cores']), initializer=init_pool)

        for i_episode in range(1, args['Eps'] + 1):
            # 获取当前策略状态字典
            agent_states = []
            agent_old_states = []
            centralized_critic_state = mappo.centralized_critic.state_dict()
            critic_optimizer_state = mappo.critic_optimizer.state_dict()

            for i in range(mappo.n_agents):
                agent_states.append(mappo.agents[i].state_dict())
                agent_old_states.append(mappo.agents_old[i].state_dict())

            input_data = [(args, agent_states, agent_old_states, centralized_critic_state,
                           critic_optimizer_state, conf_mappo, state_scale) for _ in range(args['cpu_cores'])]

            running_rewards = []
            tb_obs_list = []

            # 并行执行训练
            results = pool.map(train_mappo, input_data)

            # 收集所有进程的经验数据
            all_memories = []
            all_global_states = []
            all_shared_rewards = []
            all_dones = []

            for idx, result in enumerate(results):
                memories, global_states, shared_rewards, dones, running_reward, tb_obs = result
                all_memories.append(memories)
                all_global_states.extend(global_states)
                all_shared_rewards.extend(shared_rewards)
                all_dones.extend(dones)
                running_rewards.append(running_reward)
                tb_obs_list.append(tb_obs)

            # 使用收集的数据更新MAPPO模型
            mappo.set_device(use_gpu=True)
            mappo.update(all_memories[0], all_global_states, all_shared_rewards, all_dones, use_gpu=True)
            mappo.set_device(use_gpu=False)
            torch.cuda.empty_cache()

            # 记录训练指标
            writer.add_scalar("running_rewards", sum(running_rewards) / len(running_rewards), i_episode)

            # 合并tensorboard观察值
            tb_obs_avg = {}
            if tb_obs_list:
                for key in tb_obs_list[0]:
                    tb_obs_avg[key] = sum(obs.get(key, 0) for obs in tb_obs_list) / len(tb_obs_list)
                    writer.add_scalar(key, tb_obs_avg[key], i_episode)

            # 定期保存模型
            if i_episode % 500 == 0:
                mappo.save_models(torch_save + 'DogMR_' + str(i_episode) + '.pth')

        pool.close()
        pool.join()


def train_mappo(args):
    """
    为MAPPO训练准备的并行训练函数
    """
    # 解析输入参数
    env_args, agent_states, agent_old_states, centralized_critic_state, \
        critic_optimizer_state, conf_mappo, state_scale = args
    print(env_args)

    # 创建环境
    env = bvrdog.BVRDogV2(BVRGym_MAPPODog, env_args, aim_dog_BVRGym, f16_dog_BVRGym)

    # 初始化MAPPO算法
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    n_agents = 2  # 固定为2个智能体

    mappo = MAPPO(state_dim, action_dim, n_agents=n_agents, conf_ppo=conf_mappo, use_gpu=False)

    # 加载模型状态
    for i in range(n_agents):
        mappo.agents[i].load_state_dict(agent_states[i])
        mappo.agents_old[i].load_state_dict(agent_old_states[i])
    mappo.centralized_critic.load_state_dict(centralized_critic_state)

    # 设置模型为评估模式
    for agent in mappo.agents:
        agent.eval()
    for agent_old in mappo.agents_old:
        agent_old.eval()
    mappo.centralized_critic.eval()

    # 初始化每个智能体的记忆
    memories = [Memory() for _ in range(n_agents)]
    running_reward = 0.0

    # 存储全局状态、共享奖励和终止标志
    global_states = []
    shared_rewards = []
    dones = []

    for i_episode in range(1, env_args['eps'] + 1):
        # 重置环境
        state_dict = env.reset()

        # 提取各个智能体的状态
        agent_states_current = [state_dict[i] for i in range(n_agents)]
        global_state = np.concatenate(agent_states_current)

        # 初始化动作
        actions = [np.zeros(3) for _ in range(n_agents)]
        for i in range(n_agents):
            # 默认不使用加力燃烧室
            actions[i][2] = 0.0

        done = False

        while not done:
            # 存储全局状态
            global_states.append(global_state)

            # 每个智能体选择动作
            agent_actions = []
            for agent_id in range(n_agents):
                action = mappo.select_action(agent_id, agent_states_current[agent_id], memories[agent_id])
                agent_actions.append(action)
                # 将动作映射到环境中（前两个维度是控制输入）
                actions[agent_id][0] = action[0]  # 航向
                actions[agent_id][1] = action[1]  # 高度

            # 执行动作
            if env_args['track'] == 'Dog':
                next_state_dict, reward, done, _ = env.step(
                    actions[0], action_type=0, blue_armed=True, red_armed=True)
            elif env_args['track'] == 'DogR':
                next_state_dict, reward, done, _ = env.step(
                    actions[0], action_type=0, blue_armed=False, red_armed=True)
            next_state_dict, reward, done, _ = env.step(
                actions, action_type=0, blue_armed=True, red_armed=True
            )

            # 更新状态
            agent_states_current = [next_state_dict[i] for i in range(n_agents)]
            next_global_state = np.concatenate(agent_states_current)

            # 所有智能体共享相同的奖励和终止信号
            shared_rewards.append(reward)
            dones.append(done)

            # 为每个智能体存储奖励和终止信号
            for agent_id in range(n_agents):
                memories[agent_id].rewards.append(reward)
                memories[agent_id].is_terminals.append(done)

            global_state = next_global_state

        running_reward += reward

    running_reward = running_reward / env_args['eps']

    # 获取TensorBoard观测值
    tb_obs = get_tb_obs_dogv2(env) if env_args['track'] in ['Dog', 'DogR', 'Dogv2'] else {}

    # 处理记忆数据
    processed_memories = []
    for memory in memories:
        actions = [i.detach().numpy() for i in memory.actions]
        states = [i.detach().numpy() for i in memory.states]
        logprobs = [i.detach().numpy() for i in memory.logprobs]
        rewards = [i for i in memory.rewards]
        is_terminals = [i for i in memory.is_terminals]

        # 创建新的内存对象用于返回
        processed_memory = Memory()
        processed_memory.actions = actions
        processed_memory.states = states
        processed_memory.logprobs = logprobs
        processed_memory.rewards = rewards
        processed_memory.is_terminals = is_terminals
        processed_memories.append(processed_memory)

    return [processed_memories, global_states, shared_rewards, dones, running_reward, tb_obs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vizualize", action='store_true', help="Render in FG")
    parser.add_argument("-track", "--track", type = str, help="Tracks: M1, M2, Dog, DogR, Dogv2", default=' ')
    parser.add_argument("-cpus", "--cpu_cores", type = int, help="Nuber of cores to use", default= None)
    parser.add_argument("-Eps", "--Eps", type = int, help="Nuber of cores to use", default= int(1e3))
    parser.add_argument("-eps", "--eps", type = int, help="Nuber of cores to use", default= 5)
    #parser.add_argument("-seed", "--seed", type = int, help="radnom seed", default= None)
    args = vars(parser.parse_args())

    #if args['seed'] != None:
    #    torch.manual_seed(args['seed'])
    #    np.random.seed(args['seed'])

    if args['track'] == 'Dogv2':
        runMAPPO(args)
    else:
        runPPO(args)

# training: 
# python mainBVRGym_MultiCore.py -track M1  -cpus 10 -Eps 100000 -eps 1
# python mainBVRGym_MultiCore.py -track M2  -cpus 10 -Eps 100000 -eps 1
# python mainBVRGym_MultiCore.py -track Dog -cpus 10 -Eps 10000 -eps 1

