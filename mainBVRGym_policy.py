import argparse, time
from jsb_gym.environmets import evasive, bvrdog
import numpy as np
from enum import Enum
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import multiprocessing
from jsb_gym.RL.ppo import Memory, PPO
from jsb_gym.environmets.config import BVRGym_PPO1, BVRGym_PPO2, BVRGym_PPODog
from numpy.random import seed
from jsb_gym.TAU.config import aim_evs_BVRGym, f16_evs_BVRGym, aim_dog_BVRGym, f16_dog_BVRGym


def init_pool():
    seed()


class Maneuvers(Enum):
    Evasive = 0
    Crank = 1


# 导入专家策略函数
from Main_policy import create_action_cmd2


def expert_policy(state, env, track, info_dict=None):
    """
    专家策略函数，将当前状态转换为专家策略所需的格式并返回动作

    Args:
        state: 当前环境状态
        env: 环境对象
        track: 训练任务类型
        info_dict: 可选的信息字典，用于传递额外信息
    """
    # 这里需要将强化学习环境的状态转换为专家策略所需的格式
    # 由于两个系统的状态表示不同，这里需要进行适配

    # 为不同任务类型创建默认专家动作
    if track == 'M1' or track == 'M2':
        # 对于规避任务，使用简单的专家策略
        # 这里可以根据需要添加更复杂的逻辑
        expert_action = np.zeros(3)

        # 简单的规避策略：根据威胁方向调整航向
        if info_dict and 'threat_direction' in info_dict:
            threat_dir = info_dict['threat_direction']
            expert_action[0] = np.clip(threat_dir, -1, 1)  # 航向
        else:
            expert_action[0] = np.random.uniform(-0.5, 0.5)  # 随机航向

        expert_action[1] = 0  # 俯仰
        expert_action[2] = 1  # 最大推力

    elif track == 'Dog' or track == 'DogR':
        # 对于空战任务，使用更复杂的专家策略
        expert_action = np.zeros(3)

        # 这里可以集成Main_policy中的复杂逻辑
        # 由于状态表示不同，这里使用简化的专家策略

        # 获取敌机信息
        if hasattr(env, 'get_enemy_info'):
            enemy_info = env.get_enemy_info()
            if enemy_info:
                # 计算相对方位角
                relative_bearing = enemy_info.get('bearing', 0)
                expert_action[0] = np.clip(relative_bearing / 180, -1, 1)

                # 高度管理
                current_alt = state[2] if len(state) > 2 else 0
                if current_alt < 4000:  # 低于最低高度
                    expert_action[1] = 0.5  # 爬升
                elif current_alt > 15000:  # 高于最高高度
                    expert_action[1] = -0.5  # 下降
                else:
                    expert_action[1] = 0

                # 推力管理
                distance = enemy_info.get('distance', float('inf'))
                if distance > 120000:  # 远距离
                    expert_action[2] = 1  # 最大推力
                elif distance < 50000:  # 近距离
                    expert_action[2] = 0.3  # 中等推力
                else:
                    expert_action[2] = 0.7
        else:
            # 默认策略
            expert_action = np.array([0, 0, 0.5])

    else:
        expert_action = np.zeros(3)

    return expert_action


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

    writer = SummaryWriter('runs/' + args['track'] + '_expert')
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    memory = Memory()
    ppo = PPO(state_dim * state_scale, action_dim, conf_ppo, use_gpu=False)

    # 专家融合参数
    expert_weight = 0.3  # 专家策略权重
    rl_weight = 0.7  # 强化学习策略权重
    expert_decay = 0.995  # 专家权重衰减率
    min_expert_weight = 0.1  # 最小专家权重

    pool = multiprocessing.Pool(processes=int(args['cpu_cores']), initializer=init_pool)

    for i_episode in range(1, args['Eps'] + 1):
        ppo_policy = ppo.policy.state_dict()
        ppo_policy_old = ppo.policy_old.state_dict()

        # 动态调整专家权重
        if i_episode > 1000:
            expert_weight = max(min_expert_weight, expert_weight * expert_decay)

        input_data = [(args, ppo_policy, ppo_policy_old, conf_ppo, state_scale, expert_weight) for _ in
                      range(args['cpu_cores'])]
        running_rewards = []
        tb_obs = []

        results = pool.map(train, input_data)
        for idx, tmp in enumerate(results):
            memory.actions.extend(tmp[0])
            memory.states.extend(tmp[1])
            memory.logprobs.extend(tmp[2])
            memory.rewards.extend(tmp[3])
            memory.is_terminals.extend(tmp[4])
            running_rewards.append(tmp[5])
            tb_obs.append(tmp[6])

        ppo.set_device(use_gpu=True)
        ppo.update(memory, to_tensor=True, use_gpu=True)
        memory.clear_memory()
        ppo.set_device(use_gpu=False)
        torch.cuda.empty_cache()

        # 记录训练指标
        avg_reward = sum(running_rewards) / len(running_rewards)
        writer.add_scalar("running_rewards", avg_reward, i_episode)
        writer.add_scalar("expert_weight", expert_weight, i_episode)

        # 合并并记录观测指标
        if tb_obs:
            tb_obs0 = None
            for i in tb_obs:
                if tb_obs0 is None:
                    tb_obs0 = i
                else:
                    for key in tb_obs0:
                        if key in i:
                            tb_obs0[key] += i[key]

            if tb_obs0:
                nr = len(tb_obs)
                for key in tb_obs0:
                    tb_obs0[key] = tb_obs0[key] / nr
                    writer.add_scalar(key, tb_obs0[key], i_episode)

        # 定期保存模型
        if i_episode % 500 == 0:
            if args['track'] == 'Dog':
                torch.save(ppo.policy.state_dict(), torch_save + 'Dog' + str(i_episode) + '.pth')
            else:
                torch.save(ppo.policy.state_dict(), torch_save)

    pool.close()
    pool.join()


def train(args):
    track = args[0]['track']
    expert_weight = args[5]  # 获取专家权重

    if track == 'M1':
        env = evasive.Evasive(BVRGym_PPO1, args[0], aim_evs_BVRGym, f16_evs_BVRGym)
    elif track == 'M2':
        env = evasive.Evasive(BVRGym_PPO2, args[0], aim_evs_BVRGym, f16_evs_BVRGym)
    elif track == 'Dog':
        env = bvrdog.BVRDog(BVRGym_PPODog, args[0], aim_dog_BVRGym, f16_dog_BVRGym)
    elif track == 'DogR':
        env = bvrdog.BVRDog(BVRGym_PPODog, args[0], aim_dog_BVRGym, f16_dog_BVRGym)

    maneuver = Maneuvers.Evasive
    memory = Memory()
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    ppo = PPO(state_dim * args[4], action_dim, args[3], use_gpu=False)

    ppo.policy.load_state_dict(args[1])
    ppo.policy_old.load_state_dict(args[2])

    ppo.policy.eval()
    ppo.policy_old.eval()
    running_reward = 0.0

    # 存储专家动作相关统计
    expert_actions_used = 0
    total_actions = 0

    for i_episode in range(1, args[0]['eps'] + 1):
        action = np.zeros(3)

        if track == 'M1':
            state_block = env.reset(True, True)
            state = state_block['aim1']
            action[2] = 1  # 最大推力
        elif track == 'M2':
            state_block = env.reset(True, True)
            state = np.concatenate((state_block['aim1'][0], state_block['aim2'][0]))
            action[2] = 1  # 最大推力
        elif track == 'Dog' or track == 'DogR':
            state = env.reset()
            action[2] = 0.0  # 关闭加力燃烧室

        done = False
        episode_steps = 0

        while not done:
            episode_steps += 1
            total_actions += 1

            # 获取强化学习动作
            rl_act = ppo.select_action(state, memory)

            # 获取专家动作
            # 这里可以根据需要传递额外的环境信息
            info_dict = {
                'track': track,
                'episode': i_episode,
                'step': episode_steps
            }

            # 如果是空战任务，尝试获取敌机信息
            if track in ['Dog', 'DogR'] and hasattr(env, 'get_simple_enemy_info'):
                enemy_info = env.get_simple_enemy_info()
                if enemy_info:
                    info_dict.update(enemy_info)

            expert_act = expert_policy(state, env, track, info_dict)

            # 融合专家策略和强化学习策略
            use_expert = np.random.random() < expert_weight
            if use_expert:
                # 使用专家动作
                final_action = expert_act.copy()
                expert_actions_used += 1

                # 将专家动作转换为张量并计算log概率
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                expert_action_tensor = torch.FloatTensor(expert_act).unsqueeze(0)

                # 计算专家动作在当前策略下的log概率
                with torch.no_grad():
                    action_mean, action_std = ppo.policy_old(state_tensor)
                    dist = torch.distributions.Normal(action_mean, action_std)
                    log_prob = dist.log_prob(expert_action_tensor).sum(-1)

                # 存储到memory中
                memory.states.append(state_tensor.squeeze(0))
                memory.actions.append(expert_action_tensor.squeeze(0))
                memory.logprobs.append(log_prob)
            else:
                # 使用强化学习动作
                final_action = rl_act.copy()

                # 动作已经由select_action存储到memory中

            # 应用动作
            action[0] = final_action[0]
            action[1] = final_action[1]

            if track == 'M1':
                state_block, reward, done, _ = env.step(action, action_type=maneuver.value)
                state = state_block['aim1']
            elif track == 'M2':
                state_block, reward, done, _ = env.step(action, action_type=maneuver.value)
                state = np.concatenate((state_block['aim1'], state_block['aim2']))
            elif track == 'Dog':
                state, reward, done, _ = env.step(action, action_type=maneuver.value, blue_armed=True, red_armed=True)
            elif track == 'DogR':
                state, reward, done, _ = env.step(action, action_type=maneuver.value, blue_armed=False, red_armed=True)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # 为专家策略提供奖励信号
            if use_expert:
                # 可以在这里添加专家奖励调整
                adjusted_reward = reward * 1.1  # 稍微增加专家动作的奖励
                if len(memory.rewards) > 0:
                    memory.rewards[-1] = adjusted_reward

        running_reward += reward

    running_reward = running_reward / args[0]['eps']

    # 收集TensorBoard观测数据
    if track in ['Dog', 'DogR']:
        tb_obs = get_tb_obs_dog(env)
        # 添加专家策略使用统计
        tb_obs['expert_actions_ratio'] = expert_actions_used / max(total_actions, 1)
        tb_obs['expert_actions_used'] = expert_actions_used
    else:
        tb_obs = {}

    # 准备返回数据
    actions = [i.detach().numpy() for i in memory.actions]
    states = [i.detach().numpy() for i in memory.states]
    logprobs = [i.detach().numpy() for i in memory.logprobs]
    rewards = [i for i in memory.rewards]
    is_terminals = [i for i in memory.is_terminals]

    return [actions, states, logprobs, rewards, is_terminals, running_reward, tb_obs]


def get_tb_obs_dog(env):
    """获取空战任务的TensorBoard观测数据"""
    tb_obs = {}

    # 基础统计
    tb_obs['Blue_ground'] = getattr(env, 'reward_f16_hit_ground', 0)
    tb_obs['Red_ground'] = getattr(env, 'reward_f16r_hit_ground', 0)
    tb_obs['maxTime'] = getattr(env, 'reward_max_time', 0)

    tb_obs['Blue_alive'] = getattr(env, 'f16_alive', 0)
    tb_obs['Red_alive'] = getattr(env, 'f16r_alive', 0)

    # AIM相关统计
    aim_block = getattr(env, 'aim_block', {})
    if 'aim1' in aim_block:
        tb_obs['aim1_active'] = aim_block['aim1'].active
        tb_obs['aim1_alive'] = aim_block['aim1'].alive
        tb_obs['aim1_target_lost'] = aim_block['aim1'].target_lost
        tb_obs['aim1_target_hit'] = aim_block['aim1'].target_hit

    if 'aim2' in aim_block:
        tb_obs['aim2_active'] = aim_block['aim2'].active
        tb_obs['aim2_alive'] = aim_block['aim2'].alive
        tb_obs['aim2_target_lost'] = aim_block['aim2'].target_lost
        tb_obs['aim2_target_hit'] = aim_block['aim2'].target_hit

    # 红色方AIM统计
    aimr_block = getattr(env, 'aimr_block', {})
    if 'aim1r' in aimr_block:
        tb_obs['aim1r_active'] = aimr_block['aim1r'].active
        tb_obs['aim1r_alive'] = aimr_block['aim1r'].alive
        tb_obs['aim1r_target_lost'] = aimr_block['aim1r'].target_lost
        tb_obs['aim1r_target_hit'] = aimr_block['aim1r'].target_hit

    if 'aim2r' in aimr_block:
        tb_obs['aim2r_active'] = aimr_block['aim2r'].active
        tb_obs['aim2r_alive'] = aimr_block['aim2r'].alive
        tb_obs['aim2r_target_lost'] = aimr_block['aim2r'].target_lost
        tb_obs['aim2r_target_hit'] = aimr_block['aim2r'].target_hit

    return tb_obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vizualize", action='store_true', help="Render in FG")
    parser.add_argument("-track", "--track", type=str, help="Tracks: M1, M2, Dog, DogR", default=' ')
    parser.add_argument("-cpus", "--cpu_cores", type=int, help="Number of cores to use", default=None)
    parser.add_argument("-Eps", "--Eps", type=int, help="Total episodes", default=int(1e3))
    parser.add_argument("-eps", "--eps", type=int, help="Episodes per update", default=5)
    parser.add_argument("-expert_weight", "--expert_weight", type=float, help="Initial expert weight", default=0.3)
    parser.add_argument("-min_expert", "--min_expert", type=float, help="Minimum expert weight", default=0.1)

    args = vars(parser.parse_args())
    runPPO(args)