# BVRGym_PPODogV2.py
import numpy as np

version = 2

# 导弹配置 - 蓝方飞机1导弹
aim1_f16_1 = {'bearing': 0,
              'distance': 110e3,
              'vel': 300,
              'alt': 12e3}

aim2_f16_1 = {'bearing': 0,
              'distance': 110e3,
              'vel': 300,
              'alt': 12e3}

# 导弹配置 - 蓝方飞机2导弹
aim1_f16_2 = {'bearing': 0,
              'distance': 110e3,
              'vel': 300,
              'alt': 12e3}

aim2_f16_2 = {'bearing': 0,
              'distance': 110e3,
              'vel': 300,
              'alt': 12e3}

# 导弹配置 - 红方飞机1导弹
aim1_f16r_1 = {'bearing': 0,
               'distance': 110e3,
               'vel': 300,
               'alt': 12e3}

aim2_f16r_1 = {'bearing': 0,
               'distance': 110e3,
               'vel': 300,
               'alt': 12e3}

# 导弹配置 - 红方飞机2导弹
aim1_f16r_2 = {'bearing': 0,
               'distance': 110e3,
               'vel': 300,
               'alt': 12e3}

aim2_f16r_2 = {'bearing': 0,
               'distance': 110e3,
               'vel': 300,
               'alt': 12e3}

# 分组导弹配置
aim_f16_1 = {'aim1_f16_1': aim1_f16_1, 'aim2_f16_1': aim2_f16_1}
aim_f16_2 = {'aim1_f16_2': aim1_f16_2, 'aim2_f16_2': aim2_f16_2}
aim_f16r_1 = {'aim1_f16r_1': aim1_f16r_1, 'aim2_f16r_1': aim2_f16r_1}
aim_f16r_2 = {'aim1_f16r_2': aim1_f16r_2, 'aim2_f16r_2': aim2_f16r_2}

# 所有导弹配置
aim = {**aim_f16_1, **aim_f16_2}
aimr = {**aim_f16r_1, **aim_f16r_2}

general = {
        'env_name': 'PPODogV2',
        'f16_1_name': 'f16_1',
        'f16_2_name': 'f16_2',
        'f16r_1_name': 'f16r_1',
        'f16r_2_name': 'f16r_2',
        'sim_time_max': 60*16,
        'r_step' : 30,
        'fg_r_step' : 1,
        'missile_idle': False,
        'scale': True,
        'rec':False}

states= {
        'obs_space':8,
        'act_space': 3,
        'update_states_type': 3
}

logs= {'log_path': '/home/edvards/workspace/jsbsim/jsb_gym/logs/BVRGym',
       'save_to':'/home/edvards/workspace/jsbsim/jsb_gym/plots/BVRGym'}

sf = {  'd_min': 20e3,
        'd_max': 120e3,
        't': general['sim_time_max'],
        'mach_max': 2,
        'alt_min': 3e3,
        'alt_max': 12e3,
        'head_min': 0,
        'head_max': 360 }

# 蓝方飞机1初始位置
f16_1 = { 'lat':      58.3,
          'long':     18.0,
          'vel':      350,
          'alt':      10e3,
          'heading' : 0}

# 蓝方飞机2初始位置
f16_2 = { 'lat':      58.2,
          'long':     18.1,
          'vel':      350,
          'alt':      10e3,
          'heading' : 0}

# 红方飞机1初始位置
f16r_1 = { 'lat':      59.0,
           'long':     18.0,
           'vel':      350,
           'alt':      10e3,
           'heading' : 180}

# 红方飞机2初始位置
f16r_2 = { 'lat':      59.1,
           'long':     17.9,
           'vel':      350,
           'alt':      10e3,
           'heading' : 180}
