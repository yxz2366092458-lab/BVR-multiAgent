"""
策略模块化文件
包含无人机控制策略的完整实现
"""

import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import socket
from ctypes import string_at, addressof, sizeof

# 假设这些是已定义的结构体和类
# 从原始代码中提取的结构体定义
class MissileTrack:
    def __init__(self):
        self.WeaponID = 0
        self.TargetID = 0
        self.Position = [0.0, 0.0, 0.0]
        self.Velocity = [0.0, 0.0, 0.0]

class AttackEnemy:
    def __init__(self):
        self.EnemyID = 0
        self.TargetDis = 0.0
        self.MissilePowerfulDis = 0.0
        self.NTSstate = 0

class SOCtrl:
    def __init__(self):
        self.NTSEntityIdAssigned = 0
        self.isNTSAssigned = 0

class PlaneControl:
    def __init__(self):
        self.CmdIndex = 0
        self.CmdID = 0
        self.VelType = 0
        self.CmdSpd = 0.0
        self.CmdHeadingDeg = 0.0
        self.CmdAlt = 0.0
        self.CmdPitchDeg = 0.0
        self.TurnDirection = 0
        self.isApplyNow = 0

class OtherControl:
    def __init__(self):
        self.isLaunch = 0

class SendData:
    def __init__(self):
        self.sPlaneControl = PlaneControl()
        self.sOtherControl = OtherControl()
        self.sSOCtrl = SOCtrl()

# 全局变量定义
save_last_cmd: Dict[int, List] = {}
info_last: Dict[int, List] = {}
aircraft_states: Dict[int, Dict] = {}
aircraft_actions: Dict[int, Dict] = {}
jidong_time: List[int] = [100, 200, 300, 400, 500, 600]
aircraft_configs: Dict[int, Dict] = {
    0: {"max_altitude": 10000, "max_speed": 0.9, "target_altitude": 5000},
    1: {"max_altitude": 10000, "max_speed": 0.9, "target_altitude": 5000},
    2: {"max_altitude": 10000, "max_speed": 0.9, "target_altitude": 5000},
    3: {"max_altitude": 10000, "max_speed": 0.9, "target_altitude": 5000}
}
aircraft_action_states: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
aircraft_action_timers: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
aircraft_action_targets: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
aircraft_action_counter: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
aircraft_action_last: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
aircraft_action_current: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
aircraft_action_next: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
aircraft_action_step: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
aircraft_action_last_step: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
aircraft_action_current_step: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
aircraft_action_next_step: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
aircraft_action_last_counter: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}