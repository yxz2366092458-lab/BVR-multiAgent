import time

from CommunicationTool import *

# 四个飞机的战术阶段
air_state = [1, 1, 1, 1, 1]
# 四个飞机阶段执行时间
air_state_time = [0, 0, 0, 0, 0]
# 四个飞机阶段的目标时间
air_state_time_target = [0, 0, 0, 0, 0]
# 远距离
dis_long = 120000
# 最高高度
height_max = 15000
# 最低高度
height_min = 4000
#威胁高度差
height_menace = 4000
# π
Pi = 3.14

raida_pool = []

# 飞机动作序号
id_act_plane = [0, 0, 0, 0, 0]

#time_shot
time_shot = [0,0,0,0,0]


def main_action(info, step_num, plane):
    output_cmd = SendData()
    idd = info.DroneID
    enemy_attack = info.AttackEnemyList[0]
    enemy_found = info.FoundEnemyList[0]
    dis_min = 99999999
    id = 0
    for i in info.AttackEnemyList:
        if i.TargetDis != 0 and i.TargetDis < dis_min:
            dis_min = i.TargetDis
            id = i.EnemyID
    for i in info.AttackEnemyList:
        if i.EnemyID == id:
            enemy_attack = i
    for i in info.FoundEnemyList:
        if i.EnemyID == id:
            enemy_found = i

    id = id_act_plane[plane]
    id_act_plane[plane] = id_act_plane[plane] + 1

    if enemy_attack.NTSstate == 1:
        # 若未锁定需要通过isNTSAssigned置1和NTSEntityIdAssigned置ID实现更改锁定
        output_cmd.sOtherControl.isLaunch = 0
        output_cmd.sSOCtrl.isNTSAssigned = 1
        output_cmd.sSOCtrl.NTSEntityIdAssigned = enemy_attack.EnemyID


    if air_state[plane] == 1:
        if enemy_attack.TargetDis > dis_long:
            if (enemy_attack.TargetDis > enemy_attack.MissilePowerfulDis*1000 +
                1.8 * (height_max - info.Altitude)):

                output_cmd.sPlaneControl.CmdIndex = id
                output_cmd.sPlaneControl.CmdID = 2
                output_cmd.sPlaneControl.VelType = 0
                output_cmd.sPlaneControl.CmdSpd = 1.2
                output_cmd.sPlaneControl.isApplyNow = True
                output_cmd.sPlaneControl.CmdHeadingDeg = (enemy_found.TargetYaw+info.Yaw) * 180 / Pi
                output_cmd.sPlaneControl.ThrustLimit = 100
            elif (enemy_attack.TargetDis > enemy_attack.MissilePowerfulDis*1000 +
                1.1 * (height_max - info.Altitude)):

                output_cmd.sPlaneControl.CmdIndex = id
                output_cmd.sPlaneControl.CmdID = 3
                output_cmd.sPlaneControl.VelType = 0
                output_cmd.sPlaneControl.CmdSpd = 1.2
                output_cmd.sPlaneControl.isApplyNow = True
                output_cmd.sPlaneControl.CmdHeadingDeg = (enemy_found.TargetYaw+info.Yaw) * 180 / Pi
                output_cmd.sPlaneControl.ThrustLimit = 100
                output_cmd.sPlaneControl.CmdHeadingDeg = 25
            elif (enemy_attack.TargetDis > enemy_attack.MissilePowerfulDis*1000 +
                1.0 * (height_max - info.Altitude)):

                output_cmd.sPlaneControl.CmdIndex = id
                output_cmd.sPlaneControl.CmdID = 2
                output_cmd.sPlaneControl.VelType = 0
                output_cmd.sPlaneControl.CmdSpd = 1.2
                output_cmd.sPlaneControl.isApplyNow = True
                output_cmd.sPlaneControl.CmdHeadingDeg = (enemy_found.TargetYaw+info.Yaw) * 180 / Pi
                output_cmd.sPlaneControl.ThrustLimit = 100
            elif (enemy_attack.TargetDis > enemy_attack.MissilePowerfulDis*1000 or
                 info.Altitude > height_max):

                output_cmd.sPlaneControl.CmdIndex = id
                output_cmd.sPlaneControl.CmdID = 7
                output_cmd.sPlaneControl.VelType = 0
                output_cmd.sPlaneControl.CmdSpd = 1
                output_cmd.sPlaneControl.isApplyNow = True
                output_cmd.sPlaneControl.CmdPitchDeg = -45
                output_cmd.sPlaneControl.CmdHeadingDeg = (enemy_found.TargetYaw+info.Yaw) * 180 / Pi
                output_cmd.sPlaneControl.CmdAlt = 4000
            else:
                if info.Altitude > height_max:

                    output_cmd.sPlaneControl.CmdIndex = id
                    output_cmd.sPlaneControl.CmdID = 7
                    output_cmd.sPlaneControl.VelType = 0
                    output_cmd.sPlaneControl.CmdSpd = 0.95
                    output_cmd.sPlaneControl.isApplyNow = True
                    output_cmd.sPlaneControl.CmdPitchDeg = -30
                    output_cmd.sPlaneControl.CmdHeadingDeg = (enemy_found.TargetYaw+info.Yaw) * 180 / Pi
                    output_cmd.sPlaneControl.CmdAlt = 4000
                elif info.Altitude < height_min:

                    output_cmd.sPlaneControl.CmdIndex = id
                    output_cmd.sPlaneControl.CmdID = 3
                    output_cmd.sPlaneControl.VelType = 0
                    output_cmd.sPlaneControl.CmdSpd = 0.95
                    output_cmd.sPlaneControl.isApplyNow = True
                    output_cmd.sPlaneControl.CmdHeadingDeg = (enemy_found.TargetYaw+info.Yaw) * 180 / Pi
                    output_cmd.sPlaneControl.CmdPitchDeg = 15
                else:

                    output_cmd.sPlaneControl.CmdIndex = id
                    output_cmd.sPlaneControl.CmdID = 2
                    output_cmd.sPlaneControl.VelType = 0
                    output_cmd.sPlaneControl.CmdSpd = 3
                    output_cmd.sPlaneControl.isApplyNow = True
                    output_cmd.sPlaneControl.CmdHeadingDeg = (enemy_found.TargetYaw+info.Yaw) * 180 / Pi
                    output_cmd.sPlaneControl.ThrustLimit = 100
                    output_cmd.sPlaneControl.CmdThrust = 100
        else:
            if info.Altitude > height_max:

                output_cmd.sPlaneControl.CmdIndex = id
                output_cmd.sPlaneControl.CmdID = 7
                output_cmd.sPlaneControl.VelType = 0
                output_cmd.sPlaneControl.CmdSpd = 0.95
                output_cmd.sPlaneControl.isApplyNow = True
                output_cmd.sPlaneControl.CmdPitchDeg = -30
                output_cmd.sPlaneControl.CmdHeadingDeg = (enemy_found.TargetYaw+info.Yaw) * 180 / Pi
                output_cmd.sPlaneControl.CmdAlt = 4000
            elif info.Altitude < height_min:

                output_cmd.sPlaneControl.CmdIndex = id
                output_cmd.sPlaneControl.CmdID = 3
                output_cmd.sPlaneControl.VelType = 0
                output_cmd.sPlaneControl.CmdSpd = 0.95
                output_cmd.sPlaneControl.isApplyNow = True
                output_cmd.sPlaneControl.CmdHeadingDeg = (enemy_found.TargetYaw+info.Yaw) * 180 / Pi
                output_cmd.sPlaneControl.CmdPitchDeg = 15
            else:

                output_cmd.sPlaneControl.CmdIndex = id
                output_cmd.sPlaneControl.CmdID = 2
                output_cmd.sPlaneControl.VelType = 0
                output_cmd.sPlaneControl.CmdSpd = 3
                output_cmd.sPlaneControl.isApplyNow = True
                output_cmd.sPlaneControl.CmdHeadingDeg = (enemy_found.TargetYaw+info.Yaw) * 180 / Pi
                output_cmd.sPlaneControl.ThrustLimit = 100
        for i in info.AlarmList:
            if i.AlarmType == '导弹' and i.MisAzi != 0:
                air_state[plane] = air_state[plane] + 1
                if air_state[plane] > 4:
                    air_state[plane] = 1
    elif air_state[plane] == 2:
        missile_state = 0
        missile_Azi = 0
        for i in info.AlarmList:
            if i.AlarmType == '导弹' and i.MisAzi != 0:
                missile_state = 1
                missile_Azi = i.MisAzi
        if missile_state:

            output_cmd.sPlaneControl.CmdIndex = id
            output_cmd.sPlaneControl.CmdID = 6
            output_cmd.sPlaneControl.VelType = 0
            output_cmd.sPlaneControl.CmdSpd = 6
            output_cmd.sPlaneControl.CmdNy = 8
            output_cmd.sPlaneControl.isApplyNow = True
            a =  (missile_Azi+info.Yaw) * 180 / Pi
            if a <= 90:
                output_cmd.sPlaneControl.CmdHeadingDeg = a + 90
            else:
                output_cmd.sPlaneControl.CmdHeadingDeg = -180 + a - 90
            output_cmd.sPlaneControl.ThrustLimit = 120
            output_cmd.sPlaneControl.CmdThrust = 120
            output_cmd.sPlaneControl.TurnDirection = 1
        else:

            output_cmd.sPlaneControl.CmdIndex = id
            output_cmd.sPlaneControl.CmdID = 6
            output_cmd.sPlaneControl.VelType = 0
            output_cmd.sPlaneControl.CmdSpd = 5
            output_cmd.sPlaneControl.CmdNy = 4
            output_cmd.sPlaneControl.isApplyNow = True
            if (enemy_found.TargetYaw+info.Yaw) * 180 / Pi >= 0:
                output_cmd.sPlaneControl.CmdHeadingDeg = (enemy_found.TargetYaw+info.Yaw) * 180 / Pi - 180
            else:
                output_cmd.sPlaneControl.CmdHeadingDeg = (enemy_found.TargetYaw+info.Yaw) * 180 / Pi + 180
            output_cmd.sPlaneControl.ThrustLimit = 120
            output_cmd.sPlaneControl.TurnDirection = 1
            output_cmd.sPlaneControl.CmdPhi = 78
            output_cmd.sPlaneControl.CmdThrust = 120
        if abs(enemy_found.TargetYaw * 180 / Pi) > 170:
            air_state[plane] = air_state[plane] + 1
            if air_state[plane] > 4:
                air_state[plane] = 1
    elif air_state[plane] == 3:
        missile_state = 0
        missile_Azi = 0
        for i in info.AlarmList:
            if i.AlarmType == '导弹' and i.MisAzi != 0:
                missile_state = 1
                missile_Azi = i.MisAzi
        if missile_state:

            output_cmd.sPlaneControl.CmdIndex = id
            output_cmd.sPlaneControl.CmdID = 6
            output_cmd.sPlaneControl.VelType = 0
            output_cmd.sPlaneControl.CmdSpd = 6
            output_cmd.sPlaneControl.CmdNy = 6
            output_cmd.sPlaneControl.isApplyNow = True
            a = (missile_Azi + info.Yaw) * 180 / Pi
            if a <= 90:
                output_cmd.sPlaneControl.CmdHeadingDeg = a + 90
            else:
                output_cmd.sPlaneControl.CmdHeadingDeg = -180 + a - 90
            output_cmd.sPlaneControl.ThrustLimit = 120
            output_cmd.sPlaneControl.CmdThrust = 120
            output_cmd.sPlaneControl.TurnDirection = 1
        else:

            output_cmd.sPlaneControl.CmdIndex = id
            output_cmd.sPlaneControl.CmdID = 3
            output_cmd.sPlaneControl.VelType = 0
            output_cmd.sPlaneControl.CmdSpd = 3
            output_cmd.sPlaneControl.isApplyNow = True
            output_cmd.sPlaneControl.CmdHeadingDeg = 180
            output_cmd.sPlaneControl.CmdPitchDeg = 15

        if info.Altitude > 3000:
            air_state[plane] = air_state[plane] + 1
            if air_state[plane] > 4:
                air_state[plane] = 1
    elif air_state[plane] == 4:
        missile_state = 0
        missile_Azi = 0
        for i in info.AlarmList:
            if i.AlarmType == '导弹' and i.MisAzi != 0:
                missile_state = 1
                missile_Azi = i.MisAzi
        if missile_state:

            output_cmd.sPlaneControl.CmdIndex = id
            output_cmd.sPlaneControl.CmdID = 6
            output_cmd.sPlaneControl.VelType = 0
            output_cmd.sPlaneControl.CmdSpd = 6
            output_cmd.sPlaneControl.CmdNy = 6
            output_cmd.sPlaneControl.isApplyNow = True
            a = (missile_Azi + info.Yaw) * 180 / Pi
            if a <= 90:
                output_cmd.sPlaneControl.CmdHeadingDeg = a + 90
            else:
                output_cmd.sPlaneControl.CmdHeadingDeg = -180 + a - 90
            output_cmd.sPlaneControl.ThrustLimit = 120
            output_cmd.sPlaneControl.CmdThrust = 120
            output_cmd.sPlaneControl.TurnDirection = 1

        else:

            output_cmd.sPlaneControl.CmdIndex = id
            output_cmd.sPlaneControl.CmdID = 3
            output_cmd.sPlaneControl.VelType = 0
            output_cmd.sPlaneControl.CmdSpd = 3
            output_cmd.sPlaneControl.isApplyNow = True
            output_cmd.sPlaneControl.CmdHeadingDeg = (enemy_found.TargetYaw+info.Yaw) * 180 / Pi
            output_cmd.sPlaneControl.CmdPitchDeg = 15

        if abs(enemy_found.TargetYaw * 180 / Pi) < 15:
            air_state[plane] = air_state[plane] + 1
            if air_state[plane] > 4:
                air_state[plane] = 1
    return output_cmd


def create_action_cmd2(info, step_num, plane):
    state_EnemyList = False
    for i in info.AttackEnemyList:
        if i.EnemyID != 0:
            state_EnemyList = True
            break
    missile_state = 0
    missile_Azi = 0
    output_cmd = SendData()
    for i in info.AlarmList:
        if i.AlarmType == '导弹' and i.MisAzi != 0:
            missile_state = 1
            missile_Azi = i.MisAzi
    if missile_state:
        if abs(missile_Azi) < 80:
            a = (missile_Azi + info.Yaw) * 180 / Pi
            output_cmd.sPlaneControl.CmdIndex = id_act_plane[plane] + 1
            output_cmd.sPlaneControl.CmdID = 6
            output_cmd.sPlaneControl.VelType = 0
            output_cmd.sPlaneControl.CmdSpd = 6
            output_cmd.sPlaneControl.CmdNy = 8
            output_cmd.sPlaneControl.isApplyNow = True

            if a <= 90:
                output_cmd.sPlaneControl.CmdHeadingDeg = a + 90
            else:
                output_cmd.sPlaneControl.CmdHeadingDeg = -180 + a - 90
            output_cmd.sPlaneControl.ThrustLimit = 150
            output_cmd.sPlaneControl.CmdThrust = 150
            output_cmd.sPlaneControl.TurnDirection = 1
        else:
            output_cmd.sPlaneControl.CmdIndex = id_act_plane[plane] + 1
            output_cmd.sPlaneControl.CmdID = 2
            output_cmd.sPlaneControl.VelType = 0
            output_cmd.sPlaneControl.CmdSpd = 6
            output_cmd.sPlaneControl.isApplyNow = True
            output_cmd.sPlaneControl.CmdHeadingDeg = info.Yaw * 180 / Pi
            output_cmd.sPlaneControl.ThrustLimit = 150

    elif not state_EnemyList:
        output_cmd = SendData()
        id = id_act_plane[plane]
        id_act_plane[plane] = id_act_plane[plane] + 1
        output_cmd.sPlaneControl.CmdIndex = id
        output_cmd.sPlaneControl.CmdID = 6
        output_cmd.sPlaneControl.VelType = 0
        output_cmd.sPlaneControl.CmdNy = 4
        output_cmd.sPlaneControl.CmdSpd = 2
        # if (step_num == jidong_time[4]):
        #     output_cmd.sPlaneControl.isApplyNow = False
        output_cmd.sPlaneControl.isApplyNow = True
        output_cmd.sPlaneControl.CmdHeadingDeg = info.Yaw * 180 / Pi + 30
        #output_cmd.sPlaneControl.CmdAlt = 5000
        output_cmd.sPlaneControl.ThrustLimit = 120
        output_cmd.sPlaneControl.TurnDirection = 1
    else:
        state_missile = False
        enemy_attack = 0
        output_cmd.sOtherControl.isLaunch = 0
        id = 0
        enemy_found = 0
        dis_min = 99999999
        for i in info.AttackEnemyList:
            if i.TargetDis != 0 and i.TargetDis < dis_min:
                dis_min = i.TargetDis
                id = i.EnemyID
        for i in info.AttackEnemyList:
            if i.EnemyID == id:
                enemy_attack = i
        for i in info.FoundEnemyList:
            if i.EnemyID == id:
                enemy_found = i
        output_cmd.sOtherControl.isLaunch = 0  # islaunch上升沿导弹发射
        if enemy_attack != 0 and enemy_found != 0:
            if info.MissileNowNum > 0 and enemy_attack.NTSstate == 2 and abs(
                    enemy_found.TargetYaw * 180 / Pi) <= 30 and info.Pitch * 180 / Pi > -10 and enemy_found.TargetDis < enemy_found.AttackAllowDis * 1000 and info.Altitude < height_max and time_shot[plane] + 100 <= id_act_plane[plane]:
                state_missile = True
            if state_missile:
                if enemy_found.TargetDis < enemy_attack.MissilePowerfulDis * 1000:
                    output_cmd.sOtherControl.isLaunch = 1  # islaunch上升沿导弹发射
                    time_shot[plane] = id_act_plane[plane] + 1
                    print(1)
                elif enemy_found.TargetDis < enemy_attack.MissileMaxDis * 1000:
                    output_cmd.sOtherControl.isLaunch = 1
                    time_shot[plane] = id_act_plane[plane] + 1
                    print(2)
        output_cmd = main_action(info, step_num, plane)

    return output_cmd



'''def create_action_cmd(info, step_num):

    if (step_num <= jidong_time[0]):
        output_cmd = SendData()
        output_cmd.sPlaneControl.CmdIndex = 1
        output_cmd.sPlaneControl.CmdID = 1
        if (step_num == jidong_time[0]):
            output_cmd.sPlaneControl.isApplyNow = False
        output_cmd.sPlaneControl.isApplyNow = True
        output_cmd.sPlaneControl.CmdHeadingDeg = 180

        output_cmd.sPlaneControl.CmdAlt = 10000
        output_cmd.sPlaneControl.CmdSpd = 0.9
        output_cmd.sPlaneControl.TurnDirection = 1
        if len(info.AttackEnemyList) != 0:
            if info.AttackEnemyList[0].TargetDis / 1000 <= 
                            info.AttackEnemyList[0].MissilePowerfulDis and info.MissileNowNum > 0:
                # 判断武器是否发射
                output_cmd.sOtherControl.isLaunch = 1
            else:
                # 攻击列表内不存在敌方战机 没有发射导弹
                output_cmd.sOtherControl.isLaunch = 0

                output_cmd.sSOCtrl.NTSEntityIdAssigned = info.AttackEnemyList[0].EnemyID

            if info.AttackEnemyList[0].NTSstate == 0:
                output_cmd.sSOCtrl.isNTSAssigned = 1


    elif (step_num <= jidong_time[1]):
        output_cmd = SendData()
        output_cmd.sPlaneControl.CmdIndex = 1
        output_cmd.sPlaneControl.CmdID = 1
        if (step_num == jidong_time[1]):
            output_cmd.sPlaneControl.isApplyNow = False
        output_cmd.sPlaneControl.isApplyNow = True
        output_cmd.sPlaneControl.CmdHeadingDeg = 180
        output_cmd.sPlaneControl.CmdAlt = 10000
        output_cmd.sPlaneControl.CmdSpd = 0.9
        output_cmd.sPlaneControl.TurnDirection = 1
        # output_tosend = SedTotalCom(output_cmd)
        # init_action_bytes = string_at(addressof(output_tosend), sizeof(output_tosend))
        # sock.send(init_action_bytes)
    elif (step_num <= jidong_time[2]):
        output_cmd = SendData()
        output_cmd.sPlaneControl.CmdIndex = 3
        output_cmd.sPlaneControl.CmdID = 1
        output_cmd.sPlaneControl.VelType = 0
        output_cmd.sPlaneControl.CmdSpd = 0.9
        if (step_num == jidong_time[2]):
            output_cmd.sPlaneControl.isApplyNow = False
        output_cmd.sPlaneControl.isApplyNow = True
        output_cmd.sPlaneControl.CmdHeadingDeg = 180
        output_cmd.sPlaneControl.CmdAlt = 10000
        # output_tosend = SedTotalCom(output_cmd)
        # init_action_bytes = string_at(addressof(output_tosend), sizeof(output_tosend))
        # sock.send(init_action_bytes)
    elif (step_num <= jidong_time[3]):
        output_cmd = SendData()
        output_cmd.sPlaneControl.CmdIndex = 4
        output_cmd.sPlaneControl.CmdID = 7
        output_cmd.sPlaneControl.VelType = 0
        output_cmd.sPlaneControl.CmdSpd = 0.8
        if (step_num == jidong_time[3]):
            output_cmd.sPlaneControl.isApplyNow = False
        output_cmd.sPlaneControl.isApplyNow = True
        output_cmd.sPlaneControl.CmdPitchDeg = -20
        output_cmd.sPlaneControl.CmdHeadingDeg = 180
        output_cmd.sPlaneControl.CmdAlt = 4000
        # output_tosend = SedTotalCom(output_cmd)
        # init_action_bytes = string_at(addressof(output_tosend), sizeof(output_tosend))
        # sock.send(init_action_bytes)
    elif (step_num <= jidong_time[4]):
        output_cmd = SendData()
        output_cmd.sPlaneControl.CmdIndex = 5
        output_cmd.sPlaneControl.CmdID = 1
        output_cmd.sPlaneControl.VelType = 0
        output_cmd.sPlaneControl.CmdSpd = 0.9
        if (step_num == jidong_time[4]):
            output_cmd.sPlaneControl.isApplyNow = False
        output_cmd.sPlaneControl.isApplyNow = True
        output_cmd.sPlaneControl.CmdHeadingDeg = 180
        output_cmd.sPlaneControl.CmdAlt = 5000
        # output_tosend = SedTotalCom(output_cmd)
        # init_action_bytes = string_at(addressof(output_tosend), sizeof(output_tosend))
        # sock.send(init_action_bytes)
    elif (step_num <= jidong_time[5]):
        output_cmd = SendData()
        output_cmd.sPlaneControl.CmdIndex = 6
        output_cmd.sPlaneControl.CmdID = 3
        output_cmd.sPlaneControl.VelType = 0
        output_cmd.sPlaneControl.CmdSpd = 0.9
        if (step_num == jidong_time[5]):
            output_cmd.sPlaneControl.isApplyNow = False
        output_cmd.sPlaneControl.isApplyNow = True
        output_cmd.sPlaneControl.CmdHeadingDeg = 180
        # output_tosend = SedTotalCom(output_cmd)
        # init_action_bytes = string_at(addressof(output_tosend), sizeof(output_tosend))
        # sock.send(init_action_bytes)
    else:
        output_cmd = SendData()
        output_cmd.sPlaneControl.CmdIndex = 7
        output_cmd.sPlaneControl.CmdID = 1
        output_cmd.sPlaneControl.VelType = 0
        output_cmd.sPlaneControl.CmdSpd = 0.9
        # if (step_num == jidong_time[4]):
        #     output_cmd.sPlaneControl.isApplyNow = False
        output_cmd.sPlaneControl.isApplyNow = True
        output_cmd.sPlaneControl.CmdHeadingDeg = 180
        output_cmd.sPlaneControl.CmdAlt = 5000

    return output_cmd


# 规整上升沿
def check_cmd(cmd, last_cmd):
    # if last_cmd is None:
    #     cmd.sPlaneControl.isApplyNow = False
    #     cmd.sOtherControl.isLaunch = 0
    #     cmd.sSOCtrl.isNTSAssigned = 0
    # else:
    #     if cmd.sPlaneControl == last_cmd.sPlaneControl:
    #         cmd.sPlaneControl.isApplyNow = False
    #     if cmd.sSOCtrl == last_cmd.sSOCtrl:
    #         cmd.sSOCtrl.isNTSAssigned = 0
    return cmd'''


# 获取传输数据，生成对应无人机command指令，并传输指令逻辑
def solve(platform, plane):
    global save_last_cmd

    if platform.step > save_last_cmd[plane][1]:
        # if platform.recv_info.MissileTrackList[0].WeaponID != 0:
        #     print(platform.recv_info.DroneID, ":  vars(MissileTrackList[0])", vars(platform.recv_info.MissileTrackList[0]))

        cmd_created = create_action_cmd2(platform.recv_info, platform.step, plane)  # 生成控制指令
        # 保存上一个发送的指令
        save_last_cmd[plane][0] = cmd_created  # 更新保存指令

        # cmd_created = check_cmd(cmd_created, save_last_cmd[plane][0])  # 比较得到上升沿
        platform.cmd_struct_queue.put(cmd_created)  # 发送数据
        save_last_cmd[plane][1] = save_last_cmd[plane][1] + 1


def main(IP, Port, drone_num):
    data_serv = DataService(IP, Port, drone_num)  # 本机IP与设置的端口，使用config文件
    data_serv.run()  # 启动仿真环境

    global save_last_cmd  # 用于比较指令变化的字典全局变量
    global info_last
    info_last = {}
    save_last_cmd = {}

    for plane in data_serv.platforms:  # 初始化全局变量为None
        save_last_cmd[plane] = [None, 0]
        info_last[plane] = [None]

    while True:  # 交互循环
        try:
            for plane in data_serv.platforms:
                # time.sleep(10)
                solve(data_serv.platforms[plane], plane)  # 处理信息
                info_last[plane] = data_serv.platforms[plane]
                #print(plane, "'s step is  ", data_serv.platforms[plane].step)
        except Exception as e:
            print("Error break", e)
            break

    data_serv.close()


if __name__ == "__main__":
    IP = "192.168.110.148"
    Port = 60001
    drone_num = 4
    main(IP, Port, drone_num)
