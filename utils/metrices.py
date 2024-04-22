import numpy as np
from tqdm import tqdm


def calc_hard_turns(traj):
    # 计算大于90度的转弯次数
    # 平均一天转9次？
    cnt = 0
    for i in range(len(traj) - 2):
        # use vector dot to calculate the angle
        vec1 = traj[i + 1] - traj[i]
        vec2 = traj[i + 2] - traj[i + 1]
        cos = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if cos < 0:
            cnt += 1
    return cnt


def calc_degrees(traj):
    # 一天转过的度数，应该趋于0最好
    # 平均一天转30度？
    cnt = 0
    for i in range(len(traj) - 2):
        vec1 = traj[i + 1] - traj[i]
        vec2 = traj[i + 2] - traj[i + 1]
        if vec1.dot(vec2) == 0 or np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            continue
        cos = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos = min(cos, 1 - 1e-6)
        cos = max(cos, -1 + 1e-6)
        cnt += np.arccos(cos)
    angle_dist = np.abs(cnt) // 5
    return angle_dist


def calc_distance(traj):
    # 求距离的二范数
    # 平均37km？
    cnt = 0
    for i in range(len(traj) - 1):
        cnt += np.linalg.norm(traj[i + 1] - traj[i])
    dist_dist = np.abs(cnt) // 0.03
    return dist_dist


def calc_stay(traj):
    # 求停留的次数
    # 当停留长度是1h时：平均一天停留3.7次
    # 当停留长度是2.5h时，平均一天只停留1.6次？
    # 当停留长度是22.5j时，平均一天停留0.001次，这个倒没啥问题，因为我们筛掉了不动的人

    cnt = 0
    cnt2 = 0
    pre = None
    for i in range(len(traj)):
        if pre is None:
            cnt2 = 1
            pre = traj[i]
            continue
        if np.linalg.norm(traj[i] - pre) < 0.0001:
            cnt2 += 1
            continue
        else:
            pre = traj[i]
            if cnt2 > 10:
                cnt += 1
            cnt2 = 1
    stay_dist = np.abs(cnt) // 1
    return stay_dist


def calc_abnormal(traj):
    # 求异常点
    cnt = 0
    for i in range(1, len(traj) - 1):
        if np.linalg.norm(traj[i] - traj[i - 1]) > 0.1 and np.linalg.norm(traj[i] - traj[i + 1]) > 0.1:
            cnt += 1
    abnormal_dist = np.abs(cnt) // 1
    return abnormal_dist


def calc_metrices(trajs):
    
    hard_turn_distribution = np.zeros([21])
    angle_distribution = np.zeros([21])
    dist_distribution = np.zeros([21])
    stay_distributin = np.zeros([21])
    error_distribution = np.zeros([21])
    
    for traj_ in trajs:
        traj = traj_
        if traj.shape[0]==2:
            traj = traj.T
        cnt = calc_hard_turns(traj)
        angle_dist = calc_degrees(traj)
        dist_dist = calc_distance(traj)
        stay_dist = calc_stay(traj)
        abnormal_dist = calc_abnormal(traj)
        
        if cnt < 20:
            hard_turn_distribution[cnt] += 1
        else:
            hard_turn_distribution[20] += 1
    
        if angle_dist < 20:
            angle_distribution[int(angle_dist)] += 1
        else:
            angle_distribution[20] += 1
        
        if dist_dist < 20:
            dist_distribution[int(dist_dist)] += 1
        else:
            dist_distribution[20] += 1
            
        if stay_dist < 20:
            stay_distributin[int(stay_dist)] += 1
        else:
            stay_distributin[20] += 1
        
        if abnormal_dist < 20:
            error_distribution[int(abnormal_dist)] += 1
        else:
            error_distribution[20] += 1
    
    return [hard_turn_distribution, angle_distribution, dist_distribution, stay_distributin, error_distribution]

