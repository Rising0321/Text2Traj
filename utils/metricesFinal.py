import numpy as np
from tqdm import tqdm


def calc_distance(trajs):
    res = []
    dist_distribution = np.zeros([21])
    for traj in trajs:
        cnt = 0
        for i in range(len(traj) - 1):
            cnt += np.linalg.norm(traj[i + 1] - traj[i])
        res.append(cnt)
        cnt = np.abs(cnt) // 0.03
        cnt = int(cnt)
        if cnt < 20:
            dist_distribution[int(cnt)] += 1
        else:
            dist_distribution[20] += 1
    return np.mean(res), dist_distribution


def calc_stay(trajs):
    res = []
    stay_distribution = np.zeros([21])
    for traj in trajs:
        cnt = 0
        cnt2 = 0
        pre = None
        for i in range(len(traj)):
            if pre is None:
                cnt2 = 1
                pre = traj[i]
                continue
            if np.linalg.norm(traj[i] - pre) < 0.001:
                cnt2 += 1
                continue
            else:
                pre = traj[i]
                if cnt2 > 10:
                    cnt += 1
                cnt2 = 1
        res.append(cnt)
        if cnt < 20:
            stay_distribution[int(cnt)] += 1
        else:
            stay_distribution[20] += 1
    return np.mean(res), stay_distribution


def calc_hard_turns(trajs):
    # 计算大于90度的转弯次数
    # 平均一天转9次？
    res = []
    hard_turn_distribution = np.zeros([21])

    for traj in trajs:
        cnt = 0
        for i in range(len(traj) - 2):
            # use vector dot to calculate the angle
            vec1 = traj[i + 1] - traj[i]
            vec2 = traj[i + 2] - traj[i + 1]
            cos = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            if cos < 0:
                cnt += 1
        res.append(cnt)
        if cnt < 20:
            hard_turn_distribution[int(cnt)] += 1
        else:
            hard_turn_distribution[20] += 1
    return np.mean(res), hard_turn_distribution


def calc_diameter(trajs):
    res = []
    dia_distribution = np.zeros([21])
    for traj in trajs:
        
        from scipy.spatial.distance import pdist
        dist = pdist(traj, metric="euclidean")
        cnt = np.max(dist)
        res.append(cnt)
        
        cnt = np.abs(cnt) // 0.01
        if cnt < 20:
            dia_distribution[int(cnt)] += 1
        else:
            dia_distribution[20] += 1
    return np.mean(res), dia_distribution


def legal(gps):
    return gps[0] > 116.2 and gps[0] < 116.55 and gps[1] > 39.75 and gps[1] < 40.1


def to_cid(gps):
    x_len = (gps[0] - 116.2) // 0.01
    y_len = (gps[1] - 39.75) // 0.01
    return int(x_len * 35 + y_len)


def calc_living(trajs):
    res = np.zeros([4900, 96])
    liv_distribution = np.zeros([21])
    for traj in trajs:
        for i in range(len(traj)):
            if legal(traj[i]):
                res[to_cid(traj[i])][i] += 1
    res = np.mean(res, axis=1)
    for i in range(len(res)):
        cnt = res[i] // 5
        if cnt < 20:
            liv_distribution[int(cnt)] += 1
        else:
            liv_distribution[20] += 1
    return res, liv_distribution


def calc_in(trajs):
    res = np.zeros([4900, 96])
    in_distribution = np.zeros([21])
    for traj in trajs:
        for i in range(len(traj) - 1):
            if legal(traj[i]) and legal(traj[i + 1]):
                if to_cid(traj[i]) != to_cid(traj[i + 1]):
                    res[to_cid(traj[i + 1])][i] += 1
    res = np.mean(res, axis=1)
    for i in range(len(res)):
        cnt = res[i] // 1
        if cnt < 20:
            in_distribution[int(cnt)] += 1
        else:
            in_distribution[20] += 1
    return res, in_distribution


def calc_out(trajs):
    res = np.zeros([4900, 96])
    out_distribution = np.zeros([21])
    for traj in trajs:
        for i in range(len(traj) - 1):
            if legal(traj[i]) and legal(traj[i + 1]):
                if to_cid(traj[i]) != to_cid(traj[i + 1]):
                    res[to_cid(traj[i])][i] += 1
    res = np.mean(res, axis=1)
    for i in range(len(res)):
        cnt = res[i] // 1
        if cnt < 20:
            out_distribution[int(cnt)] += 1
        else:
            out_distribution[20] += 1
    return res, out_distribution


def calc_metrices_Final(trajs, model_name, name):
    # 8/14
    dis, dis_dist = calc_distance(trajs)
    # print("finish1")
    stay, stay_dist = calc_stay(trajs)
    # print("finish2")
    turn, turn_dist = calc_hard_turns(trajs)
    # print("finish3")
    dia, dia_dist = calc_diameter(trajs[:1000])
    # print("finish4")
    liv, liv_dist = calc_living(trajs)
    # print("finish5")
    inf, inf_dist = calc_in(trajs)
    # print("finish6")
    ouf, ouf_dist = calc_out(trajs)
    # print("finish7")
    res = [dis, stay, turn, dia, liv, inf, ouf, dis_dist, stay_dist, turn_dist, dia_dist, liv_dist, inf_dist, ouf_dist]
    # print("hi")
    # res_arr = np.asanyarray(res, dtype=object)
    # np.save(f'./metrices/{model_name}-{name}.npy', res, allow_pickle=True)
    return res
