import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.metrices import calc_metrices


def calc_hard_turns(traj):
    cnt = 0
    for i in range(len(traj) - 2):
        vec1 = traj[i + 1] - traj[i]
        vec2 = traj[i + 2] - traj[i + 1]
        cos = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if cos < 0:
            cnt += 1
    return cnt


def calc_degrees(traj):
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
    return cnt


def calc_distance(traj):
    cnt = 0
    for i in range(len(traj) - 1):
        cnt += np.linalg.norm(traj[i + 1] - traj[i])
    return cnt * 100


def calc_stay(traj):
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
    return cnt


def calc_trajs(trajs):
    cnt = 0
    value1 = 0
    value2 = 0
    value3 = 0
    value4 = 0
    for traj_ in trajs:
        traj = traj_
        cnt += 1
        value = calc_traj(traj)
        value1 += value[0]
        value2 += value[1]
        value3 += value[2]
        value4 += value[3]

    return value1 / cnt, value2 / cnt, value3 / cnt, value4 / cnt


def calc_traj(traj_):
    traj = traj_
    value1 = 0
    value2 = 0
    value3 = 0
    value4 = 0
    value1 += calc_hard_turns(traj)
    value2 += calc_degrees(traj)
    value3 += calc_distance(traj)
    value4 += calc_stay(traj)

    return value1, value2, value3, value4


def transfer_gps_to_int(gps):
    return int((gps[0] - 116.2075) * 100) * 30 + int((gps[1] - 39.7523) * 100)


def transfer_int_to_gps(x):
    return [(x // 30) / 100 + 116.2075, (x % 30) / 100 + 39.7523]


def decodeTrajs(trajs):
    now = []
    for traj in trajs:
        temp = []
        for item in traj:
            temp.append(transfer_int_to_gps(item))
        now.append(temp)
    return np.array(now)


def visual(now_trajs, epoch, name, batch_size, model_name="test0", res_texts=None):
    plt.figure(figsize=(8, 8))
    # 40.02492858828391, 116.21972799736383
    # 39.780822893842036, 116.5699544739144
    # set the boundary using above coordinate

    if len(now_trajs[0].shape) < 2:
        now_trajs = decodeTrajs(now_trajs)

    for i in range(batch_size):
        traj = now_trajs[i]
        if traj.shape[0] == 2:
            traj = traj.T
        assert traj.shape[1] == 2
        plt.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.1)
    values = calc_trajs(now_trajs[:batch_size])
    plt.title(f"turn: %.2lf, deg %.2lf, dis %.2lf, stay %.2lf" % (values[0], values[1], values[2], values[3]))
    plt.tight_layout()
    plt.savefig(f'./visualizations/{model_name}/{epoch}-{name}.png')
    plt.close()

    metrices = calc_metrices(now_trajs[:batch_size])
    np.savetxt(f'./metrices/{model_name}/{epoch}-{name}.txt', metrices)

    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    for i, image in enumerate(now_trajs):
        if i >= 16:
            break
        plt.subplot(4, 4, i + 1)
        if image.shape[0] == 2:
            image = image.T
        assert image.shape[1] == 2
        plt.plot(image[:, 0], image[:, 1], color='blue', alpha=0.1)
        values = calc_traj(image)
        plt.title(f"turn: %d, degres %d, distance %d, stay %d" % (values[0], values[1], values[2], values[3]))
    plt.savefig(f'./visualizations/{model_name}/sample-{epoch}-{name}.png')

    if res_texts is not None:
        with open(f'./visualizations/{model_name}/sample-{epoch}-{name}.txt', 'w') as f:
            for i, res_text in enumerate(res_texts):
                f.write(f"{i}: {res_text}\n")
                if i >= 16:
                    break

    plt.close()
