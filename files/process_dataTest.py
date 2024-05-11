import os
import datetime
import numpy as np
from tqdm import tqdm
import json

from utils.utils import check, to_grid

import statistics


def find_mode(lst):
    return statistics.mode(lst)


def load_config(dataset):
    with open("./config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)

    config = config[dataset]

    return config


def calc_files(walk_path):
    res = []
    for root, nowdir, nowfiles in os.walk(walk_path):
        # print(nowfiles)
        for file in nowfiles:
            if file.find("data") != -1:
                res.append(root + "/" + file)
    return res


def to_gps(gps):
    # recoverse the normalized gps to origin
    bias_x = (117.508217 + 115.416827) / 2
    bias_y = (41.058964 + 39.442078) / 2
    scale_x = (117.508217 - 115.416827) / 2
    scale_y = (41.058964 - 39.442078) / 2

    return (gps[0] * scale_x + bias_x, gps[1] * scale_y + bias_y)


living_zones = np.load("./OSMFiles/living_zones.npy")
travel_zones = np.load("./OSMFiles/travel_zone.npy")

config = load_config("GeoLife")
# files = calc_files("/home/zhangrx/OpenURToy/OpenUR/files/GeoLife/")
files = calc_files("./testFiles")
trajs = np.load(files[0])

res_trajs = []
res_uids = []
res_strs = []
cnt_uid = 0
for traj in trajs:
    flag = 0
    cnt_uid += 1
    live_dict = {}
    goto_dict = {}
    time_dict = {}

    for gps_ in traj:
        gps = to_gps(gps_)
        if check(gps, config):
            pass
        else:
            flag = 1

    if flag == 0:
        cnt = 0
        for gps_  in traj:
            gps = to_gps(gps_)
            now = to_grid(gps, config)
            if cnt < 4 * 10 or cnt > 19 * 8:
                now_pos = living_zones[now]
            else:
                now_pos = travel_zones[now]
            if now_pos == "" or now_pos == " ":
                continue

            if cnt < 4 * 10 or cnt > 22 * 8:
                if now_pos not in live_dict:
                    live_dict[now_pos] = 0
                live_dict[now_pos] += 1
            else:
                if now_pos not in goto_dict:
                    goto_dict[now_pos] = 0
                    time_dict[now_pos] = []
                goto_dict[now_pos] += 1
                time_dict[now_pos].append(cnt // 4)
            cnt += 1

        max_time = 0
        max_zone = 0
        for zone in live_dict.keys():
            if live_dict[zone] > max_time:
                max_time = live_dict[zone]
                max_zone = zone

        if max_time < 4 * 4:
            continue

        go_to_zones = []
        for zone in goto_dict:
            if goto_dict[zone] > 4 * 1 and zone != max_zone:
                go_to_zones.append((zone, find_mode(time_dict[zone])))

        if len(go_to_zones) == 0:
            continue

        now_str = "I live in " + str(max_zone) + "."
        go_str = "I want go to "
        for zone in go_to_zones:
            go_str += f"{str(zone[0])} at {str(zone[1])}, "
        go_str = go_str[:-2] + "."

        res_trajs.append(traj)
        res_uids.append(cnt_uid)
        res_strs.append(now_str + " " + go_str)
        print(res_strs[-1])

np.save("./test-Time/data.npy", np.array(res_trajs))
np.save("./test-Time/uid.npy", np.array(res_uids))
np.save("./test-Time/str.npy", np.array(res_strs))

# print(len(res_trajs))
# print(len(res_uids))
