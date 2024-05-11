import os
import numpy as np


def calc_files(walk_path):
    res = []
    for root, nowdir, nowfiles in os.walk(walk_path):
        # print(nowfiles)
        for file in nowfiles:
            if file.find("data") != -1:
                res.append(root + "/" + file)
    return res


def build_testfile(train_path, add_time=0):
    initial_strs = np.load(train_path + "str.npy")
    living_zones = np.load("./files/OSMFiles/living_zones.npy")
    travel_zones = np.load("./files/OSMFiles/travel_zone.npy")

    left_dict = {}
    right_dict = {}

    cnt = 0
    for zone in living_zones:
        if zone != "" and zone != " ":
            flag = 0
            for str in initial_strs:
                if str.find(zone) != -1:
                    flag = 1
                    break
            if flag == 1:
                if left_dict.get(zone) is None:
                    left_dict[zone] = [cnt]
                else:
                    left_dict[zone].append(cnt)
        cnt += 1
    cnt = 0

    for zone in travel_zones:
        if zone != "" and zone != " ":
            flag = 0
            for str in initial_strs:
                if str.find(zone) != -1:
                    flag = 1
                    break
            if flag == 1:
                if right_dict.get(zone) is None:
                    right_dict[zone] = [cnt]
                else:
                    right_dict[zone].append(cnt)
        cnt += 1

    # print(left, right)

    left_array = np.arange(len(left_dict))
    right_array = np.arange(len(right_dict))
    test_files = []
    test_reduced = []

    if "Beijing" in train_path:
        hour_list = [i for i in range(10, 23)]
    else:
        hour_list = [i for i in range(10, 20)]

    for i in range(1000):
        left_choice = list(left_dict.keys())[np.random.choice(left_array)]
        # print(left_choice)
        right_chocies = []
        # random a int number between [1,3]
        for j in range(np.random.randint(1, 4)):
            right_choice = list(right_dict.keys())[np.random.choice(right_array)]
            right_chocies.append(right_choice)
        right_chocies = list(set(right_chocies))
        now_str = "I live in " + left_choice + "."
        go_str = "I want go to "
        for zone in right_chocies:
            go_str += zone + ", "
        go_str = go_str[:-2] + "."
        test_files.append(now_str + " " + go_str)

        test_reduced.append((left_choice, [zone for zone in right_chocies]))
        # print(test_files[-1])
    return test_files, (test_reduced, left_dict, right_dict)


def myprepare(accelerator, model, optimizer, lr_scheduler):
    model = accelerator.prepare_model(model)
    optimizer = accelerator.prepare_optimizer(optimizer)
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)
    return model, optimizer, lr_scheduler


def to_grid(gps, json_file):
    x_range = json_file["x_range"]
    y_range = json_file["y_range"]
    grid_len = json_file["grid"]
    x_len = int((x_range[1] - x_range[0]) / grid_len)
    y_len = int((y_range[1] - y_range[0]) / grid_len)
    x = int((gps[0] - x_range[0]) / grid_len)
    y = int((gps[1] - y_range[0]) / grid_len)
    # print(gps[0],gps[1])
    return x * y_len + y


def to_gps(grid, json_file):
    x_range = json_file["x_range"]
    y_range = json_file["y_range"]
    grid_len = json_file["grid"]
    x_len = int((x_range[1] - x_range[0]) / grid_len)
    y_len = int((y_range[1] - y_range[0]) / grid_len)
    x = grid // y_len
    y = grid % y_len
    # print([(x + 0.5) * grid_len + x_range[0], (y + 0.5) * grid_len + y_range[0]])
    return [(x + 0.5) * grid_len + x_range[0], (y + 0.5) * grid_len + y_range[0]]


def norm(gps, json_file):
    x_range = json_file["x_range"]
    y_range = json_file["y_range"]
    mid = [(x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2]
    return [(gps[0] - mid[0]) / (x_range[1] - x_range[0]), (gps[1] - mid[1]) / (y_range[1] - y_range[0])]


def renorm(normed, json_file):
    x_range = json_file["x_range"]
    y_range = json_file["y_range"]
    mid = [(x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2]
    return [normed[0] * (x_range[1] - x_range[0]) + mid[0], normed[1] * (y_range[1] - y_range[0]) + mid[1]]


def check(gps, json_file):
    x_range = json_file["x_range"]
    y_range = json_file["y_range"]
    if x_range[0] <= gps[0] and gps[0] < x_range[1] and y_range[0] <= gps[1] and gps[1] < y_range[1]:
        return 1
    return 0


def calculate_metrices(test_reduced, traj):
    text, left_dict, right_dict = test_reduced
    # 住的地方判断对的
    able_live = [[0, 0], [0, 0], [0, 0]]
    # 去的地方判断对的
    able_go = [[0, 0], [0, 0], [0, 0]]
    # 全判断对
    able_both = [[0, 0], [0, 0], [0, 0]]

    for i in range(len(text)):
        match_live = 0
        match_go = 0
        full_state = (2 ** (len(text[i][1]))) - 1

        base = len(text[i][1]) - 1
        able_live[base][1] += 1
        able_go[base][1] += 1
        able_both[base][1] += 1

        for j in range(96):
            if (0 <= j and j < 4 * 8) or (j >= 4 * 22):
                if traj[i][j] in left_dict[text[i][0]] and match_live == 0:
                    match_live = 1
                    able_live[base][0] += 1
            else:
                cnt = 0
                for go_zone in text[i][1]:
                    if traj[i][j] in right_dict[go_zone]:
                        if match_go != full_state and \
                                (match_go | (2 ** cnt)) == full_state:
                            able_go[base][0] += 1
                        match_go |= (2 ** cnt)
                    cnt += 1
        if match_live == 1 and match_go == full_state:
            able_both[base][0] += 1

    print("able_live: ", able_live)
    print("able_go: ", able_go)
    print("able_both: ", able_both)
