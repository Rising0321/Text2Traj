import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

from torchvision import transforms


def to_gps(gps):
    # recoverse the normalized gps to origin
    bias_x = (117.508217 + 115.416827) / 2
    bias_y = (41.058964 + 39.442078) / 2
    scale_x = (117.508217 - 115.416827) / 2
    scale_y = (41.058964 - 39.442078) / 2

    return (gps[0] * scale_x + bias_x, gps[1] * scale_y + bias_y)


def transfer_gps_to_int(gps):
    return int((gps[0] - 116.2075) * 100) * 30 + int((gps[1] - 39.7523) * 100)


def judge_legal(gps):
    return 116.2075 <= gps[0] < 116.2075 + 0.34 and 39.7523 <= gps[1] < 39.7523 + 0.30


class Trajectories(Dataset):
    def __init__(self,
                 data_path,
                 seq_len: int,
                 epoch: int,
                 text_condition=0):

        now_path = data_path[epoch % int(len(data_path))]

        self.text_condition = text_condition

        self.trajectories = self.init_npy(now_path, 0)
        self.uids = self.init_npy(now_path.replace("data.npy", "uid.npy"), 1)
        if self.text_condition == 1:
            self.strs = self.init_npy(now_path.replace("data.npy", "str.npy"), 2)

        self.seq_len = seq_len
        self.size = len(self.trajectories)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.text_condition == 0:
            return self.uids[index], "", self.trajectories[index]
        return self.uids[index], self.strs[index], self.trajectories[index]

    def init_npy(self, data_path, type=0):
        now_list = np.load(data_path, allow_pickle=True)

        if type == 0:
            temp_list = []
            for traj in now_list:
                flag = 1
                now_traj = []
                for gps in traj:
                    gps = to_gps(gps)
                    if not judge_legal(gps):
                        flag = 0
                        break
                    now_traj.append(transfer_gps_to_int(gps))
                if flag == 1:
                    temp_list.append(now_traj)
            now_list = temp_list

        if type != 2:
            now_data = torch.tensor(np.array(now_list), dtype=float)
        else:
            now_data = now_list

        return now_data
