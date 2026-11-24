from config import *
import pandas as pd
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import Dataset

FEATURE_COLS  = ['Voltage', 'Current', 'Battery_Temp_degC']

'''
数据加载
'''
def data_load(data_path):
    # 加载.mat数据
    meas = loadmat(data_path, simplify_cells=True)['meas']
    df = pd.DataFrame(data=meas, columns=['Time', 'Voltage', "Current", "Battery_Temp_degC", "Ah"])
    # 去除数据重复的行，然后对剩下的索引重新编号
    data = df.drop_duplicates().reset_index(drop=True)

    # 计算soc列
    soc_0 = 1.0
    Q = 2.9
    data['soc'] = soc_0 + (data['Ah'] - data['Ah'].iloc[0]) / Q
    # print(data.head())
    # data.to_csv(DATASETS_DIR / '25degC_UDDS.csv', index=False)
    # data.to_csv(DATASETS_DIR / '25degC_US06.csv', index=False)

    return data

'''
特征标准化处理
'''
# 采用Z-score标准化，（原值 - 均值） / 标准差
def data_standardize(data):
    mean = data[FEATURE_COLS].mean()
    std = data[FEATURE_COLS].std().replace(0, 1)
    data_copy = data.copy()
    data_copy[FEATURE_COLS] = (data_copy[FEATURE_COLS] - mean) / std

    return data_copy

'''
构造数据集
'''
class MyDataset(Dataset):
    def __init__(self, input, win_size):
        self.win_size = win_size
        # .copy() 保证 numpy 数组可写，否则 from_numpy 会报 non-writable 警告
        self.features = torch.from_numpy(input[FEATURE_COLS ].to_numpy(dtype='float32').copy())
        self.soc = torch.from_numpy(input['soc'].to_numpy(dtype='float32').copy())
    # 计算窗口总数
    def __len__(self):
        # 窗口数 = 总长度 - 窗口大小 + 1；数据不足一个窗口时返回 0
        return max(0, len(self.features) - self.win_size + 1)
    # 返回指定索引的整条滑动窗口
    def __getitem__(self, idx):
        end_idx = idx + self.win_size
        target_features = self.features[idx:end_idx]
        target_soc = self.soc[idx:end_idx]

        return target_features, target_soc