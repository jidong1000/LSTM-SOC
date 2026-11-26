# 基于 RNN、LSTM 和 GRU 的锂电池 SOC 估计

本项目用电压、电流和电池温度估计 Panasonic 18650PF 电芯的荷电状态（SOC）。RNN、LSTM 和 GRU 均使用 25 ℃ 的 UDDS 工况训练，在 US06 上比较模型配置，并用 LA92 复评跨工况表现。

当前已完成 GRU V1–V5 的 US06 配置试验和 LSTM V7 复评，并在 US06 与 LA92 上对齐比较了 RNN V2、LSTM V6/V7 和 GRU V5。 [原始日志](logs/)

## 数据与预处理

| 用途 | 25 ℃ 工况 | 原始文件 |
| --- | --- | --- |
| 训练 | UDDS | `03-21-17_00.29 25degC_UDDS_Pan18650PF.mat` |
| 参数比较 | US06 | `03-20-17_01.43 25degC_US06_Pan18650PF.mat` |
| 跨工况复评 | LA92 | `03-21-17_09.38 25degC_LA92_Pan18650PF.mat` |

原始数据由 Phillip Kollmeyer 发布。将解压后的数据放入项目根目录的 `datasets/`，脚本会按文件名递归查找相应的 `.mat` 文件。该目录通过 `.gitignore` 管理，克隆项目后需要另行准备数据。 [Panasonic 18650PF 原始数据集](https://data.mendeley.com/datasets/wykht8y7tg/1)

1. **读取数据和计算参考 SOC。** 从 `.mat` 的 `meas` 读取 `Time`、`Voltage`、`Current`、`Battery_Temp_degC` 和 `Ah`，去掉重复行。假设文件开始时为满电，按 `SOC(t) = 1 + (Ah(t) - Ah(0)) / 2.9` 计算标签，配置容量为 `2.9 Ah`。例如 US06 的 `Ah` 从 `0` 变到 `-2.58596`，按这个口径末尾 SOC 约为 `10.83%`。
2. **标准化输入。** 按电压、电流、温度的顺序组织三列特征，并使用 UDDS 的均值和标准差做 Z-score 标准化。评估时从权重文件读取这组统计量，用于处理 US06 或 LA92。SOC 标签以 0–1 比例参与计算。
3. **构造滑动窗口。** 每次向后移动一个采样点。去重后有 `N` 个采样点时，长度为 `W` 的完整窗口共有 `N - W + 1` 个。采样间隔约为 0.1 秒，所以 100 点约覆盖 10 秒，500 点约覆盖 50 秒。 [数据加载与窗口构造](src/dataset.py)

## 训练与评估

三种模型分别使用 PyTorch 的 `nn.RNN`、`nn.LSTM` 和 `nn.GRU`，再用全连接层把每个时间步的隐藏状态映射为一个 SOC 预测。训练与评估都取**窗口最后一个时间点**的 SOC，与同一时刻的参考值比较。 [模型定义](src/model.py) · [网络结构示意图](网络结构图.png) · [架构对比图](比较.png)

- 三种模型共用 `train_sequence_model`，使用 Adam 和 MSE 损失，每批更新一次参数。新权重文件保存模型类型、模型结构参数、训练后的权重、输入特征顺序、标准化统计量和窗口长度。 [训练脚本](src/train.py)
- 评估入口支持 RNN、LSTM 和 GRU。脚本从权重文件读取模型结构和预处理参数，按时间顺序遍历完整测试集一次。逐批累计每个窗口的误差，再汇总 MSE、MAE、RMSE 和最大绝对误差，最后一个不足批大小的批次也参与统计。 [评估脚本](src/evaluate.py)
- MSE 使用 0–1 SOC 比例的平方误差。本文采用**SOC 百分点**表示 MAE、RMSE 和最大绝对误差，对应日志里乘以 100 后带 `%` 的数值。例如参考 SOC 为 80%、预测为 72%，绝对误差为 8 个百分点。

## RNN 调参过程

1. **建立 V1 基线。** 用 30 点窗口、单层 RNN 和隐藏维度 16 在 UDDS 上训练，权重转到 US06 评估后，MAE 为 8.85 个百分点。后续参数试验沿用这两个工况。
2. **把窗口拉长到 100 点。** V2 采用 100 点、单层和隐藏维度 16，US06 的 MAE 降到 3.09。过程中也试过批大小 16。同一份 V2 权重分别用评估批大小 16 和 32 运行，四项指标在日志的显示精度下一致。训练批大小会影响每轮的 Adam 更新次数。
3. **比较层数和隐藏维度。** 100 点窗口下，早期双层、隐藏维度 16 的试验得到 MSE 0.004613、MAE 5.18、RMSE 6.79、最大绝对误差 36.66。现有归档里的 V3 和 V4 是单层模型，分别把宽度改为 8 和 32。两版在 US06 上的平均误差都高于 V2。
4. **再比较其他窗口长度。** V5 使用 120 点窗口，V6 使用 80 点窗口。V5 的日志指标与归档复评有差异，见表后说明。以 US06 的 MAE 和 RMSE 为主要选型指标，最终采用 V2。

历史试验中批大小和训练轮数也有调整。下表保留日志中的批大小字段，模型结构和窗口长度另用归档权重核对。

## US06 实验记录

各版本按自己的窗口长度统计全部有效窗口。LSTM V7 的指标来自当前权重复评，其余行记录原始日志中的指标。MAE、RMSE 和最大绝对误差的单位都是 SOC 百分点。

### RNN

| 日志版本 | 窗口点数 | RNN 层数 | 隐藏维度 | 日志批大小 | MSE | MAE | RMSE | 最大绝对误差 | 归档权重 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [V1](logs/version1.txt) | 30 | 1 | 16 | 32 | 0.012901 | 8.85 | 11.36 | 36.92 | [rnn_soc.pth](models/RNN/rnn_soc.pth) |
| [V2](logs/version2.txt) | 100 | 1 | 16 | 32 | **0.001672** | **3.09** | **4.09** | 33.47 | [rnn_soc_2.pth](models/RNN/rnn_soc_2.pth) |
| [V3](logs/version3.txt) | 100 | 1 | 8 | 32 | 0.005461 | 5.69 | 7.39 | 49.23 | [rnn_soc_3.pth](models/RNN/rnn_soc_3.pth) |
| [V4](logs/version4.txt) | 100 | 1 | 32 | 32 | 0.003871 | 4.43 | 6.22 | 36.30 | [rnn_soc_4.pth](models/RNN/rnn_soc_4.pth) |
| [V5](logs/version5.txt) | 120 | 1 | 16 | 32 | 0.003512 | 4.80 | 5.93 | 31.21 | [rnn_soc_5.pth](models/RNN/rnn_soc_5.pth) |
| [V6](logs/version6.txt) | 80 | 1 | 16 | 32 | 0.021434 | 11.55 | 14.64 | 45.94 | [rnn_soc_6.pth](models/RNN/rnn_soc_6.pth) |

**RNN 归档核对。** V1、V2、V3、V4 和 V6 用对应权重复评后，窗口长度和四项指标均与日志一致。V5 的日志和归档权重都记录为 120 点，但指标有差异。用当前归档权重复评得 **47,941** 个窗口，MSE **0.003330**、MAE **4.59**、RMSE **5.77**、最大绝对误差 **46.95**。上表保留日志记录。

### LSTM

| 日志版本 | 窗口点数 | 层数 | 隐藏维度 | MSE | MAE | RMSE | 最大绝对误差 | 归档权重 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [V1](logs/lstm_version1.txt) | 100 | 1 | 16 | 0.054860 | 19.44 | 23.42 | 61.53 | [lstm_soc_1.pth](models/lstm/lstm_soc_1.pth) |
| [V2](logs/lstm_version2.txt) | 100 | 1 | 32 | 0.048855 | 20.03 | 22.10 | 42.88 | [lstm_soc_2.pth](models/lstm/lstm_soc_2.pth) |
| [V3](logs/lstm_version3.txt) | 150 | 2 | 32 | 0.011882 | 9.41 | 10.90 | 26.81 | [lstm_soc_3.pth](models/lstm/lstm_soc_3.pth) |
| [V4](logs/lstm_version4.txt) | 150 | 2 | 64 | 0.018346 | 11.66 | 13.54 | 31.98 | [lstm_soc_4.pth](models/lstm/lstm_soc_4.pth) |
| [V5](logs/lstm_version5.txt) | 200 | 2 | 64 | 0.007001 | 7.28 | 8.37 | 33.76 | [lstm_soc_5.pth](models/lstm/lstm_soc_5.pth) |
| [V6](logs/lstm_version6.txt) | 300 | 2 | 64 | 0.003423 | 4.96 | 5.85 | 23.54 | [lstm_soc_6.pth](models/lstm/lstm_soc_6.pth) |
| V7（复评） | 500 | 4 | 128 | **0.002208** | **4.16** | **4.70** | **14.06** | [lstm_soc_7.pth](models/lstm/lstm_soc_7.pth) |

V1–V6 归档权重的窗口长度、模型结构和四项指标均与对应日志一致。V7 使用当前评估入口在 US06 上复评，共统计 47,561 个窗口。

### GRU

| 日志版本 | 窗口点数 | 层数 | 隐藏维度 | MSE | MAE | RMSE | 最大绝对误差 | 归档权重 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [V1](logs/gru_version1.txt) | 300 | 2 | 64 | 0.006937 | 7.58 | 8.33 | 18.85 | [gru_soc_1.pth](models/gru/gru_soc_1.pth) |
| [V2](logs/gru_version2.txt) | 300 | 2 | 128 | 0.005003 | 5.56 | 7.07 | 23.69 | [gru_soc_2.pth](models/gru/gru_soc_2.pth) |
| [V3](logs/gru_version3.txt) | 350 | 3 | 128 | 0.004424 | 5.66 | 6.65 | 19.55 | [gru_soc_3.pth](models/gru/gru_soc_3.pth) |
| [V4](logs/gru_version4.txt) | 400 | 4 | 128 | 0.007149 | 6.29 | 8.45 | 26.66 | [gru_soc_4.pth](models/gru/gru_soc_4.pth) |
| [V5](logs/gru_version5.txt) | 500 | 4 | 128 | **0.002825** | **4.12** | **5.31** | **14.14** | [gru_soc_5.pth](models/gru/gru_soc_5.pth) |

五份 GRU 权重中的窗口长度、层数和隐藏维度均与日志一致。V4 和 V5 同为 4 层、隐藏维度 128。V5 将窗口从 400 点增至 500 点，US06 的 MAE 从 6.29 降至 4.12 个百分点。V5 权重在 US06 上复评的四项指标与日志一致。

## 代表权重的对齐复评

选取 US06 日志中 MAE 最低的 RNN V2 和 GRU V5，以及 LSTM V6 和 V7。四份权重各自使用训练时的窗口长度，统一从去重数据的第 500 个采样点开始统计，保证每个预测对应同一时刻的参考 SOC。上方实验记录按各版本的全部有效窗口统计。

| 25 ℃ 工况 | 模型 | 对齐后窗口数 | MSE | MAE（百分点） | RMSE（百分点） | 最大绝对误差（百分点） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| US06 | RNN V2 | 47,561 | **0.001684** | **3.10** | **4.10** | 33.47 |
| US06 | LSTM V6 | 47,561 | 0.003437 | 4.98 | 5.86 | 23.54 |
| US06 | LSTM V7 | 47,561 | 0.002208 | 4.16 | 4.70 | **14.06** |
| US06 | GRU V5 | 47,561 | 0.002825 | 4.12 | 5.31 | 14.14 |
| LA92 | RNN V2 | 140,375 | 0.000330 | 1.28 | 1.82 | 36.03 |
| LA92 | LSTM V6 | 140,375 | 0.000120 | 0.81 | 1.10 | 13.52 |
| LA92 | LSTM V7 | 140,375 | 0.000090 | 0.71 | 0.95 | 15.57 |
| LA92 | GRU V5 | 140,375 | **0.000075** | **0.65** | **0.87** | **5.26** |

### 结论

1. **US06 的选型取决于误差指标。** RNN V2 用 100 点历史和 353 个可训练参数取得最低 MAE 和 RMSE。LSTM V7 的最大绝对误差最低，比 LSTM V6 低 9.48 个百分点。同为 500 点、4 层、隐藏维度 128 时，GRU V5 的 MAE 略低于 LSTM V7，可训练参数分别为 348,417 和 464,513。
2. **GRU V5 在 LA92 上四项指标均最低。** LSTM V7 的 MAE 也比 V6 低 0.10 个百分点，最大绝对误差则高 2.05 个百分点。
3. **LSTM 的历史窗口试验。** V4 至 V6 同为 2 层、隐藏维度 64，窗口从 150、200 增到 300 点，US06 的全窗口 MAE 依次为 11.66、7.28 和 4.96 个百分点。V7 同时调整了窗口长度、层数和隐藏维度。

此前对齐到第 300 个采样点的 US06 分段分析中，参考 SOC 低于 25% 时，LSTM V6 和 RNN V2 的 MAE 分别为 2.67 和 5.64 个百分点。参考 SOC 达到 25% 及以上时，LSTM V6 平均低估 5.32 个百分点。

US06 参与配置选择，LA92 用于跨工况复评。这些结果对应当前归档权重，各版本的窗口长度、参数量和训练配置存在差异。

## 复现方法

复评使用了 Conda 的 `soc` 环境，版本为 Python 3.9.25、PyTorch 2.1.2、pandas 2.3.3、SciPy 1.13.1 和 tqdm 4.67.3。评估入口默认以 LSTM V7 运行 US06，自动选择可用的 CUDA 或 CPU。在项目根目录运行以下命令，或在 VS Code 中选用 `soc` 解释器后运行评估脚本。 [评估脚本](src/evaluate.py)

```powershell
conda activate soc
python src/evaluate.py
```

切换模型时，在评估入口修改 `model_class`，并在 `checkpoint_paths` 中指定对应权重。评估 LSTM V6 时，将 LSTM 路径改为 `lstm_soc_6.pth`。

```python
model_class = LSTM
checkpoint_paths = {
    GRU: MODELS_DIR / "gru" / "gru_soc_5.pth",
    LSTM: MODELS_DIR / "lstm" / "lstm_soc_7.pth",
    RNN: MODELS_DIR / "RNN" / "rnn_soc_2.pth",
}
```

脚本使用权重中保存的模型参数、标准化统计量和窗口长度。评估 LA92 时，将 `test_path` 中的文件名改为数据表中的 LA92 文件名。

默认运行会统计所选模型的全部有效窗口。复现对齐复评表时，在创建 `test_dataset` 后加入以下代码，使四份权重从第 500 个采样点开始统计。

```python
from torch.utils.data import Subset

offset = 500 - checkpoint["win_size"]
test_dataset = Subset(test_dataset, range(offset, len(test_dataset)))
```

重新训练时，在配置文件中设置窗口长度、隐藏维度和层数，在训练脚本中将 `model_class` 设为 `RNN`、`LSTM` 或 `GRU`。当前训练入口默认使用 CUDA 训练 LSTM，配置为 500 点窗口、4 层、隐藏维度 128、学习率 0.001、批大小 32 和 5 轮。为新实验设置独立的 `save_path` 并确认保存目录存在后，在项目根目录运行 `python src/train.py`。 [配置文件](src/config.py) · [训练脚本](src/train.py)
