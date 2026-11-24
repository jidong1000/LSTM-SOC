from math import sqrt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from config import DATASETS_DIR, ROOT_DIR
from dataset import MyDataset, data_load
from model import RNN

def evaluate(model, test_dataset, batch_size=32, device="cpu"):
    num_windows = len(test_dataset)
    if num_windows == 0:
        raise ValueError("测试数据不足一个滑动窗口")

    model.to(device)
    model.eval()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    max_absolute_error = 0.0

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred, _ = model(x)

            # 与训练一致，只比较每个窗口最后一个时间点的 SOC。
            error = y_pred[:, -1, 0] - y[:, -1]
            absolute_error = error.abs()

            squared_error_sum += error.square().sum().item()
            absolute_error_sum += absolute_error.sum().item()
            max_absolute_error = max(
                max_absolute_error, absolute_error.max().item()
            )

    mse = squared_error_sum / num_windows
    return {
        "mse": mse,
        "mae": absolute_error_sum / num_windows,
        "rmse": sqrt(mse),
        "max_error": max_absolute_error,
    }

if __name__ == "__main__":
    checkpoint = torch.load(
        ROOT_DIR / "rnn_soc_2.pth", map_location="cpu", weights_only=True
    )
    model = RNN(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state_dict"])

    test_path = next(DATASETS_DIR.rglob("03-20-17_01.43 25degC_US06_Pan18650PF.mat"))
    test_data = data_load(test_path)

    # 用 UDDS 训练时保存的统计量标准化 US06 的输入特征。
    feature_cols = checkpoint["feature_cols"]
    feature_mean = pd.Series(checkpoint["feature_mean"])
    feature_std = pd.Series(checkpoint["feature_std"])
    test_data[feature_cols] = (test_data[feature_cols] - feature_mean) / feature_std
    test_dataset = MyDataset(test_data, win_size=checkpoint["win_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = evaluate(model, test_dataset, device=device)
    print(f"测试窗口数：{len(test_dataset)}")
    print(f"测试 MSE：{metrics['mse']:.6f}")
    print(f"测试 MAE：{metrics['mae'] * 100:.2f}%")
    print(f"测试 RMSE：{metrics['rmse'] * 100:.2f}%")
    print(f"最大绝对误差：{metrics['max_error'] * 100:.2f}%")
