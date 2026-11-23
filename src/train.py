import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import RNN
from dataset import *
from config import *
from tqdm import tqdm

data_path = next(DATASETS_DIR.rglob("03-21-17_00.29 25degC_UDDS_Pan18650PF.mat"))
raw_data = data_load(data_path)
features = raw_data[FEATURE_COLS]
# 对原始特征数据进行归一化处理
data = data_standardize(raw_data)
# my_dataset:(target_features, target_soc) tuple类型
my_dataset = MyDataset(data)
print(f"有{len(my_dataset)}个滑动窗口")
# # 第一条my_dataset，即整个元组，再索引元组的第一个位置，即target_features
# print(my_dataset[0][0].shape)
# input_size即数据的特征数
model = RNN(input_size=my_dataset[0][0].shape[-1], hidden_size=16)

def train(model, dataset, lr, epoch_num, batch_size, device='cuda'):
    model.to(device)
    model.train()
    # 定义损失函数和优化器
    # 回归任务一般选择MSE
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num):
        train_loss = 0.0
        dataloader = DataLoader(dataset, batch_size, shuffle=True)
        for x, y in tqdm(dataloader, desc=f'epoch:{epoch + 1:02d}'):
            x, y = x.to(device), y.to(device)
            # 前向传播
            y_pred, _ = model(x)
            # 计算损失
            loss_value = loss(y_pred[:, -1, 0], y[:, -1])
            # 反向传播
            loss_value.backward()
            # 参数优化
            optimizer.step()
            # 梯度清零
            optimizer.zero_grad()

            train_loss += loss_value.item() * x.size(0)
        # 计算平均损失
        this_loss = train_loss / len(dataset)
        print(f"平均损失 = {this_loss:.6f}")

lr = 0.001
epoch_num = 10
batch_size = 32

train(model, my_dataset, lr, epoch_num, batch_size)

save_path = ROOT_DIR / 'rnn_soc.pth'
torch.save(
    {
        'model_state_dict':model.state_dict(),
        'model_args':
        {
            'input_size':model.rnn.input_size,
            'hidden_size':model.rnn.hidden_size,
            'num_layers':model.rnn.num_layers,
        },
        'feature_cols':FEATURE_COLS,
        'feature_mean':features.mean().to_dict(),
        'feature_std':features.std().replace(0, 1).to_dict(),
        'win_size':my_dataset.win_size,
    },
    save_path,
)
print(f"模型已保存到{save_path}")