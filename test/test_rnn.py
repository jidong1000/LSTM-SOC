import torch
from torch import nn

# 创建RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        # RNN层
        self.RNN = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # 全连接层
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input, hx=None):
        output, hn = self.RNN(input, hx)
        output = self.linear(output)

        return output, hn
    
# 8个滑动窗口 每个滑动窗口内30个采样点 每个采样点的特征维度是3
x = torch.randn(8, 30, 3)

model = RNN(input_size=3, hidden_size=16)

# 前向传播
output, _ = model(x)
print(f"output.shape:{output.shape}")
y_pred = output[:, -1, :].squeeze(-1)
print(f"y_pred.shape:{y_pred.shape}")
