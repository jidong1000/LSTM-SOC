import torch
from torch import nn

# 回归任务，预测的是一个具体的数值，而不是类别
# 回归任务通常不需要在输出层添加激活函数
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input, hx=None):
        output, _ = self.rnn(input, hx)
        y = self.linear(output)

        return y

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input):
        output, (_, _) = self.lstm(input)
        y = self.linear(output)

        return y

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru, _ = self.gru(x)
        y = self.linear(gru)

        return y