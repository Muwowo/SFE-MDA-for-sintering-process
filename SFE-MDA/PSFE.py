from __future__ import unicode_literals, print_function
import utils
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import pandas as pd
from utils import *
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# 定义网络参数
n_layers = 1
batch_size = 64

# LSTM模型类
class LSTMNetwork(nn.Module):
    def __init__(self, INPUT_SIZE, Hidden_Size, Num_Layer, output_size):
        super(LSTMNetwork, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE, 
            hidden_size=Hidden_Size, 
            num_layers=Num_Layer, 
            batch_first=True, 
            # dropout=dropout_p
        )
        self.HS = Hidden_Size
        self.fc1 = nn.Linear(Hidden_Size, Hidden_Size)
        self.fc2 = nn.Linear(Hidden_Size, output_size)

    def forward(self, x, hidden):
        rnn_out, _ = self.rnn(x, hidden)
        rnn_out = torch.tanh(rnn_out)
        out = self.fc1(rnn_out[:, -1, :])
        out = self.fc2(out)
        return out, rnn_out
    
    def initHidden(self, Batch_Size):
        hidden = torch.zeros(n_layers, Batch_Size, self.HS)
        cell = torch.zeros(n_layers, Batch_Size, self.HS)
        hidden = hidden.to(device)
        cell = cell.to(device)
        return hidden, cell
    
def do_train(UZI):
    input_len = 40
    output_len = 1
    data = pd.read_csv("./DATA/x_temperature_BTP.csv", index_col='time').iloc[:, list(range(0, 18))+[-1]]
    y_max = data['BTP'].max()
    y_min = data['BTP'].min()
    # 划分数据集
    train_num = 8040 - input_len

    # 异常值处理
    for column_name in data.drop(columns=['BTP'], inplace=False, axis=1).columns:
        data = outliners(data, column_name).copy()   

    # 归一化
    for column_name in data.columns:
        data[column_name] = (data[column_name] - data[column_name].min()) / (data[column_name].max() - data[column_name].min())

    data = data.replace(np.nan, 0)
    # 初始化训练参数
    input_size = data.shape[1]
    output_size = output_len
    hidden_size = UZI
    n_layers = 1
    learning_rate = 0.001
    batch_size = 64
    num_epoch = 200

    # 训练集，验证集构造
    data = np.array(data)
    tvdata, tvlabel = time_window_variable(data, input_len, output_len)
    train_x = tvdata[:train_num]
    train_y = tvlabel[:train_num]
    valid_x = tvdata[train_num:]
    valid_y = tvlabel[train_num:]
    print(train_x.shape)
    print(train_y.shape)
    print(train_y)

    # 转格式为tensor+载入数据
    # 训练集
    train_x = torch.from_numpy(train_x).float().to(device)
    train_y = torch.from_numpy(train_y).float().to(device)
    train_dataset = MYDataset(train_x, train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    # 验证集
    valid_x = torch.from_numpy(valid_x).float().to(device)
    valid_y = torch.from_numpy(valid_y).float().to(device)
    valid_dataset = MYDataset(valid_x, valid_y)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

    # 定义网络
    LSTM = LSTMNetwork(input_size, hidden_size, n_layers, output_size)
    optimizer = torch.optim.Adam(LSTM.parameters(), lr=learning_rate, weight_decay=0)
    loss_func = nn.MSELoss()

    # 使用GPU训练
    LSTM = LSTM.to(device)
    loss_func = loss_func.to(device)
    
    # 模型训练
    VALID_LOSS = []
    for epoch in range(num_epoch):
        train_loss = []
        for step, (x, y) in enumerate(train_loader):
            init_hidden = LSTM.initHidden(x.size()[0])
            output, rnn_output = LSTM(x, init_hidden)
            y = y.unsqueeze(1)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.to('cpu')
            train_loss.append(loss.data.numpy())

        # 模型验证
        valid_loss = []
        for step, (x, y) in enumerate(valid_loader):
            init_hidden = LSTM.initHidden(x.size()[0])
            output, rnn_output = LSTM(x, init_hidden)
            y = y.unsqueeze(1)
            loss = loss_func(output, y)
            loss = loss.to('cpu')
            valid_loss.append(loss.data.numpy())
        logger.debug(
            'Epoch:{}, 训练Loss:{:.5f}, 校验Loss:{:.5f}'.format(epoch + 1, np.mean(train_loss), np.mean(valid_loss)))    
        VALID_LOSS.append(np.mean(valid_loss))
    pd.DataFrame(VALID_LOSS).to_csv('./file/log/PSFE_loss/' + str(UZI) + '.csv')

    # 保存模型
    torch.save(LSTM, "./MODEL/Variable1.pt") 
    
    metrics = np.zeros((4, 5))
    # 画图
    LSTM = torch.load("./MODEL/Variable1.pt")
    LSTM = LSTM.to(device)
    LSTM.eval()

    init_hidden = LSTM.initHidden(valid_x.size()[0])
    output, rnn_output = LSTM(valid_x, init_hidden)
    BTP_PREDICT = output[:1000]
    BTP_ACTUAL = valid_y[:1000]
    
    valid_y = valid_y.to('cpu').detach().numpy()
    BTP_ACTUAL = BTP_ACTUAL.to('cpu').detach().numpy()
    BTP_PREDICT = BTP_PREDICT.to('cpu').detach().numpy()
    rnn_output = rnn_output[:, -1, :].to('cpu').detach().numpy()

    # 反归一化
    BTP_ACTUAL = (y_max - y_min) * BTP_ACTUAL + y_min
    BTP_PREDICT = (y_max - y_min) * BTP_PREDICT + y_min
    
    # 查看t+1时刻的BTP指标
    for t in range(5):
        BTP_actual = BTP_ACTUAL[:1000]
        BTP_predict = BTP_PREDICT[:1000]
        R_SCORE = r2_score(BTP_actual, BTP_predict)
        metrics[0, t] = R_SCORE
        HR = hr(BTP_actual, BTP_predict, 0.05)
        metrics[1, t] = HR
        RMSE = rmse(BTP_actual, BTP_predict)
        metrics[2, t] = RMSE
        MAE = mae(BTP_actual, BTP_predict)  
        metrics[3, t] = MAE
        print("t+", t+1, "时刻BTP")
        print('R_SCORE:', R_SCORE)
        print('HR:',HR)
        print('RMSE:',RMSE)
        print('MAE:',MAE) 
        print('--------------------------------------------')

    # 画图
    fig, axs = plt.subplots(5, 1, figsize=(10, 5), dpi=200, sharex=True, sharey=False)
    plt.rcParams['font.sans-serif']=['Times New Roman']
    font = {'family': 'Times New Roman', 'weight': 600, 'style': 'normal', 'size': 15}
    font_1 = {'family': 'Times New Roman', 'weight': 'bold', 'style': 'normal', 'size': 20}
    Text_List = ['t+1', 't+2', 't+3', 't+4', 't+5']
    color_List = ['forestgreen', 'goldenrod', 'navy', 'darkred', 'c']
    for i in range(5):
        x = np.linspace(0, len(BTP_predict), len(BTP_predict))
        BTP_actual = BTP_ACTUAL[:]
        BTP_predict = BTP_PREDICT[:]
        if i == 0:
            axs[i].plot(x, BTP_actual, color='black', linewidth=1, label='actual')
        else:
            axs[i].plot(x, BTP_actual, color='black', linewidth=1)
        axs[i].plot(x, BTP_predict, color=color_List[i], linewidth=1, label=Text_List[i])
        axs[i].set_ylim(50, 70)
        axs[i].set_title(f'R2={r2_score(BTP_actual, BTP_predict):.2f}, RMSE={rmse(BTP_actual, BTP_predict):.3f}', fontsize=8)

    lines = []
    labels = []
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines, labels, loc='upper right', fontsize=9)

    # axs.set_xlabel('Predicted Value', fontdict=font)  # 设置x轴标签
    # axs.set_ylabel('Actual Value', fontdict=font)  # 设置y轴标签

    fig.text(0.45, 0.03, 'time', ha='center', fontdict=font)
    fig.text(0.03, 0.5, 'BTP', va='center', rotation='vertical', fontdict=font)
    fig.text(0.45, 0.95, 'LSTM-TD3', ha='center', fontdict=font_1)
    plt.subplots_adjust(left=0.1, right=0.8, hspace=0.5)
    # plt.show()

    hidden_BTP = pd.DataFrame()
    print(rnn_output.shape)
    for i in range(rnn_output.shape[1]):
        hidden_text = 'hidden' + str(i+1)
        hidden_BTP[hidden_text] = rnn_output[:-4, i]

    tvdata, tvlabel = time_window(data, input_len, 5)
    valid_y = tvlabel[train_num:]
    print(valid_y.shape)

    hidden_BTP['BTP_t+1'] = valid_y[:, 0]
    hidden_BTP['BTP_t+2'] = valid_y[:, 1]
    hidden_BTP['BTP_t+3'] = valid_y[:, 2]
    hidden_BTP['BTP_t+4'] = valid_y[:, 3]
    hidden_BTP['BTP_t+5'] = valid_y[:, 4]

    hidden_BTP.corr().to_csv('./DATA/PSFE_COR/PSFE_cor' + str(UZI) + '.csv')
    
for i in tqdm(range(10)):
    do_train(i+1)