import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from tqdm import tqdm

# Encoder-Decoder模型类
class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_len):
        super(TimeSeriesAutoencoder, self).__init__()

        # 编码
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.hidden_to_mean = nn.Linear(hidden_dim, latent_dim)
        self.l1 = nn.Linear(latent_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.l2 = nn.Linear(latent_dim, latent_dim)
        self.output_len = output_len
        # 解码
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        hidden1, _ = self.encoder(x)
        hidden1 = hidden1[:, -1, :]
        latent = self.hidden_to_mean(hidden1)
        latent = torch.tanh(latent)
        hidden2 = self.latent_to_hidden(latent)
        hidden2 = hidden2.unsqueeze(1).repeat(1, self.output_len, 1)
        output, _ = self.decoder(hidden2)
        return output, latent
 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def do_train(UZI):
    data = pd.read_csv("./DATA/temperature_BTP.csv", index_col='time')

    # 得到变量边界值，方便反归一化
    data_max = data.max()
    data_min = data.min()

    print(data_max.shape)
    # 归一化
    for column_name in data.columns:
        data[column_name] = (data[column_name] - data[column_name].min()) / (data[column_name].max() - data[column_name].min())

    # 划分数据集
    input_len = 40
    output_len = 5

    train_num = 8040 - input_len  # 为了验证集标签对齐

    data = data.replace(np.nan, 0)
    data = np.array(data)
    tvdata, tvlabel = time_window_temp(data, input_len, output_len)
    _, tvBTP = time_window(data, input_len, output_len)
    train_x = tvdata[:train_num]
    train_y = tvlabel[:train_num]
    valid_x = tvdata[train_num:]
    valid_y = tvlabel[train_num:]
    train_BTP = tvBTP[:train_num]
    valid_BTP = tvBTP[train_num:]

    input_dim = train_x.shape[2]
    hidden_dim = 64
    latent_dim = UZI
    num_epoch = 200
    batch_size = 32
    learning_rate = 0.001

    # 测试集
    train_x = torch.from_numpy(train_x).float().to(device)
    train_y = torch.from_numpy(train_y).float().to(device)
    train_dataset = MYDataset(train_x, train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    # 验证集
    valid_x = torch.from_numpy(valid_x).float().to(device)
    valid_y = torch.from_numpy(valid_y).float().to(device)
    valid_dataset = MYDataset(valid_x, valid_y)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

    model = TimeSeriesAutoencoder(input_dim, hidden_dim, latent_dim, output_len)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    loss_funcion2 = nn.MSELoss().to(device)

    # 训练
    VALID_LOSS = []
    for epoch in range(num_epoch):
        train_loss = []
        for step, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            output, latent = model(x)
            loss = loss_funcion2(output, y)
            loss.backward()
            optimizer.step()
            loss = loss.to('cpu')
            train_loss.append(loss.data.numpy())
        
        # 模型验证
        valid_loss = []
        for step, (x, y) in enumerate(valid_loader):
            output, latent = model(x)
            loss = loss_funcion2(output, y)
            loss = loss.to('cpu')
            valid_loss.append(loss.data.numpy())
        logger.debug(
            'Epoch:{}, 训练Loss:{:.5f}, 校验Loss:{:.5f}'.format(epoch + 1, np.mean(train_loss), np.mean(valid_loss)))
        VALID_LOSS.append(np.mean(valid_loss))
    pd.DataFrame(VALID_LOSS).to_csv('./file/log/TSFE_loss/' + str(UZI) + '.csv')

    torch.save(model, "./MODEL/TempFeature1.pt") 

    # 验证集验证
    model = torch.load("./MODEL/TempFeature1.pt")
    model = model.to(device)
    model.eval()

    output, latent = model(valid_x)
    print(latent.size())
    latent = latent.to('cpu').detach().numpy()

    t = 0  # t=0,1,2,3,4
    num = 7  # 风箱编号7-13
    actual = valid_y[:, t, num - 7].to('cpu').detach().numpy()
    predict = output[:, t, num - 7].to('cpu').detach().numpy()

    max = data_max[num - 7]
    min = data_min[num - 7]

    # 反归一化
    actual = (max - min) * actual + min
    predict = (max - min) * predict + min  

    R_SCORE = r2_score(actual, predict)
    HR = hr(actual, predict, 0.03)
    RMSE = rmse(actual, predict)
    MAE = mae(actual, predict)  
    print('R_SCORE:', R_SCORE)
    print('HR:',HR)
    print('RMSE:',RMSE)
    print('MAE:',MAE) 

    fig, axs = plt.subplots(1, 1, figsize=(10, 5), dpi=200, sharex=True, sharey=False)

    x = np.linspace(0, len(actual), len(predict))
    plt.rcParams['font.sans-serif']=['Times New Roman']
    font = {'family': 'Times New Roman', 'weight': 600, 'style': 'normal', 'size': 15}
    font_1 = {'family': 'Times New Roman', 'weight': 'bold', 'style': 'normal', 'size': 20}
    axs.plot(x, actual, color='black', linewidth=1, label='actual')
    axs.plot(x, predict, color='red', linewidth=1, label='predict')
    axs.tick_params(labelsize=7)

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
    fig.text(0.03, 0.5, '', va='center', rotation='vertical', fontdict=font)
    fig.text(0.45, 0.95, 'Temperature Feature Extractor', ha='center', fontdict=font_1)
    plt.subplots_adjust(left=0.1, right=0.8, hspace=0.5)
    # plt.show()

    # 计算验证集潜变量与BTP相关性
    latent_BTP = pd.DataFrame()
    for i in range(latent.shape[1]):
        latent_text = 'latent' + str(i+1)
        latent_BTP[latent_text] = latent[:, i]

    latent_BTP['BTP_t+1'] = valid_BTP[:, 0]
    latent_BTP['BTP_t+2'] = valid_BTP[:, 1]
    latent_BTP['BTP_t+3'] = valid_BTP[:, 2]
    latent_BTP['BTP_t+4'] = valid_BTP[:, 3]
    latent_BTP['BTP_t+5'] = valid_BTP[:, 4]

    latent_BTP.corr().to_csv('./DATA/TSFE_COR/TSFE_cor' + str(UZI) + '.csv')
    
for i in tqdm(range(10)):
    do_train(i+1)