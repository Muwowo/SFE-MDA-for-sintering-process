import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from utils import *
import torch
import MDA_Agent
import torch.nn as nn
from math import sqrt, exp
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.ticker as ticker

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# Define network parameters
hidden_size = 5
n_layers = 1
batch_size = 32

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
        self.fc1 = nn.Linear(Hidden_Size, Hidden_Size)
        self.fc2 = nn.Linear(Hidden_Size, output_size)

    def forward(self, x, hidden):
        rnn_out, _ = self.rnn(x, hidden)
        rnn_out = torch.tanh(rnn_out)
        out = self.fc1(rnn_out[:, -1, :])
        out = self.fc2(out)
        return out, rnn_out
    
    def initHidden(self, Batch_Size):
        hidden = torch.zeros(n_layers, Batch_Size, hidden_size)
        cell = torch.zeros(n_layers, Batch_Size, hidden_size)
        hidden = hidden.to(device)
        cell = cell.to(device)
        return hidden, cell
    

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
        hidden2 = self.latent_to_hidden(latent)
        hidden2 = hidden2.unsqueeze(1).repeat(1, self.output_len, 1)
        output, _ = self.decoder(hidden2)
        return output, latent
    

def train(a1, a2, a3, a4, a5, k1, k2, k3, k4, k5):

    data = pd.read_csv("./DATA/x_temperature_BTP.csv", index_col='time').iloc[:, list(range(0, 18))+[-1]]

    # 划分数据集
    input_len = 40
    output_len = 5
    train_num = 8040 - input_len  # 为了和LSTM算法验证集标签对齐
    y_max = data['BTP'].max()
    y_min = data['BTP'].min()

    for column_name in data.columns:
        data[column_name] = (data[column_name] - data[column_name].min()) / (data[column_name].max() - data[column_name].min())

    data = data.replace(np.nan, 0)
    data = np.array(data)

    # 加载LSTM，得到多维度1时间步长的隐变量，作为状态空间,满足markov条件
    LSTM = torch.load("./MODEL/Variable.pt")
    LSTM = LSTM.to(device)
    LSTM.eval()

    # 得到潜变量1
    tvdata_LSTM, tvlabel_LSTM = time_window(data, input_len, output_len)
    train_x_LSTM = tvdata_LSTM[:train_num]
    train_y = tvlabel_LSTM[:train_num]
    valid_x_LSTM = tvdata_LSTM[train_num:]
    valid_y = tvlabel_LSTM[train_num:]

    train_x_LSTM = torch.from_numpy(train_x_LSTM).float().to(device)
    valid_x_LSTM = torch.from_numpy(valid_x_LSTM).float().to(device)

    init_hidden = LSTM.initHidden(train_x_LSTM.size()[0])
    _, train_x = LSTM(train_x_LSTM, init_hidden)
    train_x = train_x[:, -1, :]

    init_hidden = LSTM.initHidden(valid_x_LSTM.size()[0])
    _, valid_x = LSTM(valid_x_LSTM, init_hidden)
    valid_x = valid_x[:, -1, :]

    # 转为numpy
    train_x = train_x.to('cpu').detach().numpy()
    valid_x = valid_x.to('cpu').detach().numpy()

    # print(train_x.shape)
    # print(valid_x.shape)

    # 加载Temp_Feature_Extractor
    model = torch.load("./MODEL/TempFeature.pt")
    model = model.to(device)
    model.eval()

    temp = pd.read_csv("./DATA/temperature_BTP.csv", index_col='time')

    # 归一化
    for column_name in temp.columns:
        temp[column_name] = (temp[column_name] - temp[column_name].min()) / (temp[column_name].max() - temp[column_name].min())
    temp = np.array(temp)
    tvtemp, _ = time_window_temp(temp, input_len, output_len)
    train_temp = tvtemp[:train_num]
    valid_temp = tvtemp[train_num:]
    # print(train_temp.shape)

    train_temp = torch.from_numpy(train_temp).float().to(device)
    valid_temp = torch.from_numpy(valid_temp).float().to(device)
    _, train_latent = model(train_temp)
    _, valid_latent = model(valid_temp)
    # print(train_latent.size())

    train_latent = train_latent.to('cpu').detach().numpy()
    valid_latent = valid_latent.to('cpu').detach().numpy()

    # 合并自变量
    train_x = np.concatenate((train_x, train_latent), axis=1)
    valid_x = np.concatenate((valid_x, valid_latent), axis=1)

    RMSE_mean_list = []
    RMSE5_list = []

    # 环境建模
    class Env:
        def __init__(self, x, y, train=False, minibatch_size=2000):
            self.memory = x
            self.labels = y
            
            self.observation_space = np.array([[]], dtype=float)
            self.check = np.array([[]], dtype=float)
            
            self.observation_dim = self.memory.shape[1]
            self.action_dim = self.labels.shape[1]
            
            self.i = 0
            self.e = 0.01
            self.train = train
            
            if train:
                self._max_episode_steps = minibatch_size
            else:
                self._max_episode_steps = self.memory.shape[0]
            
            self.minibatch_size = minibatch_size
            
            self.MSE = 0
            self.HR = 0
            
        def reset(self):
            self.i = 0
            self.MSE = 0
            
            if not self.train:
                self.observation_space = self.memory.copy()
                self.check = self.labels.copy()
            else:
                indices = np.random.choice(np.arange(self.memory.shape[0]), size=self.minibatch_size, replace=False)
                self.observation_space = self.memory[indices]
                self.check = self.labels[indices]
                
            return self.observation_space[self.i].tolist()
        
        def step(self, action):
            reward = 0
            # 5步BTP分别的预测MSE
            MSE1 = (action[0] - self.check[self.i][0])**2
            MSE2 = (action[1] - self.check[self.i][1])**2
            MSE3 = (action[2] - self.check[self.i][2])**2
            MSE4 = (action[3] - self.check[self.i][3])**2
            MSE5 = (action[4] - self.check[self.i][4])**2
            MSE_mean = np.mean((action - self.check[self.i])**2)
            HR1 = 1 if abs(action[0] - self.check[self.i][0]) <= a1*self.e else 0
            HR2 = 1 if abs(action[1] - self.check[self.i][1]) <= a2*self.e else 0
            HR3 = 1 if abs(action[2] - self.check[self.i][2]) <= a3*self.e else 0
            HR4 = 1 if abs(action[3] - self.check[self.i][3]) <= a4*self.e else 0
            HR5 = 1 if abs(action[4] - self.check[self.i][4]) <= a5*self.e else 0

            RMSE_mean_list.append(sqrt(MSE_mean))
            RMSE5_list.append(MSE5)
            # 每步的奖励划分

            reward = -0.1 * ((k1-HR1)*MSE1 + (k2-HR2)*MSE2 + (k3-HR3)*MSE3 + (k4-HR4)*MSE4 + (0.5)*MSE5)
            # reward = -MSE_mean
            self.MSE += MSE_mean
            self.HR += 1/5 * (HR1 + HR2 + HR3 + HR4 + HR5)

            self.i += 1
            if self.i == self.observation_space.shape[0]:
                done = True
                observation = self.observation_space[self.i-1].tolist()
                self.MSE /= self.observation_space.shape[0]
                self.HR /= self.observation_space.shape[0]
            else:
                done = False
                observation = self.observation_space[self.i].tolist()
        
            return observation, reward, done
        
    # 参数设计
    start_timesteps = 2000  # 起始随机行动的步数
    eval_freq = 5000        # 每2000步进行一次评估
    max_timesteps = 20000   # 最大步数
    expl_noise = 0.03    # 噪音
    batch_size = 64
    discount = 0.995       # 折扣因子
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.2
    policy_freq = 5
    target_freq = 20

    env = Env(train_x, train_y)
    env.train = True
    state_dim = env.observation_dim
    action_dim = env.action_dim
    max_action = 1
    min_action = 0

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "min_action": min_action,
        "batch_size": batch_size,
        "discount": discount,
        "tau": tau,
        
        "policy_freq": policy_freq,
        "target_freq": target_freq
    }
    kwargs["policy_noise"] = policy_noise
    kwargs["noise_clip"] = noise_clip

    policy = MDA_Agent.MDA(**kwargs)

    # 模型训练
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    Q1 = []
    Q2 = []
    Q3 = []
    TQMAX = []
    TQMID = []
    TQMIN = []
    for t in range(int(max_timesteps)):
        episode_timesteps += 1
        
        # Select action randomly or according to policy
        if t < start_timesteps:
            action = np.random.uniform(min_action, max_action, size=action_dim)
        else:
            action = policy.select_action(np.array(state)) + np.random.normal(0, expl_noise, size=action_dim)
            # action = policy.select_action(np.array(state))
            
        # Perform action
        next_state, reward, done = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        
        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            current_Q1, current_Q2, current_Q3, tqmax, tqmid, tqmin = policy.train(replay_buffer)
            Q1.append(current_Q1)
            Q2.append(current_Q2)
            Q3.append(current_Q3)
            TQMAX.append(tqmax)
            TQMID.append(tqmid)
            TQMIN.append(tqmin)
        if done: 

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

    # 保存模型
    policy.save('./MODEL/UZI/policy')

    # 创建指标记录数组，方便实验
    metrics = np.zeros((4, 5))

    # 加载模型并验证
    policy.load('./MODEL/UZI/policy')
    env_test = Env(valid_x, valid_y)
    state, done = env_test.reset(), False
    BTP_ACTUAL = env_test.check
    BTP_PREDICT = []
    avg_reward = 0
    while not done:
        action = policy.select_action(np.array(state))
        BTP_PREDICT.append(action)
        state, reward, done = env_test.step(action)
        avg_reward += reward

    BTP_PREDICT = np.array(BTP_PREDICT)

    # 反归一化
    BTP_ACTUAL = (y_max - y_min) * BTP_ACTUAL + y_min
    BTP_PREDICT = (y_max - y_min) * BTP_PREDICT + y_min

    # 查看t+1时刻的BTP指标
    for t in range(5):
        BTP_actual = BTP_ACTUAL[:1000, t]
        BTP_predict = BTP_PREDICT[:1000, t]
        R_SCORE = r2_score(BTP_actual, BTP_predict)
        metrics[0, t] = R_SCORE
        HR = hr(BTP_actual, BTP_predict, 0.03)
        metrics[1, t] = HR
        RMSE = rmse(BTP_actual, BTP_predict)
        metrics[2, t] = RMSE
        MAE = mae(BTP_actual, BTP_predict)
        metrics[3, t] = MAE

    # 画图
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), dpi=200, sharex=False, sharey=False)
    pos = [axs[0], axs[1], axs[2], axs[3], axs[4]]

    plt.rcParams['font.sans-serif']=['Times New Roman']
    font = {'family': 'Times New Roman', 'weight': 600, 'style': 'normal', 'size': 20}
    font_1 = {'family': 'Times New Roman', 'weight': 'bold', 'style': 'normal', 'size': 20}
    font_2 = {'family': 'Times New Roman', 'weight': 400, 'style': 'normal', 'size': 10}
    Text_List = ['t+1', 't+2', 't+3', 't+4', 't+5']
    color_List = ['magenta', 'orange', 'limegreen', 'cyan', 'red']
    for i in range(5):
        x = np.linspace(0, len(BTP_predict), len(BTP_predict))
        BTP_actual = BTP_ACTUAL[:1000, i]
        BTP_predict = BTP_PREDICT[:1000, i]
        if i == 0:
            pos[i].plot(x, BTP_actual, color='black', linewidth=2, label='Actual')
        else:
            pos[i].plot(x, BTP_actual, color='black', linewidth=2)
        pos[i].plot(x, BTP_predict, color=color_List[i], linewidth=1, label=Text_List[i])
        pos[i].set_xlim(0, 1000)
        pos[i].set_ylim(50, 70)
        pos[i].xaxis.set_major_locator(ticker.MultipleLocator(200))
        pos[i].yaxis.set_major_locator(ticker.MultipleLocator(5))
        pos[i].grid(True)

    lines = []
    labels = []
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines, labels, loc='upper right', fontsize=11)

    fig.text(0.5, 0.03, 'Samples', ha='center', fontdict=font)
    fig.text(0.03, 0.5, 'BTP Values', va='center', rotation='vertical', fontdict=font)
    fig.text(0.5, 0.95, 'SFE-MDA', ha='center', fontdict=font_1)
    plt.subplots_adjust(left=0.1, right=0.9, hspace=0.5)
    plt.show()

    # Q值变化趋势
    plt.figure(figsize=(10, 5), dpi=200)
    x = np.linspace(1, len(Q1), len(Q1))
    y1 = np.array(Q1)
    y2 = np.array(Q2)
    y3 = np.array(Q3)
    y = (y1 + y2 + y3) / 3
    plt.scatter(x, y, s=1, marker='o', c='darkorchid')
    plt.title("Q trend")
    # plt.show()
    pd.DataFrame(y).to_csv('MDA_Q.csv')

    # 3个Target Q值变化趋势
    plt.figure(figsize=(8, 3.5), dpi=200)
    x = np.linspace(1, len(TQMAX), len(TQMAX))
    y1 = np.array(TQMAX)
    y2 = np.array(TQMID)
    y3 = np.array(TQMIN)
    plt.scatter(x, y1, s=1, marker='v', c='peru', label='max')
    plt.scatter(x, y2, s=1, marker='v', c='steelblue', label='mid')
    plt.scatter(x, y3, s=1, marker='v', c='darkorchid', label='min')
    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("target q value")
    # plt.show()

    # RMSE变化
    plt.figure(figsize=(10, 5), dpi=200)
    x = np.linspace(1, len(RMSE_mean_list), len(RMSE_mean_list))
    plt.scatter(x, RMSE5_list, c='r', marker='*', s=0.2)
    plt.ylim(0, 1)
    plt.xlabel("Number of iterations", fontdict=font_1)
    plt.ylabel("RMSE of each batch", fontdict=font_1)
    plt.show()
    pd.DataFrame(RMSE5_list).to_csv('./file/log/RMSE/2.csv')

    return metrics

k1=(0.922+0.857+0.785+0.719+0.66)/0.922
k2=(0.922+0.857+0.785+0.719+0.66)/0.857
k3=(0.922+0.857+0.785+0.719+0.66)/0.785
k4=(0.922+0.857+0.785+0.719+0.66)/0.719
k5=(0.922+0.857+0.785+0.719+0.66)/0.66
print(k1, k2, k3, k4, k5)
start_time = time.time()

metrics = train(0.2, 0.2, 0.2, 0.2, 0.2, k1, k2, k3, k4, k5)
print(metrics)

end_time = time.time()
print("执行时间:", end_time-start_time)
