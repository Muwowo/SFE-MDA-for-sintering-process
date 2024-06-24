from __future__ import unicode_literals, print_function
import pandas as pd
import pymysql
import numpy as np
import torch
from sklearn.metrics import r2_score
from math import sqrt
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from scipy.optimize import curve_fit
import copy

# 数据库配置
class sqlConfig:
    db_name = ""  # 数据库名
    db_user = ""  # 用户名
    db_host = ""  # IP地址
    db_port = ''  # 端口号
    db_passwd = ""  # 密码


def sql_con():
    # 打开数据库连接，此方法指定数据库
    conn = pymysql.connect(
        port=sqlConfig.db_port,
        host=sqlConfig.db_host,
        user=sqlConfig.db_user,
        passwd=sqlConfig.db_passwd,
        db=sqlConfig.db_name,
        charset="utf8mb4"
    )
    return conn


def read_para(table_name, seq_len, columns, where_str):
    print('正在连接数据库,待读取' + table_name)
    conn = sql_con()
    cursor = conn.cursor()
    sql = f"SELECT * from {table_name} where 0 = 0 {where_str}"
    cursor.execute(sql)
    result = cursor.fetchall()
    df = pd.DataFrame(data=list(result), columns=columns).iloc[-seq_len:, :]
    df.set_index('time', inplace=True)
    cursor.close()
    conn.close()
    print('已成功读取' + table_name)
    return df

def read_time(table_name, seq_len, columns, where_str):
    print('正在连接数据库，待读取时间编号...')
    conn = sql_con()
    cursor = conn.cursor()
    sql = f"SELECT * from {table_name} where 0 = 0 {where_str}"
    cursor.execute(sql)
    result = cursor.fetchall()
    df = pd.DataFrame(data=list(result), columns=columns).iloc[-seq_len:, :]
    cursor.close()
    conn.close()
    print('已成功读取时间编号')
    return df['time']

def time_window(dataset, input_len, output_len):
    '''
    构建时间序列数据集的函数，输出的数据集格式为(batch, seq_len, input_size)
    dataset:原始数据集
    input_len:序列长度，即时间窗的长度
    data:形状(batch, input_len, input_size)
    labels:形状(batch, output_len, output_size)
    '''
    data = []
    labels = []
    # index从0开始
    start_index = input_len - 1
    end_index = len(dataset) - 1
    for i in range(start_index, end_index - output_len + 1):
        data_1 = []
        data_2 = []
        for j in range(i-start_index, i+1):
            data_1.append(dataset[j, :])
        for j in range(i+1, i+1+output_len):
            data_2.append(dataset[j, -1])
        data.append(data_1)
        labels.append(data_2)

    return np.array(data), np.array(labels)

def time_window_variable(dataset, input_len, output_len):
    '''
    构建时间序列数据集的函数，输出的数据集格式为(batch, seq_len, input_size)
    dataset:原始数据集
    input_len:序列长度，即时间窗的长度
    data:形状(batch, input_len, input_size)
    labels:形状(batch, output_len, output_size)
    '''
    data = []
    labels = []
    # index从0开始
    start_index = input_len - 1
    end_index = len(dataset) - 1
    for i in range(start_index, end_index - output_len + 1):
        data_1 = []
        data_2 = []
        for j in range(i-start_index, i+1):
            data_1.append(dataset[j, :])
        data_3 = copy.deepcopy(data_1)
        data_3[start_index][-1] = data_3[start_index-1][-1]
        data_2.append(dataset[i, -1])
        data.append(data_3)
        labels.append(dataset[i, -1])

    return np.array(data), np.array(labels)

def time_window_temp(dataset, input_len, output_len):
    '''
    构建时间序列数据集的函数，输出的数据集格式为(batch, seq_len, input_size)
    dataset:原始数据集
    seq_len:序列长度，即时间窗的长度
    data:形状(batch, input_len, input_size)
    labels:形状(batch, output_len, output_size)
    '''
    data = []
    labels = []
    # index从0开始
    start_index = input_len - 1
    end_index = len(dataset) - 1
    for i in range(start_index, end_index - output_len + 1):
        data_1 = []
        data_2 = []
        for j in range(i-start_index, i+1):
            data_1.append(dataset[j, :-1])
        for j in range(i+1, i+1+output_len):
            data_2.append(dataset[j, :-1])
        data.append(data_1)
        labels.append(data_2)

    return np.array(data), np.array(labels)    
def outliners(data, col):
    def box_plot_outliners(data_ser):
        IQR = data_ser.quantile(0.75)-data_ser.quantile(0.25)
        val_low = data_ser.quantile(0.25) - IQR * 1.5
        val_up = data_ser.quantile(0.75) + IQR * 1.5
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        # print(rule_low)
        return rule_low, rule_up, val_low, val_up
    data_n = data.copy()
    data_series = data_n[col].copy()
    rule_low, rule_up, val_low, val_up = box_plot_outliners(data_series)
    data_n.loc[rule_up, col] = val_up
    data_n.loc[rule_low, col] = val_low
    return data_n

# 定义评价指标
def r2(actual, predict):
    a = 0
    b = 0
    for i in range(len(actual)):
        a += pow((predict[i] - np.mean(actual)), 2)
        b += pow((actual[i] - np.mean(actual)), 2)
    r_2 = a / b
    return r_2


def rmse(actual, predict):
    sum = 0
    for i in range(len(actual)):
        sum += pow((predict[i] - actual[i]), 2)
    return sqrt(sum / len(actual))


def mae(actual, predict):
    sum = 0
    for i in range(len(actual)):
        sum += abs((predict[i] - actual[i]))
    return sum / len(actual)

def hr(actual, predict, ratio):
    sum = 0
    for i in range(len(actual)):
        if abs((actual[i] - predict[i])/(actual[i])) <= ratio:
            sum += 1
    return sum / len(actual)

def mape(actual, predict):
    sum = 0
    for i in range(len(actual)):
        sum += abs((actual[i] - predict[i])/(actual[i]))
    return sum / len(actual) * 100

def error(actual, predict):
    return actual - predict

    
# 构建数据集
class MYDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.feature = x_tensor
        self.label = y_tensor

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return self.feature.size(0)
    

# 拟合BTP四次曲线
def cal_BTP_4(x, y):
    def fourth_order_poly(x, a, b, c, d, e):
        return a * x**4 + b * x**3 + c * x**2 + d * x + e
    PARAMS = []
    for i in range(len(y)):
        params, cov = curve_fit(fourth_order_poly, x, y[i])
        PARAMS.append(params)
    PARAMS = np.array(PARAMS)
    BTP = np.zeros(len(y))
    A = PARAMS[:, 0]
    B = PARAMS[:, 1]
    C = PARAMS[:, 2]
    D = PARAMS[:, 3]
    E = PARAMS[:, 4]

    def solve_cubic(a, b, c, d):
        roots = np.roots([4*a, 3*b, 2*c, d])
        roots.sort()
        return roots

    BTP = np.zeros(len(y))
    for i in range(len(y)):
        roots = solve_cubic(A[i], B[i], C[i], D[i])
        for j in range(len(roots)):
            if 12*A[i]*roots[j]**2 + 6*B[i]*roots[j] + 2*C[i] < 0:
                BTP[i] = roots[j]
    return BTP


# 强化学习经验池
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(5000)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)   
        )