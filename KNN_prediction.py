# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   KNN_prediction.py
@Time    :   2018/11/25 16:06
@Desc    :
"""

import csv
import time
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

np.random.seed(48)


class Parameters:
    """
    网络的配置超参数
    """
    loop_num = 4  # 预测使用的空间节点个数
    time_intervals = 4  # 预测使用的时间滞后个数
    predict_intervals = 0  # 0,1,2,3分别表示5-20分钟预测
    predict_loop = 96  # 96表示159.57号检测线圈
    save_file_path = None
    read_file_path = None


def mean_absolute_percentage_error(pred, real):
    """
    Calculate mean absolute percentage error
    :param pred: prediction value
    :param real: ground truth value
    :return: the MAPE value
    """
    if not len(pred) == len(real):
        raise Exception('Input data dimension in mape calculation module must be consist')
    pred = np.array(pred, dtype=float).flatten()
    real = np.array(real, dtype=float).flatten()
    mape = sum(abs(pred - real) / real) / len(pred)
    mae = sum(abs(pred - real)) / len(pred)
    return mape, mae


def train_test_data(param):
    """
    生成训练和测试数据
    :param param: 数据配置参数
    :return: x_train, x_test, y_train, y_test
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    # 96表示的是159.57号检测线圈的数据
    select_loop = [x for x in range(param.predict_loop - param.loop_num // 2,
                                    param.predict_loop + param.loop_num // 2)]

    def data_pro(data, time_steps=None):
        """
        数据处理，将列状的数据拓展开为行形式

        :param data: 输入交通数据
        :param time_steps: 分割时间长度
        :return: 处理过的数据
        """
        if time_steps is None:
            time_steps = 1
        size = data.shape
        data = np.array(data)
        temp = np.zeros((size[0] - time_steps + 1, size[1] * time_steps))
        for i in range(data.shape[0] - time_steps + 1):
            temp[i, :] = data[i:i + time_steps, :].flatten()
        return temp

    data = pd.read_csv(param.read_file_path)
    label = np.array(data.iloc[param.time_intervals + param.predict_intervals:, param.predict_loop]).reshape(-1, 1)
    data = data.iloc[:, select_loop]
    data = data_pro(data, time_steps=param.time_intervals)
    data = data[: -(1 + param.predict_intervals)]
    return train_test_split(data, label, test_size=0.2, shuffle=True, random_state=42)


def train(data, param):
    """
    对输入神经网络的数据进行训练和评估
    :param data: 输入训练和测试数据
    :param param: 网络配置的参数
    :return: 返回神经网络的测试数据集表现
    """
    x_train, x_test, y_train, y_test = data
    x_train = x_train.reshape((-1, param.loop_num * param.time_intervals))
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape((-1, param.loop_num * param.time_intervals))
    y_test = y_test.reshape(-1, 1)
    # 分别初始化对特征值和目标值的标准化器
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    # 训练数据都是数值型，所以要标准化处理
    X_train = ss_X.fit_transform(x_train)
    X_test = ss_X.transform(x_test)
    # 目标数据（房价预测值）也是数值型，所以也要标准化处理
    # 说明一下：fit_transform与transform都要求操作2D数据，而此时的y_train与y_test都是1D的，因此需要调用reshape(-1,1)，例如：[1,2,3]变成[[1],[2],[3]]
    y_train = ss_y.fit_transform(y_train)
    y_test = ss_y.transform(y_test)

    # 初始化k近邻回归器，并且调整配置，使得预测方式为根据距离加权回归：weights = 'distance'
    dis_knr = KNeighborsRegressor(weights='distance')
    dis_knr.fit(X_train, y_train)
    dis_knr_y_predict = dis_knr.predict(X_test)
    mape, mae = mean_absolute_percentage_error(ss_y.inverse_transform(y_test),
                                               ss_y.inverse_transform(dis_knr_y_predict))

    return [mape, mae]


def STFSA(param, step=4, eps=4):
    """
    对于时空特征进行选择
    :param param: 网络配置参数类
    :param step: 每次增加数据的步长
    :param cur_space: 现在的空间点数目
    :param cur_time: 现在的时间滞后数目
    :param eps: 提前终止的要求
    :return: 输入数据的维度大小和最终误差
    """
    param.loop_num = 4
    param.time_intervals = 4
    data = train_test_data(param)
    opt_error = train(data, param)
    opt_space = 4
    cur_space = 2 * step
    num = 0
    while cur_space - opt_space < eps * step:
        print(num)
        param.loop_num = cur_space
        data = train_test_data(param)
        cur_error = train(data, param)
        if opt_error[0] - cur_error[0] > 0.1:
            opt_space = cur_space
            opt_error = cur_error
        cur_space += step
        num += 1
    # param.time_intervals = opt_time
    param.loop_num = opt_space
    opt_time = 4
    cur_time = 2 * step
    print('tuning up spatial')
    while cur_time - opt_time < eps * step:
        print(num)
        param.time_intervals = cur_time
        data = train_test_data(param)
        cur_error = train(data, param)
        if opt_error[0] - cur_error[0] > 0.1:
            opt_time = cur_time
            opt_error = cur_error
        cur_time += step
        num += 1
    param.time_intervals = opt_time
    return param, opt_error


if __name__ == '__main__':
    params = Parameters()
    params.read_file_path = r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\data_all.csv'
    params.save_file_path = r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\KNN.csv'
    params.time_intervals = 5
    params.loop_num = 4
    with open(params.save_file_path, 'w', newline='') as csvfile:
        fieldnames = ['loop_number',
                      'MAPE_5min', 'MAE_5min',
                      'MAPE_10min', 'MAE_10min',
                      'MAPE_15min', 'MAE_15min',
                      'MAPE_20min', 'MAE_20min',
                      'running_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    for select_loop in range(92, 102):
        params.predict_loop = select_loop
        result = []
        time_start = time.time()
        for pred_interval in range(4):
            params.predict_intervals = pred_interval
            data = train_test_data(params)
            mape, mae = train(data, params)
            result.append(mape)
            result.append(mae)
        time_end = time.time()
        with open(params.save_file_path, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'loop_number': select_loop,
                             'MAPE_5min': result[0], 'MAE_5min': result[1],
                             'MAPE_10min': result[2], 'MAE_10min': result[3],
                             'MAPE_15min': result[4], 'MAE_15min': result[5],
                             'MAPE_20min': result[6], 'MAE_20min': result[7],
                             'running_time': time_end - time_start})
        print(select_loop)
