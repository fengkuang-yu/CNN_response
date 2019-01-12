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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


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


def data_generate(data, time_steps=1, pred_intervals=1):
    """
    数据处理，将列状的数据拓展开为行形式
    :param data: 输入交通数据
    :param time_steps: 分割时间长度
    :return: 处理过的数据和标签
    :param pred_intervals: 预测时间间隔
    """
    data = np.array(data, dtype=float)
    length = len(data)
    try:
        data[0, 0] = data[0, 0]
    except IndexError:
        print('data_pro module input data must be a 2-dimensional matrix')
    sample_num = length - time_steps - pred_intervals + 1
    temp = np.zeros((sample_num, time_steps))
    for i in range(sample_num):
        temp[i, :] = data[i:i + time_steps, :].flatten()
    return temp, data[-sample_num:]


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
    x_train, x_test, y_train, y_test = data
    x_train = x_train.reshape((-1, param.loop_num * param.time_intervals))
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape((-1, param.loop_num * param.time_intervals))
    y_test = y_test.reshape(-1, 1)
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    # 训练数据都是数值型，所以要标准化处理
    X_train = ss_X.fit_transform(x_train)
    X_test = ss_X.transform(x_test)
    # 目标数据（房价预测值）也是数值型，所以也要标准化处理
    # 说明一下：fit_transform与transform都要求操作2D数据，而此时的y_train与y_test都是1D的，因此需要调用reshape(-1,1)，例如：[1,2,3]变成[[1],[2],[3]]
    y_train = ss_y.fit_transform(y_train).ravel()
    y_test = ss_y.transform(y_test).ravel()

    # 对svr的参数进行交叉验证选取
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    # svr_rbf = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid=param_grid)
    svr_rbf = SVR(kernel='rbf', gamma=0.1)
    svr_rbf.fit(X_train, y_train)
    svr_rbf_y_predict = svr_rbf.predict(X_test)

    # 对两种配置下的k近邻回归模型在相同测试集下进行性能评估
    # 预测方式为根据加权距离的KNR
    mape, mae = mean_absolute_percentage_error(ss_y.inverse_transform(y_test.reshape(-1, 1)),
                                               ss_y.inverse_transform(svr_rbf_y_predict.reshape(-1, 1)))
    return [mape, mae]


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
