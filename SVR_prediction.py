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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import learning_curve


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


if __name__ == '__main__':
    data_path = r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\data_all_21.csv'
    file_path = r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\SVR_prediction_error.csv'
    flow_data = pd.read_csv(data_path, index_col=0)
    flow_data.index = pd.date_range(start='2016-02-01 00:00:00', periods=16704, freq='5min', normalize=True)
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['loop_number',
                      'MAPE_5min', 'MAE_5min',
                      'MAPE_10min', 'MAE_10min',
                      'MAPE_15min', 'MAE_15min',
                      'MAPE_20min', 'MAE_20min']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    for select_loop in range(21):
        result = []
        for pred_interval in range(1, 5):
            observation_lags = 5
            flow_data_real = np.array(flow_data.iloc[:, select_loop]).reshape(-1, 1)
            data_, target_ = data_generate(flow_data_real, time_steps=observation_lags, pred_intervals=pred_interval)
            X_train, X_test, y_train, y_test = train_test_split(data_, target_, test_size=0.2, random_state=33)
            # 分别初始化对特征值和目标值的标准化器
            ss_X = StandardScaler()
            ss_y = StandardScaler()
            # 训练数据都是数值型，所以要标准化处理
            X_train = ss_X.fit_transform(X_train)
            X_test = ss_X.transform(X_test)
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
            svr_rbf = SVR(kernel='rbf',gamma=0.1)
            svr_rbf.fit(X_train, y_train)
            svr_rbf_y_predict = svr_rbf.predict(X_test)

            # 对两种配置下的k近邻回归模型在相同测试集下进行性能评估
            # 预测方式为根据加权距离的KNR
            mape, mae = mean_absolute_percentage_error(ss_y.inverse_transform(y_test.reshape(-1, 1)),
                                                       ss_y.inverse_transform(svr_rbf_y_predict.reshape(-1, 1)))
            result.append(mape)
            result.append(mae)
        with open(file_path, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'loop_number': select_loop + 86,
                             'MAPE_5min': result[0], 'MAE_5min': result[1],
                             'MAPE_10min': result[2], 'MAE_10min': result[3],
                             'MAPE_15min': result[4], 'MAE_15min': result[5],
                             'MAPE_20min': result[6], 'MAE_20min': result[7]})
        print(select_loop)
