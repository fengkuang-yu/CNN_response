# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   KNN_prediction.py
@Time    :   2018/11/25 16:06
@Desc    :
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


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
    return mape


flow_data = pd.read_csv(
    r'D:\Users\yyh\TensorFlow-workspace\demo1\CNN_traffic_prediction\20181125_KNN\train_test_data_68.csv', index_col=0)
data_, target_ = data_generate(flow_data, time_steps=20)

X_train, X_test, y_train, y_test = train_test_split(data_, target_, test_size=0.25, random_state=33)
# 分析回归目标值的差异
print('The max target value is ', np.max(target_))
print('The min target value is ', np.min(target_))
print('The average target value is ', np.mean(target_), '\n\n\n')

# 第三步：训练数据和测试数据标准化处理

# 分别初始化对特征值和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()
# 训练数据都是数值型，所以要标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
# 目标数据（房价预测值）也是数值型，所以也要标准化处理
# 说明一下：fit_transform与transform都要求操作2D数据，而此时的y_train与y_test都是1D的，因此需要调用reshape(-1,1)，例如：[1,2,3]变成[[1],[2],[3]]
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

# 第四步：使用两种不同配置的K近邻回归模型进行训练，并且分别对测试数据进行预测

# 1.初始化k近邻回归器，并且调整配置，使得预测方式为平均回归：weights = 'uniform'
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)
# 2.初始化k近邻回归器，并且调整配置，使得预测方式为根据距离加权回归：weights = 'distance'
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

# 第五步：对两种配置下的k近邻回归模型在相同测试集下进行性能评估
# 使用R-squared、MSE、MAE指标评估

# 1.预测方式为平均回归的KNR
print('R-squared value of uniform-weighted KNR is',
      uni_knr.score(X_test, y_test))
print('the MSE of uniform-weighted KNR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print('the MAE of uniform-weighted KNR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print('the MAPE of uniform-weighted KNR is',
      mean_absolute_percentage_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))


# 2.预测方式为根据加权距离的KNR
print('R-squared value of distance-weighted KNR is',
      dis_knr.score(X_test, y_test))
print('the MSE of distance-weighted KNR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
print('the MAE of distance-weighted KNR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
print('the MAPE of distance-weighted KNR is',
      mean_absolute_percentage_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))


if __name__ == '__main__':
    from sklearn.preprocessing import *
    import numpy as np

    a = np.array([x for x in range(12)]).reshape(3, 4)

