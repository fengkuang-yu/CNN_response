# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   seasonal_arima.py
@Time    :   2018/12/2 18:54
@Desc    :
"""

import time
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler


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


def select_model_parameter(data):
    """
    通过该AIC准则对模型的参数进行选择
    :param data: 输入的交通流量数据
    :return: 返回最优的模型参数组合
    """
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    warnings.filterwarnings("ignore")  # 忽略参数配置时的警告信息
    aic = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}288 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue


flow_data = pd.read_csv(r'CNN_traffic_prediction/20181125_seasonal_arima/train_test_data_68.csv',
                        index_col=0)

mod = sm.tsa.statespace.SARIMAX(flow_data,
                                order=(2, 0, 2),
                                seasonal_order=(1, 0, 1, 288),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12))
plt.show()
pred = results.get_prediction(start=pd.to_datetime('2016-03-30 00:00:00'), dynamic=False)
one_step_prediction = pred.predicted_mean
if __name__ == '__main__':
    pass
