# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   seasonal_arima.py
@Time    :   2018/12/2 18:54
@Desc    :
"""

import csv
import itertools
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller  # Dickey-Fuller test


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


def select_model_parameter(data):
    """
    通过该AIC准则对模型的参数进行选择
    :param data: 输入的交通流量数据
    :return: 返回最优的模型参数组合
    """
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 288) for x in list(itertools.product(p, d, q))]
    warnings.filterwarnings("ignore")  # 忽略参数配置时的警告信息
    aic_min = 0
    optimal_param = None
    optimal_seasonal_param = None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                if results.aic < aic_min:
                    optimal_param = param
                    optimal_seasonal_param = param_seasonal
            except:
                continue
    print('ARIMA{}x{}288 - AIC:{}'.format(optimal_param, optimal_seasonal_param, aic_min))


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')  # autolag : {‘AIC’, ‘BIC’, ‘t-stat’, None}
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


if __name__ == '__main__':
    # 数据生成
    data_path = r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\data_all_21.csv'
    file_path = r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\prediction_error.csv'
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
        flow_data_real = flow_data.iloc[:, select_loop]
        flow_shift = flow_data_real.shift(288)
        flow_shift = flow_shift.dropna()
        flow_data_train = flow_data_real.diff(288)
        flow_data_train = flow_data_train.dropna()

        mod = sm.tsa.statespace.SARIMAX(flow_data_train,
                                        order=(1, 0, 1),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

        results = mod.fit()
        # residuals = pd.DataFrame(results.resid)  # 对于训练数据的拟合的残差值
        # residuals = residuals.rename(columns={0: 'Residuals'})  # 改变列的名字
        # smoothed_deterministic = results.fittedvalues  # deterministic中取出residuals的剩余值
        # y == smoothed_deterministic + residuals + history_average

        # 利用建立好的模型进行预测
        arima_pre = pd.Series()
        arima_pre2 = pd.Series()
        arima_pre3 = pd.Series()
        arima_pre4 = pd.Series()
        start_time = time.time()
        for date in pd.date_range(start='2016-03-18', periods=3340, freq='5min', normalize=True):
            pred = results.get_prediction(start=date, dynamic=True, full_results=True)
            arima_pre = arima_pre.append(pred.predicted_mean[0:1])
            arima_pre2 = arima_pre2.append(pred.predicted_mean[1:2])
            arima_pre3 = arima_pre3.append(pred.predicted_mean[2:3])
            arima_pre4 = arima_pre4.append(pred.predicted_mean[3:4])
        end_time = time.time()
        time_lag = end_time - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_lag // 60, time_lag % 60))  # 打印出来时间

        # 返还真实值
        arima_pre_real = (arima_pre + flow_shift).dropna()
        arima_pre2_real = (arima_pre2 + flow_shift).dropna()
        arima_pre3_real = (arima_pre3 + flow_shift).dropna()
        arima_pre4_real = (arima_pre4 + flow_shift).dropna()

        # 返回预测和真实值的误差
        mape_5, mae_5 = mean_absolute_percentage_error(arima_pre_real, flow_data_real[arima_pre_real.index])
        mape_10, mae_10 = mean_absolute_percentage_error(arima_pre2_real, flow_data_real[arima_pre2_real.index])
        mape_15, mae_15 = mean_absolute_percentage_error(arima_pre3_real, flow_data_real[arima_pre3_real.index])
        mape_20, mae_20 = mean_absolute_percentage_error(arima_pre4_real, flow_data_real[arima_pre4_real.index])

        # 写入cvs文件
        with open(file_path, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'loop_number': select_loop + 86,
                             'MAPE_5min': mape_5, 'MAE_5min': mae_5,
                             'MAPE_10min': mape_10, 'MAE_10min': mae_10,
                             'MAPE_15min': mape_10, 'MAE_15min': mae_10,
                             'MAPE_20min': mape_10, 'MAE_20min': mae_10, })
