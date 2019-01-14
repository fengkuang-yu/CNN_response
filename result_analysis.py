# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   result_analysis.py
@Time    :   2018/12/23 16:39
@Desc    :
"""

import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def corr_heat_map():
    plt.figure(figsize=(5, 5))
    ax = plt.subplot()
    sns.set()
    data = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\data_all.csv')
    data = data.iloc[:, 66:126]
    corr_matrix = data.corr()
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(corr_matrix, ax=ax, cmap=cmap)
    ax.set_title('')
    ax.set_xlabel('Loop Detector Number', fontsize=12)
    ax.set_ylabel('Loop Detector Number', fontsize=12)
    ax.xaxis.set_tick_params(rotation=45, labelsize=8)
    ax.yaxis.set_tick_params(rotation=45, labelsize=8)
    plt.show()


def auto_corr():
    plt.figure(figsize=(5, 5))
    ax = plt.subplot()
    sns.set()
    ax.set_xlabel('Time lag', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    data = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\data_all.csv')
    data = data.iloc[:, 96]
    plot_acf(data, lags=200, ax=ax, linewidth=0.3, title='')
    plt.show()


def generate_MAPE_3D(loop_num, pred_time):
    if pred_time == 5:
        temp = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\CNN\{}min\l'
                           r'oop{}_res_error.csv'.format(pred_time, loop_num))
    else:
        temp = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\CNN\{}min\l'
                       r'oop{}_res_error{}.csv'.format(pred_time, loop_num, pred_time))
    mape_pd = pd.DataFrame(temp.MAPE.values.reshape(10, 10),
                           index=['space4', 'space8', 'space12', 'space16', 'space20',
                                  'space24', 'space28', 'space32', 'space36', 'space40'],
                           columns=['time_lag4', 'time_lag8', 'time_lag12', 'time_lag16', 'time_lag20',
                                    'time_lag24', 'time_lag28', 'time_lag32', 'time_lag36', 'time_lag40'])
    mae_pd = pd.DataFrame(temp.MAE.values.reshape(10, 10),
                          index=['space4', 'space8', 'space12', 'space16', 'space20',
                                 'space24', 'space28', 'space32', 'space36', 'space40'],
                          columns=['time_lag4', 'time_lag8', 'time_lag12', 'time_lag16', 'time_lag20',
                                   'time_lag24', 'time_lag28', 'time_lag32', 'time_lag36', 'time_lag40'])
    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig)
    X = np.arange(10)
    Y = np.arange(10)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(mape_pd)[X, Y]
    plt.rcParams['savefig.dpi'] = 100  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率
    ax.plot_surface((X+1)*4, (Y+1)*4, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.set_title("NO.{} Loop Detector {} min Traffic Flow Prediction".format(loop_num, pred_time), fontsize=16)
    ax.set_xlabel("Loop Detector Number", fontsize=16)
    ax.set_ylabel("Observation Time Lags", fontsize=16)
    ax.set_zlabel("Mean Absolute Percentage Error(MAPE)", fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    plt.box()
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig)
    X = np.arange(10)
    Y = np.arange(10)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(mae_pd)[X, Y]
    plt.rcParams['savefig.dpi'] = 100  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率
    ax.plot_surface((X+1)*4, (Y+1)*4, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.set_title("NO.{} Loop Detector {} min Traffic Flow Prediction".format(loop_num, pred_time), fontsize=16)
    ax.set_xlabel("Loop Detector Number", fontsize=16)
    ax.set_ylabel("Observation Time Lags", fontsize=16)
    ax.set_zlabel("Mean Absolute Error(MAE)", fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    plt.box()
    plt.show()

    ax = plt.subplot()
    mape_pd.plot(ax=ax)
    ax.xaxis.set_tick_params(rotation=30, labelsize=12)
    plt.show()

    ax = plt.subplot()
    mape_pd.T.plot(ax=ax)
    ax.xaxis.set_tick_params(rotation=30, labelsize=12)
    plt.show()


def bar_baselines():
    import numpy as np
    import matplotlib.pylab as plt

    plt.style.use('seaborn')
    n_groups = 4

    # running time
    SVR = (2.40, 2.76, 2.47, 2.68)
    SARIMA = (4.03, 4.02, 4.03, 4.03)
    KNN = (0.15, 0.14, 0.15, 0.15)
    ANN = (0.6346, 0.68768, 0.691, 0.6104)
    CNN = (0.816, 0.817, 0.8497, 0.9075)
    STFSA_ANN = (7.71, 11.6, 11.88, 15.71)
    STFSA_CNN = (11.4, 12.33, 13.5, 19.31)

    # MAPE
    # SVR = (7.925,8.356,9.228,9.438)
    # SARIMA = (10.309,10.635,10.907,11.095)
    # KNN = (8.074,8.501,8.646,8.819)
    # ANN = (6.346,8.768,9.621,10.442)
    # CNN = (6.292,8.171,8.497,9.075)
    # STFSA_ANN = (6.548,7.821,8.590,9.010)
    # STFSA_CNN = (6.000,7.365,8.043,8.169)


    fig, ax = plt.subplots(figsize=(8,6))
    plt.rcParams['savefig.dpi'] = 100  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率
    index = np.arange(n_groups)
    index = index + 0.15
    bar_width = 0.2
    opacity = 0.9

    plt.bar(index, SVR, bar_width / 2, alpha=opacity, label='SVR')
    plt.bar(index + 0.5 * bar_width, SARIMA, bar_width / 2, alpha=opacity, label='SARIMA')
    plt.bar(index + 1.0 * bar_width, KNN, bar_width / 2, alpha=opacity, label='KNN')
    plt.bar(index + 1.5 * bar_width, ANN, bar_width / 2, alpha=opacity, label='ANN')
    plt.bar(index + 2.0 * bar_width, CNN, bar_width / 2, alpha=opacity, label='CNN')
    plt.bar(index + 2.5 * bar_width, STFSA_ANN, bar_width / 2, alpha=opacity, label='STFSA+ANN')
    plt.bar(index + 3.0 * bar_width, STFSA_CNN, bar_width / 2, alpha=opacity, label='STFSA+CNN',color='darkorange')

    plt.xlabel('Forecasting Horizon', fontsize=16, color='k')
    # plt.ylabel('Training time(Minutes)', fontsize=16, color='k')
    plt.ylabel('Training Time (Minutes)', fontsize=16, color='k')
    plt.xticks(index-0.05 + 2 * bar_width, ('5min', '10min', '15min', '20min'), fontsize=12, color='k')
    plt.yticks(fontsize=12, color='k')  # change the num axis size
    plt.ylim(0, 25)  # The ceil
    plt.legend(ncol=7, loc=2, mode='expand', fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass