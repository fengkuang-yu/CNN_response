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


def corr_heat_map():
    ax = plt.subplot()
    sns.set()
    data = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\data_all.csv')
    data = data.iloc[:, 66:126]
    corr_matrix = data.corr()
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(corr_matrix, ax=ax, cmap=cmap)
    ax.set_title('Correlation Between Loop Detectors')
    ax.set_xlabel('Loop Detector Number')
    ax.set_ylabel('Loop Detector Number')
    ax.xaxis.set_tick_params(rotation=45, labelsize=8)
    ax.yaxis.set_tick_params(rotation=45, labelsize=8)
    plt.show()
    corr_matrix.to_csv(r"D:\桌面\corr_matrix.csv")


if __name__ == '__main__':
    temp = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\10min\loop96_res_error10.csv')
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
    fig = plt.figure(figsize=(10, 6))
    ax = Axes3D(fig)
    X = np.arange(10)
    Y = np.arange(10)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(mape_pd)[X, Y]
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.set_title("NO.96 Loop Detector 10 min Traffic Flow Prediction", fontsize=16)
    ax.set_xlabel("Loop Detector Number", fontsize=16)
    ax.set_ylabel("Observation Time Lags", fontsize=16)
    ax.set_zlabel("Mean Absolute Percentage Error(MAPE)", fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    plt.box()
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    ax = Axes3D(fig)
    X = np.arange(10)
    Y = np.arange(10)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(mae_pd)[X, Y]
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.set_title("NO.96 Loop Detector 15 min Traffic Flow Prediction", fontsize=16)
    ax.set_xlabel("Loop Detector Number", fontsize=16)
    ax.set_ylabel("Observation Time Lags", fontsize=16)
    ax.set_zlabel("Mean Absolute Error(MAE)", fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    plt.box()
    plt.show()
