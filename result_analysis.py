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
    n_groups = 5
    SVR = (0.84472049689441, 0.972477064220183, 1.0, 0.9655172413793104, 0.970970970970971)
    SARIMA = (1.0, 0.992992992992993, 1.0, 0.9992348890589136, 0.9717125382262997)
    KNN = (0.70853858784893, 0.569731081926204, 0.8902900378310215, 0.8638638638638638, 0.5803008248423096)
    STFSA_CNN = (0.90786948176583, 0.796122576610381, 0.8475120385232745, 0.8873762376237624, 0.5803008248423096)

    fig, ax = plt.subplots(figsize=(8,6))
    index = np.arange(n_groups)
    bar_width = 0.3
    opacity = 0.4

    rects1 = plt.bar(index, SVR, bar_width / 2, alpha=opacity, label='SVR')
    rects2 = plt.bar(index + 0.5 * bar_width, SARIMA, bar_width / 2, alpha=opacity, label='SARIMA')

    rects3 = plt.bar(index + 1.0 * bar_width, KNN, bar_width / 2, alpha=opacity, label='KNN')
    rects4 = plt.bar(index + 1.5 * bar_width, STFSA_CNN, bar_width / 2, alpha=opacity, label='STFSA+CNN')

    # plt.xlabel('Category', fontsize=16)
    plt.ylabel('Training time', fontsize=16)
    plt.title('Scores by group and Category')
    # plt.xticks(index - 0.2+ 2*bar_width, ('balde', 'bunny', 'dragon', 'happy', 'pillow'))
    plt.xticks(index - 0.2 + 2 * bar_width, ('5min', '10min', '15min', '20min', 'pillow'), fontsize = 12)
    plt.yticks(fontsize=12)  # change the num axis size
    plt.ylim(0, 1.5)  # The ceil
    # plt.legend(bbox_to_anchor=(0,-0.15,1,0), ncol=4, loc=2, mode='expand',borderaxespad=0)
    plt.legend(ncol=4, loc=2, mode='expand', borderaxespad=0)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass