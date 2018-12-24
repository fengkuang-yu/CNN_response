# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   LeetCode.py
@Time    :   2018/11/26 18:52
@Desc    :
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

ax = plt.subplot()
sns.set()
data = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\data_all.csv')
data = data.iloc[:, 66:126]
corr_matrix = data.corr()
cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(corr_matrix, ax=ax, cmap=cmap)
ax.set_title('Correlation Between Loop Detectors')
ax.set_xlabel('Loop Detector Number')
ax.set_ylabel('Loop Detector Number')
ax.xaxis.set_tick_params(rotation=45, labelsize=8)
ax.yaxis.set_tick_params(rotation=45, labelsize=8)
plt.show()
corr_matrix.to_csv(r"D:\桌面\corr_matrix.csv")