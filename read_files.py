# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   read_files.py
@Time    :   2019/1/12 22:42
@Desc    :
"""

import os
import pandas as pd
min = 20
file_path = r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\CNN\{}min'.format(min)
file_list = os.listdir(r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation\data\CNN\{}min'.format(min))
file_list = file_list[-8:]+file_list[0:2]
list_all = []
for cur in file_list:
    file = os.path.join(file_path, cur)
    temp = pd.read_csv(file)
    list_all.append(temp.MAE[34])
a = pd.DataFrame(list_all)
a.to_csv(r'D:\桌面\temp{}.csv'.format(min))
if __name__ == '__main__':
    pass