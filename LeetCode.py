# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   LeetCode.py
@Time    :   2018/11/26 18:52
@Desc    :
"""
import csv

with open(r'D:\桌面\names.csv', 'w', newline='') as csvfile:
    fieldnames = ['time_space', 'MAPE', 'MAE']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
for i in range(5):
    with open(r'D:\桌面\names.csv', 'a+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        time_space = '{}*{}'.format(5,4)
        writer.writerow({'time_space': time_space, 'MAPE': 5, 'MAE': 5})