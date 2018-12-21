# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   LeetCode.py
@Time    :   2018/11/26 18:52
@Desc    :
"""
class addSum():
    def sum(self, n):
        if n == 1:
            return 1
        else:
            return n + self.sum(n-1)
a = addSum()
a.sum(4)