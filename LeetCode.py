# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   LeetCode.py
@Time    :   2018/11/26 18:52
@Desc    :
"""


class Solution:
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        if not s1:
            return True
        s1 = list(s1)
        s1.sort()
        for i in range(len(s2) - len(s1) + 1):
            list_s2 = list(s2[i: i + len(s1)])
            list_s2.sort()
            if list_s2 == s1:
                return True
        return False

res = Solution()
res.checkInclusion('adc', 'dcda')