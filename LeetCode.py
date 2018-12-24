# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   LeetCode.py
@Time    :   2018/11/26 18:52
@Desc    :
"""


class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s.strip()
        s = s.split()
        return ' '.join(s[::-1])


res = Solution()
res.reverseWords('  ')