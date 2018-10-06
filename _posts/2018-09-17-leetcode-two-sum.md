---
layout: post
title:  "1.Two Sum"
date:   2018-09-17 17:40:41
categories: leetcode
tags: 数组
mathjax: true
author: Narcissus
---



`思路:`这道题我写了两次代码，虽然题目简单，但是得考虑时空复杂度，否则不能全通过。为什么用python实现呢？因为想到可以用向量化计算nums=nums[:,None]+nums[None]，然后再查找一次即可，结果空间复杂度太大，$O(n^2)$的复杂度,于是直接改循环了,但是for循环中j的变化不要从0开始，不然会重复计算导致时间复杂度太高。

`code:`

```python
import numpy as np
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # if len(nums)==0:
        #     return []
        # nums=np.array(nums)
        # nums=nums[:,None]+nums[None]
        # idx=np.eye(nums.shape[0]))
        # nums[idx[0],idx[1]]=target-1
        # nums=np.where(nums==target)
        # if(nums[0].shape[0]==0):
        #     return []
        # return np.array(nums)[0].tolist()

        for i,val_i in enumerate(nums):
            j=i+1
            for j in range(i+1,len(nums)):
                if (val_i+nums[j])==target:return [i,j]
                j+=1
        return []

```

