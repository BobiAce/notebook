# coding: utf-8
'''
number one
Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
'''
#思路 1 ******- 时间复杂度: O(N^2)******- 空间复杂度: O(1)******
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
##思路 2 ******- 时间复杂度: O(N)******- 空间复杂度: O(N)******
class Solution(object):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    def twoNum(self,nums,target):
        lookup = {}
        for i,num in enumerate(nums):
            if target - num in lookup:
                return [lookup[target - num],i]
            else:
                lookup[num] = i


"""
python 链表：
"""
class Node:
    def __init__(self, dataval=None):
        self.dataval = dataval
        self.nextval = None

class SLinkedList:
    def __init__(self):
        self.headval = None

lian = SLinkedList()
lian.headval = Node("Mon")
e2 = Node("Tue")
e3 = Node("Wed")

# 连接第一第二个节点
lian.headval.nextval = e2

# 连接第二第三个节点
e2.nextval = e3

print(e2.nextval)
#结果为e3内存地址<__main__.Node object at 0x0000001A0F9644BE0>
print(e2.nextval.dataval)
#结果为e3所代表的值Wed



