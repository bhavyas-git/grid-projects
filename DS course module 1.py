#!/usr/bin/env python
# coding: utf-8

# In[55]:


# ----- Two sums -----
def twoSum(nums, target):
    nummap = {}
    n = len(nums)

    for i in range(n):
        comp = target - nums[i]
        if comp in nummap:
            return [nummap[comp], i]
        nummap[nums[i]] = i

    return []  #no solution

nums = [3,2,4]
twoSum(nums,6)


# In[100]:


# ---- dup detection ----
def dup(nums):
    hash = {}
    n = len(nums)

    for i in range(n):
        if nums[i] in hash:
            return True
        hash[nums[i]] = i

    return False

nums = [1,2,3,1]
dup(nums)


# In[158]:


# ---- best stock options ----
def stock(prices):
    buy = prices[0]
    profit = 0

    for i in range(1, len(prices)):
        if prices[i]<buy:
            buy = prices[i]
        elif  prices[i] - buy > profit:
            profit = prices[i] - buy

    return profit

prices = [7,1,5,3,6,4]
stock(prices)


# In[178]:


# ---- valid anagram ----
def isAnagram(s, t):  
    if len(s) != len(t):
        return False

    counter = {}

    for i in s:
        counter[i] = counter.get(i, 0) + 1

    for i in t:
        if i not in counter or counter[i] == 0:
            return False
        counter[i] = counter[i] - 1

    return True

s = "anagram"
t = "nagaram"
isAnagram(s, t)


# In[190]:


# ---- valid parenthesis ----
def isValid(s):
        stack = []
        mapping = {")":"(", "}":"{", "]":"["}

        for char in s:
            if char in mapping.values():
                stack.append(char)
            elif char in mapping.keys():
                if not stack or mapping[char] != stack.pop():
                    return False

        return not stack

s = "{"
isValid(s)


# In[210]:


# ---- Product of array except self ----
def productExceptSelf(nums):
    new = []
    product = 1
    n = len(nums)

    for i in range(n):
        product *= nums[i]
    for i in range(n):
        new.append(product / nums[i])

    return new

nums = [1,2,3,4]
productExceptSelf(nums)


# In[213]:


# ---- Max sub array ----
def maxSubArray(nums):            
        sum = nums[0]
        total = 0

        for n in nums:
            if total < 0:
                total = 0
            total += n
            sum = max(sum, total)

        return sum

nums = [-2,1,-3,4,-1,2,1,-5,4]
maxSubArray(nums)


# In[215]:


# ---- 3Sum ----
def threeSum(nums):
        nums.sort()
        res = []
        n = len(nums)

        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            j, k = i + 1, n - 1

            while j < k:
                total = nums[i] + nums[j] + nums[k]

                if total == 0:
                    res.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1

                    # Skip duplicate second elements
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1

                elif total < 0:
                    j += 1
                else:
                    k -= 1

        return res

nums = [-1,0,1,2,-1,-4]
threeSum(nums)


# In[218]:


# ---- merge intervals ----
def merge(intervals):
        intervals.sort()  # Sort by start time
        merged = []
        prev = intervals[0]

        for i in range(1, len(intervals)):
            if intervals[i][0] <= prev[1]:  # Overlap
                prev[1] = max(prev[1], intervals[i][1])  # Merge
            else:
                merged.append(prev)
                prev = intervals[i]

        merged.append(prev)
        return merged

intervals = [[1,3],[2,6],[8,10],[15,18]]
merge(intervals)


# In[224]:


# ---- group anagrams ----
from collections import defaultdict
def groupAnagrams(strs):    
        ans = defaultdict(list)

        for s in strs:
            key = "".join(sorted(s))
            ans[key].append(s)

        return list(ans.values())

strs = ["eat","tea","tan","ate","nat","bat"]
groupAnagrams(strs)


# In[296]:


# ---- reversed linked list ----
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverseList(head):
    # convert list to linked list
    dummy = ListNode(0)
    curr = dummy
    for x in head:
        curr.next = ListNode(x)
        curr = curr.next

    head = dummy.next

    # reverse linked list
    prev = None
    curr = head

    while curr:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp

    # convert back to normal list
    ans = []
    curr = prev
    while curr:
        ans.append(curr.val)
        curr = curr.next

    return ans


# input
head = [1,2,3,4,5]

print(reverseList(head))


# In[299]:


# ---- linked list cycle ----
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def hasCycle(head, pos):
    # convert list to linked list
    dummy = ListNode(0)
    curr = dummy
    nodes = []

    for x in head:
        curr.next = ListNode(x)
        curr = curr.next
        nodes.append(curr)

    head = dummy.next

    # create cycle if pos exists
    if pos != -1:
        curr.next = nodes[pos]

    # detect cycle (Floyd slow-fast)
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True

    return False


# input
head = [1,2]
pos = 0   # cycle at index 1

print(hasCycle(head, pos))


# In[258]:


# ---- container with most water ----
def maxArea(height):
        max_area = 0
        left = 0
        right = len(height) - 1

        while left < right:
            max_area = max(max_area, (right - left) * min(height[left], height[right]))

            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return max_area

height = [1,8,6,2,5,4,8,3,7]
maxArea(height)


# In[261]:


# ---- find min in sorted rotated array -----
def findMin(nums):

        left = 0
        right = len(nums) - 1

        while left < right:
            mid = (left + right) // 2

            if nums[mid] <= nums[right]:
                right = mid
            else:
                left = mid + 1

        return nums[left]

nums = [3,4,5,1,2]
findMin(nums)


# In[264]:


# ---- longest repeating character replacement ----
from collections import defaultdict
def characterReplacement(s, k):
        m = len(s)
        longest = 0 
        left = 0
        counts = defaultdict(int)
        maxf = 0

        for right in range(m) : 
            counts[s[right]] +=1
            maxf = max(maxf , counts[s[right]])
            while (right-left+1) - maxf > k : 
                counts[s[left]] -= 1
                left += 1 
            longest = max(longest , right-left+1)

        return longest

s = "AABABBA"
k = 1
characterReplacement(s, k)


# In[279]:


# ---- longest substring without repeating chars -----
from collections import deque
def lengthOfLongestSubstring(s):
        res = 0
        q = deque()
        for c in s:
            if c in q:
                while q.popleft() != c:
                    pass
            q.append(c)
            res = max(res, len(q))

        return res

s = "abcabcbb"
lengthOfLongestSubstring(s)


# In[276]:


# ---- number of islands ----
class Solution:

    def numIslands(self, grid):
        if not grid:
            return 0

        count = 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    self.dfs(grid, i, j)
                    count += 1
        return count

    def dfs(self, grid, i, j):
        if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] != "1":
            return

        grid[i][j] = "#"
        self.dfs(grid, i+1, j)
        self.dfs(grid, i-1, j)
        self.dfs(grid, i, j+1)
        self.dfs(grid, i, j-1)

grid = [
["1","1","1","1","0"],
["1","1","0","1","0"],
["1","1","0","0","0"],
["0","0","0","0","0"]
]

sol = Solution()
print(sol.numIslands(grid))


# In[280]:


# ---- palidromic substrings ----
def countSubstrings(s):
        n = len(s)
        res = 0

        def expand(i, j):
            nonlocal res
            while i >= 0 and j < n and s[i] == s[j]:
                res += 1
                i -= 1
                j += 1

        for k in range(n):
            expand(k, k)  # Odd length
            expand(k, k+1)  # Even length
        return res

s = "abc"
countSubstrings(s)


# In[300]:


# ---- palindromic substrings ----
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def removeNthFromEnd(head, n):
    # convert list to linked list
    dummy = ListNode(0)
    curr = dummy
    for x in head:
        curr.next = ListNode(x)
        curr = curr.next

    head = dummy.next

    # logic
    res = ListNode(0, head)
    slow = res
    fast = head

    for _ in range(n):
        fast = fast.next

    while fast:
        fast = fast.next
        slow = slow.next

    slow.next = slow.next.next

    # convert back to list
    ans = []
    curr = res.next
    while curr:
        ans.append(curr.val)
        curr = curr.next

    return ans


# input
head = [1,2,3,4,5]
n = 2

print(removeNthFromEnd(head, n))


# In[ ]:




