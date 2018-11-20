# TODO {{
""" 
@ Redo again

https://www.lintcode.com/problem/window-sum/description
https://www.lintcode.com/problem/triangle-count/description
myPow
https://www.lintcode.com/problem/palindrome-partitioning-ii/description # use memo

@ Review Later
https://www.jiuzhang.com/tutorial/segment-tree/237
https://www.jiuzhang.com/solution/strstr-ii/#tag-highlight-lang-python

"""
# TODO Pow(x, n)
# -1 // 2, special numerical operation handling
# see how others handle math problem
# deal with n < 2, e.g. n == -1 , n == 0

# TODO
# naming for max, min, hash, set, dict...
# max_, min_, maximum, minimum

# NOTE 
# python + operator used when O(1)
# }}

# 精華總結 {{
"""
http://www.1point3acres.com/bbs/thread-436925-1-1.html
http://joshuablog.herokuapp.com/Leetcode-%E6%80%BB%E7%BB%93.html
1. 分解问题的角度: fix 某一维度，尝试另一维度上的所有可能
   a. 可能是array的(i, j)pointers, b. 可能是矩形的长与宽, c. 可能是tree的每一个subtree, d. 可能是情景题的每一对pair...
2. 求所有解的, 暴力上backtracking吧
3. 如果问最短/最少的, 先想BFS、DP这对好基友
4. 如果环相关/重复访问, DFS + visited state雄起
5. 如果问连通性, 静态靠DFS/BFS, 动态靠Union-Find
6. 如果有依赖性, 想想Topologic order 和indegree
7. DAG的万能套路 DFS+memo, 再到DP
8. 建图的时候想想vertex, edges/neighbors, cost分别是什么。如果出现cycle, 别忘了给vertex增加状态
9. 树相关, 永远有backtracking 和 pure recursion两条路
10. 遇到字符串/字典/char board相关的, Trie tree总是可以试试的
11. Range里求最大/最小/sum等特征值, Segment tree会是不错的选择
12. Matrix和Array通常都是1. Two Pointers, 2. Sliding Window(fixed & not fixed), 3. DP
13. DP题型往往是: a. 问你可不可以啊, 数量有多少啊, b. 两个string上match来match去的, c. 1D/2D array 相关, d. 博弈游戏
14. 破解DAG cycle想想哪个维度是具有单调性的: 常见的steps, directions, paths
15. Reversed idea非常重要, 可能会帮助你破题: 最长可能是某种最短的反面, 
    最多可能是某种最少的反面, obstacle的反面是reachable, subarray的反面是array中的剩下元素, left的反面是right。
16. Look up别忘了HashMap/HashSet, HashMap + DLL是常见hybrid数据结构。
17. 找规律试试那些旁门左道: 单调Stack/双端Deque
18. 排序大法总是可以试试的
19. 时空复杂度: a. backtracking相关, 想想branching factor和height
                         b. DFS+memo/DP相关, 想想state数量, 以及每个state的cost
                         c. tree相关, 总是要考虑balanced 和 single linked list的 
                         d. array/矩阵相关, 先数数你有多少个for loops 
                         e. binary search application相关, 别忘了check function开销
                         f. stack/queue/deque相关, 常说的吃进去一次又吐出来一次
                         g. Java的string是朵奇葩, string concatenation不是免费的
                         h. 没人知道n是什么, 先告诉别人m，n，k，V，E是什么
20. 比较不同sol的trade offs: a. Time/Space complexity异同
                                             b. online/offline算法
                                             c. pre-computation cost
                                             d. 不同APIs的call frequency差异会导致不同的时间要求
                                             e. extension: 是否适用于generic parameters/stream input
                                             f. 线程安全/large scale

简而言之 DP有六部曲 个人建议面试时写在白板上 清楚明了 一步都不能少 基本上你写出这些 implement起来也非常简单了 
1. definition: dp 或者 dp[j] 表示什么含义，比如largest subarray sum ending at arr, and must include arr. 
   注意语言描述, 包括还是不包括arr/matrix[j]
2. induction rule: dp 的 dependency 是怎么样的，依赖于dp[i-1] 还是 dp[i+1] 还是 dp[k] for all k < i or all k > i 等等，试着过几个小例子推导一下
3. base case:  往往是dp[0]，二维往往是第一行， 第一列，也就是dp[0], dp[0][j]
4. result: 往往的dp[n], max(dp) 等等, 从definition 出发
5. filling order: 也就是你填dp表格的顺序，从induction rule 的 dependency出发 判断的从左到右 还是 从左上到右下
6. optimized: 分为时间和空间两方面。时间的比较难，因为往往需要你重新define dp state 和 induction rule。
   空间比较简单，可以根据induction rule看出来，比如斐波那契数列: dp = dp[i - 1] + dp[i - 2], 那么dp 只依赖于前两个元素，就不需要create 整个 dp array，两个variable即可，空间可以从O(n)优化到O(1)。
最后, 多总题多总结多积累小tips，熟能生巧后dp其实是非常简单，也非常有套路的，一些induction rule 的常见pattern 你一眼就能看出来了。
"""
# }}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
                                Corner Cases                
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
# {{
# BFS
while queue: continue

# DFS/BT/BST
if root is None: return None
if tree_node is not None: continue

# Linked List
if not head or not head.next: return None
      
                          
"""                             TYPE CHECKING 
value equal: ==, != 
reference equal: is, is not
False: 0, "", (), {}, [] or None
True: Others
Check false:
    if x is None: pass
    if x == 0: pass
    if x == []: pass
    if x == {}: pass
    if not input: pass
"""
# Logical Operator
and, or, no

# Numeric type
# Integers have unlimited precision.
float('inf')
-float('inf')

import sys
sys.maxsize
-sys.maxsize-1

from math import floor, ceil
abs(x)
int(x)
int(x, 16) # decimal to hex
float(x)
round(x)
5 / 2 == 2.5
# The result is always rounded towards minus infinity: 
5//2 == 2
1//2 ==  0
(-1)//2 == -1 
1//(-2) == -1 
(-1)//(-2) == 0

# Exception Handling
try:
    pass

except A:
    print('Catch exception')
except:
    raise
else:
    # if no exception
    pass

finally:
    pass

# }}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
                            SEQUENCE OPERATION 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
# {{
min(sequence)
max(sequence)
len(sequence)

# initialize sequence
[False] * 5 # 1D init
[[0] * 3 for _ in range(10)] # 2D init, copy reference

# concat sequence
list1 + list2

# String
s.upper() 
s.lower()
s.replace("old", "new")
s.find("substring") # return index or -1
"substring" in s # return boolean
s.isalpha()
s.isdigit()
s.split(",")
",".join(['first', 'second'])
ord('a') == 97
chr(97) == 'a'
# a - z
list(map(chr, range(ord('a'), ord('z')+1)))
# remove leading or trailing substring
str.strip("pattern")
# split
str.split(',')

# Regex
import re
re.findall('\w+', 'app, book. code') # ['app', 'book', 'code']
re.search('\d{3,5}', '1234d(3333)').group(0) #  first location
# \d == [0-9]
# \w == [a-zA-Z0-9_]
# \W == [^a-zA-Z0-9_]

# List
l.append(x) # equals to l[len(l):] = [x]
l.pop() # remove last element
l.pop(0) # remove first element
l.insert(i, x) # O(N)
l.extend([1,2,3])
l.sort() # in-place
l.sort(reverse=True)
l.sort(key=lambda x: x[1])
l.sort(key=lambda x: (-x[1], x[0])) # sort by multiple attributes
l.index("element")

sorted(l) # iterable -> new list
sorted({1:'a', 2: 'b'}) == [1, 2] # collect keys O(N), sort O(NlogN)
sorted(List, key=lambda x: x[1])

# insertion sort
bisect.insort(nums, val)

# list comprehension
vec = [[1,2,3], [4,5,6], [7,8,9]]
[if num > 0 else 0 num for elem in vec for num in elem]
[num for elem in vec for num in elem if num > 0] # filter

# reverse
nums = [1,2,3,4,5]
nums.reverse() # in-place
list(reversed(nums[2:])) # need list
nums[::-1] # new object 
nums[:1:-2] == [5, 3] # first spot is infered as the last number

# random sample
import random
nums[random.randint(0, len(nums) - 1)]
# }}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
                                DATA STRUCTURE 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
# Built-in Function {{

# List 
l = [1, 2]
t = (3, 4)
array = []
# append
array.append(l)
array.append(t) # [[1, 2], (3, 4)]
# extend
array += l
array += t # [1, 2, 3, 4]
# list extend list
l + list(t) # [1, 2, 3, 4]

# Stack [ ...  top]
stack = [1,2] # 
stack.append(3) # [1,2,3]
stack.pop()
stack[-1]

# Queue [ dequeue ... enqueue]
from collections import deque
queue = deque([])
queue.append(3)
queue.popleft()
queue[0]

# Hash
from collections import defaultdict, Counter
hash = {}
hash = {x: 0 for x in range(10)}
len(hash)
hash[key] = 10
hash.get(key) # 10
hash.get(key, 20) # getOrDefault

count = defaultdict(int) # str, list, lambda: "init"
# if called count['non_key'], then count['non_key'] == 0
c = Counter() 
# if called c['non_key'], then non_key still not in c
c + d # add two counters together:  c[x] + d[x]
c - d # subtract (keeping only positive counts)
c & d # intersection:  min(c[x], d[x]) 
c | d # union:  max(c[x], d[x])

del hash[key]
print(hash.keys())
for k, v in hash.iteritems():
# Check hash contains key  
if dict_.get(key) is not None: pass
# Sequence (e.g. list, dict, tuple, str) Contains
if key in dict_: pass
if e not in list_: pass
def hashfunc(string, hashsize):
    code = 0
    for c in string:
        code = code * 31 + ord(c)
        code = code % hashsize

    return code

# LinkedHashMap
from collections import OrderedDict
OrderedDict()

# Set
s = {}
s = {1, 2, (2,4)}
s.add(1)
s.discard(10) # remove will raise KeyError
x in s
len(s)

# Heap
from heapq import *
min_heap = []
heapify(min_heap) # O(N)
heappush(min_heap, (key1, key2)) # O(log n) , sorted by key1 -> key2
heappop(min_heap) # O(log n)
min_heap[0] # get smallest

max_heap = []
heappush(max_heap, -element)
-heappop(max_heap)
-max_heap[0] # get smallest

def remove(heap, num): # O(N)
    heap[heap.index(num)] = heap[-1]
    heap.pop()
    heapify(heap)

# Top K online algorithm
# min heap: need O(K) space, compare with heap[0]
# max heap: need O(N) space, cannot compare with largest on top

# }}

# LINKED LIST {{

# Reverse
prev, cur = head, head.next
prev.next = None
while cur:
    next = cur.next
    cur.next = prev
    prev, cur = cur, next
new_head = prev

# Slow Fast =>  ... middle(n // 2) | slow ...
slow, fast = head, head.next
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next

# Linked List Cycle
# start from head, head.next

# Formula:
# 1. fast = 2 * slow 
# 2. fast - slow = C

# => fast = 2 * slow = slow + C
# => C = slow

# How far fast need to walk to cycle point?
# fast pos = slow
# C - (slow - L) = L

# }}

# Hash + Singly LinkedList{{

pos = {} # store previous pointer
dummy = tail = ListNode(0)
def LRU_add(num):
    if num in pos:
        remove(num)

    pos[num] = dummy
    dummy.next = ListNode(num)
    pos[dummy.next.next.val] = dummy.next  

def remove(num):
    prev = pos[num]
    del pos[num]

    if prev.next == tail:
        tail = prev
    else:
        prev.next = prev.next.next
        # move back pos of next node of removed node 
        pos[prev.next] = prev

# }}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
                                BINARY SEARCH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
# OOOOO[O] [X]XXXXXX  {{
# [1,1] find 1 inf loop if start < end

# Bsearch on Index
def bsearch(self, nums, target):
    start, end = 0, len(nums) - 1
    while start + 1 < end;
        mid = (start + end) // 2
        
        if nums[mid] < target: 
            start = mid # start belong to left part
        else: 
            end = mid # end belong to right part

    # If target is right part:
    # nums[mid] < target: 
    #    1. [ ... start | end(target) ... ]
    #    2. [ start(target) ... ] <- corner case

    # If target is left part:
    # nums[mid] <= target: 
    #    1. [ ... start(target) | end ... ]
    #    2. [ ... end(target) ] <- corner case

    # find first, check start -> end
    if nums[start] == target: 
        return start
    # find last, check start <- end
    return end

# Bsearch on Value
# e.g. Median of 2 sorted lists / k-th in 2D matrix / copy books 
def bsearch(self, nums, target):
    def count(val): pass
    start, end = min_val, max_val

    while start + 1e-5 < end: # minimum diff
        mid = (start + end) / 2
        if count(mid) < target: 
            start = mid
        else: 
            end = mid

    if count(start) == target: 
        return start
    return end

# }}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
                                TWO POINTERS
main idea:
    swap with left or right when idx is not in the correct part
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# TWO POINTERS -> <-  {{
"""
unstable

1. qsort/qselect
2. two sum series
"""

# Two way partition : ... right | left  ...  
left, right = 0, len(nums) - 1
while left <= right:
    if nums[left] <= pivot: # quick sort use < pivot
        left += 1
    elif nums[right] > pivot: 
        right -= 1
    else:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1
# }}

# QUICK SORT  {{
"""
tricky part when encountering pivot value
nums[left] < pivot: left++
nums[right] > pivot: right--

1. avoid inf loop
if set <= pivot or >= pivot, then when one part 
is always <= pivot or >= pivot, divide and conquer
will divide them into [nums] [] and cause inf loop 

2. balance
can swap left pivot and right pivot to balance the split
, as the result, when [pivot pivot ... pivot],
it will be divided into two subarray and will not cause 
inf loop

each time it is done, 
right | left  or  right | pivot | left

"""
def qsort(self, nums, start, end):
    # remains one element or empty
    if start >= end:
        return
        # return nums[k], since partition finished

    pivot = nums[(start + end) // 2]

    left, right = start, end
    while left <= right: 
        if nums[left] < pivot:
            left += 1
        elif nums[right] > pivot:
            right -= 1
        else:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    # qsort                           # ... right | left ... or ... right == left ...
    self.qsort(nums, start, left - 1) #  left - 1 |
    self.qsort(nums, left, end)       #           |  left
 
    # qselect kth idx
    if k < left: # k in left part
        return self.qselect(nums, start, left - 1, k)
    else: # k in right part
        return self.qselect(nums, left, end, k)

### Other Sorting
def bubble(nums):
    # nums[end] will be the largest in range(0, end+1)
    # From end - 1 to 1
    for end in range(len(nums)-1, 0, -1):
        # From 0 to end-1
        for i in range(end):
            if nums[i] > nums[i+1]:
                nums[i], nums[i+1] = nums[i+1], nums[i]

def insertion(nums):
    for end in range(1, len(nums)):
        i = end
        # optimized, less write operation 
        tmp = nums[i]
        while i > 0 and nums[i-1] > nums[i]:
            nums[i] = nums[i-1]
            i -= 1
        nums[i] = tmp

        # swap
        while i > 0 and nums[i-1] > nums[i]:
            nums[i-1], nums[i] = nums[i], nums[i-1]
            i -= 1
# }}

# TWO POINTERS -> -> {{
"""
stable
1. Two way partition : ... | left  ... 
2. Sliding Window

"""
# Two way partition: [ ... | left ... ] i 
left = 0
for i in range(len(nums)):
    # nums[i] is left part
    if nums[i] < pivot:
        nums[left], nums[i] = nums[i], nums[left] 
        left += 1

# left, right = 0, 0
# while right < len(nums):
    # if nums[right] < pivot:
        # nums[left], nums[right] = nums[right], nums[left]
        # left += 1
    # right += 1

# Three way partition: [ ... | left ... right | i ... ]
left, i, right = 0, 0, len(nums) - 1
while i <= right:
    # nums[i] is right part
    while i <= right and nums[i] > pivot:
        nums[i], nums[right] = nums[right], nums[i]
        right -= 1
    # nums[i] is left part
    if nums[i] < pivot:
        nums[i], nums[left] = nums[left], nums[i]
        left += 1
    i += 1

# }}

# SLIDING WINDOW   {{
# Count: Two Pointers + Hash
# Max/Min: Monotonous Deque
# Median: Heap + Hash
# Window sum + max/min: Heap

# window: ... non pattern ...  0 |  ... pattern characters ...
#                                     ^
#                                     |
#                       window[ch] satisfy pattern[ch] amount
# pattern chars in window always >= 0, 
# window[right] > 0 means we encounter pattern, so count++
# window[left] > 0 means we encounter pattern, so count--

# https://leetcode.com/problems/find-all-anagrams-in-a-string/discuss/92007/Sliding-Window-algorithm-template-to-solve-all-the-Leetcode-substring-search-problem.

window = collections.Counter(p)
count = 0
res = []

left = 0
for i in range(len(s)):
    if window[s[i]] > 0:
        count += 1
    window[s[i]] += 1

    # fix window size == len(p)
    if i < len(p) - 1:
        continue

    # while for dynamic window, if for fixed window
    while count == len(p): # conunt satisfy condition
        res.append(s[left:i+1]) # compute result

        window[s[left]] -= 1
        if window[s[left]] > 0:
            count -= 1
        left += 1

# }}

# TWO POINTERS <- ->  {{

# Palindrome
left, right = middle, middle # odd
left, right = middle, middle + 1 # even
while low >= 0 and high < len(s) and s[low] == s[high]:
    low -= 1
    high += 1
return low >= high # True if palindrom

# }}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
                                   BFS                               
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Tree {{

from collections import deque
if not root: 
    return 0

# level = 0
queue = deque([root])
while queue:
    # level += 1
    for _ in range(len(queue)): # Level Order
        node = queue.popleft()
        traverse(node)
        for child in [node.left, node.right]:
            if not child:
                continue
            queue.append(child)
# }}

UNDIRECTED GRAPH, need visited # {{
"""
Valid Tree: 
    1. len(edges) == len(nodes) - 1
    2. len(visited) == len(nodes), one connected component  
"""

level = 0 # if root element is 1, o.w. -1
queue = deque([root])
visited = set([root])
while queue:
    # level += 1
    for _ in range(len(queue)): # Level Order
        node = queue.popleft()
        traverse(node)
        for child in node.neighbors:
            if child and child not in visited: # Check possible children
                queue.append(child)
                visited.add(child)
# has cycle
# if len(visited) != len(nodes)

# Grid/Matrix
queue = deque([(x, y)])
visited = set([(x, y)])
while queue:
    x, y = queue.popleft()
    traverse(grid[x][y])
    for dx, dy in [[0,1], [0,-1], [1,0], [-1,0]]:
        i, j = x + dx, y + dy
        if 0 <= i < m and 0 <= j < n and (i, j) not in visited:
            queue.append((i, j))
            visited.add((i, j))

 # }}

# TOPOLOGICAL SORT (DAG) {{
def topSort(graph):
    in_degree = {v: 0 for v in graph}
    for v in in_degree:
        for neighbor in v.neighbors:
            id_degree[neighbor] += 1

    # 1. check than one topo order
    # if len(queue) > 1
    queue = [v for v in in_degree if in_degree[v] == 0]
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in node.neighbors:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # has cycle
    # for node in in_degree:
    #     if in_degree[node] != 0:
    #         return False
    return order

# Bidirectional BFS
# Given undirected graph, start and end, O(X ^ N) -> O(2 * X ^(N/2))
# }}

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                    DFS
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# POSTORDER, Divide and Conquer # {{
def dfs(root):
    if not root:
        # return 0 / None

    # divide
    left = dfs(root.left)
    right = dfs(root.right)

    # merge
    res = left + right
    return res

def postorder_iterative(root):
    stack, node = [], root
    while stack or node:
        # push all /\ branch
        while node:
            stack.append(node)
            node = node.left if node.left else node.right

        node = stack.pop()
        visit(node)
        # backtrack from left, then add right 
        if stack and stack[-1].left == node:
            node = stack[-1].right
        else:
            node = None

# }}

# INORDER, BST {{
def inorder_iterative(node):
    stack = []
    while stack or node:
        # push all / branch
        while node:
            stack.append(node)
            node = node.left
        # visit node from stack, move node ->
        node = stack.pop()
        visit(node)
        node = node.right

def next_node(root, target):
    stack = []
    node = root
    while node != target:
        if node.val < target.val:
            node = node.right
        else: # >= target put into stack
            stack.append(node)
            node = node.left
    target == next(node)
    return next(node)

class BSTIterator:
    def __init__(self, root):
        self.stack = []
        # push all / branch
        while root != None:
            self.stack.append(root)
            root = root.left

    def hasNext(self):
        return len(self.stack) > 0

    def next(self):
        stack = self.stack
        node = stack.pop()

        # push all right / branch, from left to right
        right = node.right
        while right:
            stack.append(right)
            right = right.left
        return node

    def prev(self):
        stack = self.stack
        node = stack.pop()

        # push all left \ branch, from right to left
        left = node.left
        while left:
            stack.append(left)
            left = left.right
        return node
# }}

# PREORDER {{
def preorder_iterative(root):
    if not root: return 
    stack = [root]
    while stack:
        node = stack.pop()
        visit(node)
        if node.right: # push right first
            stack.append(node.right)
        if node.left: # then left
            stack.append(node.left)
# }}

# GRAPH DFS, Backtrack Path only, Record Visited forever {{

def dfs(matrix, res, path, x, y, visited):
    for dx, dy in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
        i, j = x + dx, y + dy
        if valid((i, j)) and (i, j) not in visited:
            visited.add((i, j)) # every point should be visited only once
            path.append(matrix[i][j])
            dfs(matrix, res, path, i, j, visited)
            path.pop()

        # dx, dy = -dy, dx
        # Rotation matrix:    | cosø -sinø | , ø = 90º, | 0 -1 | * (x, y) == (-y, x)  
        #                     | sinø  cosø |            | 1  0 |

# cycle detection 
for a, b in edges:
    graph[a].add(b) # Directed Graph
    graph[b].add(a) # Undirected Graph

def dfs(graph, cycle, path, root, parent, visited):
    for child in graph[root]:
        if child not in visited:
            visited.add(child)
            path.append(child)
            dfs(graph, cycle, path, child, root, visited)
            path.pop()
            # visited.remove(child) # shouldn't remove

        # get cycle path
        elif child != parent:
            for i in range(len(path) - 1, -1, -1):
                if path[i] == child:
                    cycle.append(path[i:] + [child])

# }}

# COMBINATION DFS, using START{{
def subset(nums):
    def helper(nums, res, path, start):
        res.append(path[:])
        for i in range(start, len(nums)):
            # i > 0     : distinct subset elements (e.g. [1, 2, 3])
            # i > start : distinct subsets (e.g. [1, 2, 2])
            if i > start nums[i] == nums[i - 1]: 
                continue
            path.append(nums[i])
            helper(nums, res, path, i + 1) # i if repeat choose
            path.pop()

    res = []
    nums.sort() # having repeat
    helper(nums, res, [], 0)
    return res

# }}

# PERMUTATION DFS, using VISITED{{
def permutation(nums):
    def helper(nums, res, path, visited):
        if len(path) == len(nums):
            res.append(path[:])

        for i in range(len(nums)):
            if visited[i]:
                continue
            # [1_, 2_, not visited 2, cur -> 2_, 3] is not allowed
            # i > 0            : distinct permutation elements (e.g. [1, 2, 3])
            # not visited[i-1] : distinct permutations (e.g. [1, 2, 2])
            if i > 0 and nums[i] == nums[i - 1] and i - 1 not in visited:
                continue                                                

            visited.add(i)
            path.append(nums[i])
            helper(nums, res, path, visited)
            path.pop()
            visited.remove(i)

    res = []
    nums.sort() # having repeat
    helper(nums, res, [], set())
    return res

# ITERATIVE PERMUTATION

# prev  _2 4  3 1 (find 2 < 4)
#       _3 4 _2 1
# next  _3 1  2 4 (find 3 > 1)
def next_permutation(nums):
    for pivot in range(len(nums) - 2, -1, -1):
        if nums[pivot] < nums[pivot + 1]:
            for right in range(len(nums) - 1, pivot, -1):
                if nums[pivot] < nums[right]:
                    nums[pivot], nums[right] = nums[right], nums[pivot]
                    nums[pivot + 1:] = nums[:pivot:-1]
                    return nums
                    # return True 

    return nums[::-1]
    # return False

def prev_permutation(nums):
    for pivot in range(len(nums) - 2, -1, -1):
        if nums[pivot] > nums[pivot + 1]:
            for right in range(len(nums), pivot, -1):
                if nums[pivot] > nums[right]:
                    nums[pivot], nums[right] = nums[right], nums[pivot]
                    nums[pivot + 1:] = nums[:pivot:-1]
                    return nums
                    # return True
    return nums[::-1]
    # return False
# }}

# MEMOIZATION: when parameter easy e.g. i, j{{
# word break / wildcard matching return sufix combinations
# palindrom memoization
# word pattern cannot use memo since hash cannot be record
# use hash memo[(i, j)] in some cases
def memoization(s):
    n = len(s)
    memo = [[False] * n for _ in range(n)]
    for i in range(n):
        memo[i][i] = True
    for i in range(1, n):
        memo[i][i-1] = True
    
    for span in range(1, n):
        for i in range(n - span):
            j = i + span
            memo[i][j] = memo[i+1][j-1] and s[i] == s[j]
    return memo

def enumeratePalindrome(s):
    def middle(s, i, j):
        while i >= 0 and j < len(s) and s[i] == s[j]:
            i, j = i -1, j + 1
        return s[i + 1: j] # longest palindrome
    middle(s, mid, mid)
    middle(s, mid, mid+1)
# }}

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                              Union Find

1. M union and find operations on N objects takes O(N + M lg* N) time. 
lg*N similar to O(1)
2. Dynamic Graph

DAG: no cycle 
Tree: DAG + single parent
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Component Nums, Component Size {{

parent = {i: i for i in range(n)}
count = {i: 1 for i in range(n)}
# group = len(count)
def find(x):
    while x != parent[x]:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y): 
    p1, p2 = find(x), find(y)
    if p1 == p2:
        return

    parent[p2] = p1 # can comp p1 p2, then p1 -> p2
    # count[p1] += count[p2]
    # del count[p2]

# cycle detection of undirected graph:
# for v1, v2 in edges:
#   if find(v1) == find(v2): find cycle
#   else: union(v1, v2)


# OPTIMAL UF

# parent = {num: -1 for num in nums} 
# count = len(parent)
# # Quick Find with path compression O(1)
# def find(x): 
    # if parent[x] < 0:
        # return x

    # parent[x] = find(parent[x]) # edge connection
    # # edge value e.g. parent[x].val *= x.val 
    # return parent[x]

# # Quick Union by size O(1) 
# # x -> y, x is parent
# def union(x, y): 
    # p1, p2 = find(x), find(y)
    # if p1 == p2:
        # return

    # if parent[p1] > parent[p2]:
        # p1, p2 = p2, p1

    # parent[p1] += parent[p2]
    # parent[p2] = p1
    # count -= 1


# }}

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Trie

Ending condition:
DFS:
    if i == len(word): return
BFS:
    init dist = -1 
    if dist == len(word): return
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# {{
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

def build(words):
    root = TrieNode()
    for word in words:
        insert(root, word)
    return root

def insert(root, word):
    node = root
    for c in word:
        if c not in node.children:
            node.children[c] = TrieNode()
        node = node.children[c]

    node.is_word = True

def find(root word):
    node = root
    for c in word:
        if c not in node.children:
            return None
        node = node.children[c]

    return node

# }}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
                            MONOTONOUS STACK / DEQUE                               

Find max or left < right / in array : monotonically decrease stack \
Find min or left > right \ in array : monotonically increase stack /
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# MONOTONOUS STACK   {{

# stack([1,2,8,10]).push(5) => stack([1,2,5])

def push(stack, nums, i):
    while stack and nums[stack[-1]] < nums[i]:
        # get result from each pop, stack[-1]
        # res[stack[-1]] = i - stack[-1]
        stack.pop()
    stack.append(i)

# i < j, find minimum j – i such that arr[i] < arr[j] for each element
def min_dist(nums):
    res = [0] * len(nums)
    stack = [] # descending stack # ascending if arr[i] > arr[j]
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]: # > if arr[i] > arr[j]
            res[stack[-1]] = i - stack[-1]
            stack.pop()
        stack.append(i)
    return res

# assert min_dist_asc([34, 8, 10, 3, 2, 80, 30, 33, 1]) == [5, 1, 3, 2, 1, 0, 1, 0, 0]

# i < j, find maximum j – i such that arr[i] < arr[j]
def max_dist(nums):
    res = 0
    stack = [] # descending stack # ascending if arr[i] > arr[j]
    for i in range(len(nums)):
        if not stack or nums[stack[-1]] > nums[i]: # < if arr[i] > arr[j]
            stack.append(i)

    for i in range(len(nums) - 1, -1, -1):
        while stack and nums[stack[-1]] < nums[i]: # > if arr[i] > arr[j]
            res = max(res, i - stack[-1])
            stack.pop()
    return res

# assert max_dist_asc([34, 8, 10, 3, 2, 80, 30, 33, 1]) == 6

# }}

# DEQUE  {{
# Monotone Stack with Left limitation

# Max/Min Sliding Window 
for i in range(k - 1, len(nums)):
    push(dq, nums, i)
    # get result from dq[0]
    # res.append(nums[dq[0]])
    left = i + k - 1
    if dq[0] == nums[left]:
        dq.popleft()

def push(dq, nums, i):
    while dq and nums[dq[-1]] < nums[i]:
        dq.pop()
    dq.append(i)

# }}

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                   Sweep Line / Interval / Segment Tree
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# SWEEP LINE {{
points = []
for i in intervals:
    points.append([i.start, 1])
    points.append([i.end, -1])
points.sort()

# Skyline Problem: sweep x, store y on heap
# Time Intersections, count (start 2, end 1)
# Meeting Rooms II, max(count)
# Merge Intervals
#   1. sweep line and count (start 1, end 1)
#   2. sort by start, if overlap: update end, else: output interval

# }}

# Greedy Sort by End {{
# if overlap (cur_start < prev_end):
# else: prev_end = cur_start 

# Burst Ballons, ignore cur_start < prev_end
# Non-overlapping Intervals, remove cur_start < prev_end

# }}

# Segment Tree {{
# 1. O(logN) interval query sum/max/min, O(logN)  
# 2. cannot use if add / delete array elements
#
# from    [1, 4, 2, 3]
#
#
# to          [0,3]
#            (val=4)
#          /         \
#      [0,1]         [2,3]
#     (val=4)       (val=3)
#     /    \         /    \
#  [0,0]  [1,1]   [2,2]  [3,3]
# (val=1)(val=4) (val=2)(val=3)


class SegmentTreeNode:
    def __init__(self, start, end):
        this.start, this.end = start, end
        this.left, this.right, this.val = None, None, None

# T: O(N), S: O(N)
def build(A, start, end):
    root = SegTreeNode(start, end)
    if start > end:
        return None
    elif start == end:
        root.val = A[start]
    else:
        mid = (start + end) // 2
        root.left = build(A, start, mid)
        root.right = build(A, mid + 1, end)
        root.val = sum/max/min(root.left.val, root.right.val)
    return root

# O(log N)
def query(root, start, end):
    if end < root.start or root.end < start:
        return 0/-inf/inf

    if start <= root.start and root.end <= end:
        return root.val

    left = query(root.left, start, end)
    right = query(root.right, start, end)
    return sum/max/min(left, right)

# O(log N)
def modify(root, index, val):
    if not root:
        return

    if root.start == root.end:
        root.val = val
        return 

    # index in the left part
    if index <= root.left.end: 
        modify(root.left, index, val)
    else:
        modify(root.right, index, val)

    root.val = sum/max/min(root.left.val, root.right.val)

# }}

# Binary Indexed Tree  {{

# }}

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                            Bitwise Operation
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# {{
def count_one(n):
    count = 0
    while n:
        n &= (n - 1) # remove rightmost 1 bit
        count += 1
    return count

def count_ones(n):
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + i % 2
    return dp
# }}

# States {{

https://www.allaboutcircuits.com/textbook/digital/chpt-7/converting-truth-tables-boolean-expressions/
https://leetcode.com/problems/single-number-ii/discuss/43296/An-General-Way-to-Handle-All-this-sort-of-questions.
https://www.cnblogs.com/bjwu/p/9323808.html
Single Number II
collect output is 1 and | alltogether

# original table
current   incoming  next
a b            c    a b
0 0            0    0 0
0 1            0    0 1
1 0            0    1 0
0 0            1    0 1
0 1            1    1 0
1 0            1    0 0

# a table
current   incoming  next
a b            c    a b
1 0            0    1 0
0 1            1    1 0
a = a & ~b & ~c | ~a & b &c

# b table
current   incoming  next
a b            c    a b
0 1            0    0 1
0 0            1    0 1
b = ~a & b & ~c | ~a & ~b &c

mapping
00 => 0
01 => 1
10 => 1

# }}

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                   Big Data / Stream Data (One Pass)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Top K freq {{
# 
# hash + heap
# Largest K: min heap, Smallest K: max heap

# Memory cannot fit

# hash = [memory size array]
# hash[ hash_function(key) % len(hash) ] += 1
# low probability lose accuracy, but long tail will decrease this happen

# Bucketize to use larger granuality


# 295. Find Median from Data Stream
[ ... small in max_heap | larget in min_heap ... ]
len(max_heap) == len(min_heap) or len(max_heap) + 1 == len(min_heap)
# }}

# Reservoir Sampling {{
# Randomly choosing a sample of k items from n items in streaming data
#
# Algo: 对于前k个数，我们全部保留，对于第i（i>k）个数，我们以 k / i 的概率保留第i个数，并以 1 / k  的概率与前面已选择的k个数中的任意一个替换。

def ReservoirSampling(nums, k):
    res = []
    n = len(nums)
    for i in range(k):
        res.append(nums[i])

    for i in range(k, n):
        rand = random.randint(0, i - 1)
        if rand < k:
            res[rand] = nums[i]
    return res

# }}

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        Dynamic Programming
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# SOLVE MATH PROBLEM FIRST {{

# Step 1: (LHS) State Definition, from last state
# Step 2: (RHS) Transition Function
# Step 3: (IF)  Edge Case, initial
# Step 4: (FOR) Compute Order
# Be careful about dp table and original list 

# CATEGORIES
# Max/Min, e.g. path
coin[n] = min(coin[n - c1], coin[n - c2]...) + 1
# Count
count[i][j] = count[i-1][j] + count[i][j-1]
# Exist
for j in range(i-1):
    jump[i] = jump[i] or (jump[j] and A[j] + j >= i)

# Print Path
pi = sizeof(dp) # dp[i][j] choose dp[i-1][??]
dp[i][j] = value
pi[i][j] = choice 

# }}

# OPTIMIZATION {{

### SPACE OPTIMIZATION  

# REDUCE TO 2 DIEMENTION: transition function

# f[i] = f[i-1] + ... , i == new, i - 1 == old
old, now = 0, 1
for i in range(m):
    old, now = now, old

# f[i] = f[i-1] + f[i-2] + ..., i == 
old, prev, now = 0, 1, 2
for i in range(m):
    old, prev, now = prev, now, old

# REDUCE TO 1 DIEMENTION: draw pictures

# From Left to Right
for i in range(left, right):
    old  |               dp[i-1][j]
         |                     |
         |                     v
    now  | dp[i][...] --->  dp[i][j]

# From Right to Left
for i in range(right - 1, -1, -1):
    old  | dp[i-1][...]  dp[i-1][j]
         |              \      |
         |               \     v
    now  |                dp[i][j]

### TIME OPTIMIZATION
# 1. Look at transition function
# 2. Draw picture
# }}

# COORDINATE {{

# Unique Path
dp[0][0] = 1
dp[i][j] = dp[i-1][j] + dp[i][j-1] 

# Maximum Product Subarray
max_prod[i] = max(nums[i], nums[i] * max_prod[i-1], nums[i] * min_prod[i-1])
min_prod[i] = min(nums[i], nums[i] * max_prod[i-1], nums[i] * min_prod[i-1])

# Longest Continuous Increasing Subsequence
dp[i] = max(1, dp[i - 1] | nums[i-1] < nums[i])

# Longest Increasing Subsequence
dp[i] = max(1, dp[j] | 0 <= j < i and nums[j] < nums[i])

# Bomb Enemy
up[i][j] = up[i-1][j] + grid[i-1][j] == 'E'

# }}

# SEQUENCE {{
"""
dp = [0] * (n + 1)
dp[0] = empty
"""

# PREFIX SUM 
prefix = [0] * (n + 1)
for i in range(1, n + 1):
    prefix[i] = prefix[i - 1] + nums[i - 1]
interval_sum(i, j) = prefix[j] - prefix[i - 1]

# Best Time Buy and Sell (Maximum Subarray)
min_sum[i] = min(min_sum[i - 1], prefix[i])
max_sum[i] = max(max_sum[i - 1], prefix[i] - min_sum[i - 1])

# Best Time Buy and Sell III
dp = [[0] * (k + 1) for _ in range(n + 1)]
dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + P[i - 1] - P[i - 2]) # Sell 
dp[i][j] = max(dp[i - 1][j] + P[i - 1] - P[i - 2], dp[i - 1][j - 1]) # Buy

# Paint House
dp = [[float('inf')] * k for _ in range(n + 1)]
dp[0] = [0] * k
dp[i][j] = min(dp[i - 1][k] | k != j) + cost[i - 1][j]

# Digital Flip
dp = [[float('inf')] * 2 for _ in range(n + 1)]
dp[0] = [0] * 2
dp[i][0] = min(dp[i - 1][0], dp[i - 1][1]) + nums[i - 1] == 1
dp[i][1] = dp[i - 1][1] + nums[i - 1] == 0

# }}

# DOUBLE SEQUENCE {{

# Regular Expression Matching
dp = [[False] * (m + 1) for _ in range(n + 1)]
dp[0][0] = True
dp[i][0] = False
for i in range(1, len(p) + 1):
    if p[i - 1] == '*':
        dp[0][i] = dp[0][i - 2]

for i in range(1, m + 1):
    for j in range(1, n + 1):
        if p[j-1] == '*': # match 0 or match .*
            dp[i][j] = dp[i][j - 2] or (dp[i-1][j] and match(s[i-1], p[j-2]))
        else: # match one char
            dp[i][j] = dp[i-1][j-1] and match(s[i-1], p[j-1])

# }}

# PARTITION{{
"""
continous
限定段數 -> 加狀態
"""

# Decode Ways
dp = [0] * (n + 1)
dp[0] = 1
dp[i] = (dp[i - 1] if 1 <= s[i - 1] <= 9) + (dp[i - 2] if 11 <= s[i - 2:i] <=26)

# Word Break
dp = [False] * (n + 1)
dp[0] = True
dp[i] = dp[i] or (dp[j] and s[j:i] in words | max(0, i - max_len) <= j < i)

# Perfect Square
dp = [float('inf')] * (n + 1)
dp[0] = 0
dp[i] = min(dp[i - j * j] | 1 <= j * j <= i) + 1

# Palindrome Partitioning II
dp = [float('inf')] * (n + 1) 
dp[0] = 0
dp[i] = min(dp[j] + 1 | 0 <= j < i and is_palindrome[j][i - 1])

# Copy Books
dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
dp[0] = [0] * (k + 1)
dp[i][k] = min(dp[i][k], max(dp[j][k-1], sum(A[j:i])))

# }}

# BACKPACK {{
"""
dp = [] * (weight + 1)
"""

# Backpack (Max Weight), Space Optimize O(W) <--
# last step: 
dp = [[False] * (weight + 1) for _ in range(n + 1)]
dp[0][0] = True
dp[i][w] = dp[i - 1][w] or dp[i - 1][w - A[i - 1]] # [EXIST] i items can weigh to w

dp = [0] * (weight + 1)
for i in range(len(A)):
    for w in range(weight, A[i] - 1, -1):
        dp[w] = max(dp[w], dp[w - A[i]] + A[i]) # [MAX/MIN]

# Backpack (Combination Count, Distinct Items), Space Optimize O(W) <--
# last step: _ 2 + 4 _ = 6 or 1 + 2 + _ 3 _ = 6
dp = [[0] * (weight + 1) for _ in range(n + 1)]
dp[0][0] = 1
dp[i][w] = dp[i - 1][w] + dp[i - 1][w - A[i - 1]] # [COUNT]

# Backpack (Combination Count, Repeat Items), Space Optimize O(W) <--
# last step: 1 + 3 + 2 + _1_ = 7
dp = [0] * (weight + 1)
dp[0] = 1 # easy to forget this
dp[i] = sum(dp[i - A[w]] for w in range(n)) # [COUNT]

# Backpack (Max Value, Distinct Items)
dp = [[0] * (weight + 1) for _ in range(n + 1)]
dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - W[i - 1]] + V[i - 1]) # i items, exclude i itself

# Backpack (Max Value, Repeat Items)
# dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - k * W[i - 1]] + k * V[i - 1] | 1 <= k) 
dp[i][w] = max(dp[i - 1][w], dp[i][w - W[i - 1]] + V[i - 1]) # i items, include i itself


# }}

# Interval{{
"""
dp[i][j] = dp[i + 1][j - 1]

# Filling Order by length
for length in range(2, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        dp[i][j] = dp[i+1][j-1]

# Tricky way
# [i] = ...[i + 1], <-- from right to left
# [j] = ...[j - 1], --> from left to right 
for i in range(n - 1, -1, -1):
    for j in range(i + 1, n): 
        pass

# 去頭去尾、消去型
"""

# isPalindrome
for length in range(2, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        dp[i][j] = dp[i+1][j-1] and s[i] == s[j]

# Longest Palindromic Subsequence
for length in range(3, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        dp[i][j] = max(dp[i+1][j], dp[i][j-1], dp[i+1][j-1] + 2 | s[i] == s[j])

# Scramble String
for length in range(2, n + 1):
    for i in range(n - length + 1):
        for j in range(n - length + 1):
            for k in range(1, length):
                if dp[i][j][k] and dp[i + k][j + k][length - k]:
                    dp[i][j][length] = True
                    break
                if dp[i][j + length - k][k] and dp[i + k][j][length - k]:
                    dp[i][j][length] = True
                    break

# Burst Balloons
# dp[i][j] = max(dp[i][k] + nums[i] * nums[k] * nums[j] + dp[k][j]) | i < k < j
for length in range(3, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        for k in range(i + 1, j):
            dp[i][j] = max(dp[i][j], dp[i][k] + nums[i] * nums[k] * nums[j] + dp[k][j])

# }}

# GAMBLE{{
"""
Define state from the 1st step
"""
# Coins in a Line 
dp[i] = (not dp[i - 1]) or (not dp[i - 2]) # choose one you lose or choose two you lose

# Coins in a Line III
# S_x = max(- S_y + last coin)
dp = [[0] * n for _ in range(n)]
dp[i][i] = coins[i]
for length in range(2, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        dp[i][j] = max(- dp[i + 1][j] + coins[i], - dp[i][j - 1] + coins[j])

# }}

