# Literals
Tuples: (1, 2, 3)
Lists: [1, 2, 3]
Dicts: {1: 'one', 2: 'two'}
Sets: {1, 2, 3}

""" 
@ TYPE
"""
# int, float, complex
# type annotation
# static type checker: mypy
from typing import List, Tuple, Dict
def annotation(name: str, id: int) -> List[int]:
    print(name)
    print(id)
    return [3,4]

x is y equals id(x) == id(y)

# Immutable objects can be hashable, mutable objects can't be hashable.
spam = {('a', 'b', [1, 2, 3]): 'hello'} # hashing error if immutable object contains mutable list

# Pros of imuutability: garbage collection efficiency, lock-free operation, efficient substructure sharing
 
type(2) == int
type((1,2)) == tuple


""" 
@ SCOPE 
"""

# functions, classes, no block scope
# list comprehension, generator
def lexical_scoping(x):
    def adder(y):
        return x + y
    return adder
lexical_scoping(3)(4)
add3 = lexical_scoping(3)
add3(4) == 7

def global_var():
    global x
    print(x == True) # 10
x = 10

def unboundError():
    print(x == True) # Error
    x = 1
x = 10

# confusing 
# If there is an assignment to a variable inside a function, that variable is considered local.
i = 4
def foo(x):
    def bar():
        print(i, end=' ')
    for i in x:  # i *is* local to Foo, so this is what Bar sees
        print(i, end=' ')
    bar()
foo([1,2,3]) # 1 2 3 3

""" 
@ Decorator
"""
# https://zhuanlan.zhihu.com/p/22810357
# https://stackoverflow.com/questions/17330160/how-does-the-property-decorator-work



""" 
@ Objected Oriented Programming
"""

# CLASS Object
# https://blog.csdn.net/brucewong0516/article/details/79121179
class Animal(object):  # 类对象

    class_attr = 0  # 公有类属性
    __class_private_attr = None  # 私有类属性

    def __init__(self):  # 魔法方法
        self.instance_attr = ‘haha’  # 公有实例属性
        self.__instance_private_att = ‘man’  # 私有实例属性

    def public_method(self):  # 公有方法  self指向实例对象
        pass

    def __private_method(self):  # 私有方法
        pass

    def _import_exclusive_method(self):
        pass

    @classmethod
    def class_method(cls):  # 类方法  cls 指向类对象
        pass

    @staticmethod
    def static_method():  # 静态方法，可以没有参数
        pass

# INHERITENCE
# https://blog.csdn.net/jinxiaonian11/article/details/53727339?utm_source=blogxgwz1

class Base:
    def __init__(self):
        print('Base')
class A(Base):
    def __init__(self):
        super().__init__()
        print('A')

class B(Base):
    def __init__(self):
        super().__init__()
        print('B')

class C(A,B):
    def __init__(self):
        super().__init__()
        print('C')

# using super, Base only initialize once
c = C() # Base B A C

# Super will traverse mro tuple, find next method
print(C.__mro__)
(<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>, <class '__main__.Base'>, <class 'object'>)

--------------------- 
作者：浓酒不消愁，代码渐瘦 
来源：CSDN 
原文：https://blog.csdn.net/jinxiaonian11/article/details/53727339 
版权声明：本文为博主原创文章，转载请附上博文链接！

""" 
@ Functional Programming 
"""

add = lambda x, y: x + y
add(1, 2)

# Unpack argument list
def unpack(self, *args, **kwargs)
args = [1,5]
kwargs = {"a": 10, "b": 20}

# Map
list(map(add, [(1,2), (3,4)]))
# Filter
list(filter(lambda x: x < 0, range(-10, 10)))
# Reduce
from functools import reduce
product = reduce((lambda x, y: x * y), [1, 2, 3, 4])
# sum, min, max
sum(nums)

# zip
zipped = list(zip([1,2], [3,4])) # [(1,3), (2,4)]
unzipped = list(zip(*zipped)) #[(1,2), (3,4)]
for [a, b], c in zip([[1,2], [3,4]], [5,6]):
    print(a, b, c)

# enumerate
for idx, element in enumerate([4,3,2]):
    print(idx, element)


"""
@ File I/O
"""
fp = input() # read from stdin
with open(fp, 'r') as f:
    # data = f.readlines()
    data = f.read().split('\n')
    

with open(fp, 'w') as f:
    f.write(data)

"""
@ Special Usage
"""

# callable
# Good for building API
# https://zhuanlan.zhihu.com/p/33227806
import hashlib

class Hasher(object):
    """
    A wrapper around the hashlib hash algorithms that allows an entire file to
    be hashed in a chunked manner.
    """
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def __call__(self, file):
        hash = self.algorithm()
        with open(file, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), ''):
                hash.update(chunk)
        return hash.hexdigest()

md5    = Hasher(hashlib.md5)
sha1   = Hasher(hashlib.sha1)

from filehash import sha1
print sha1('somefile.txt')



# true if all conditions are satisfied
all(val < x for val in nums) 

# assume nums is sorted, insert val and maintain the sorted order
bisect.insort(nums, val) 

# print without newline
print(string, end = '')

if __name__ == '__main__':
    main()


# pretty print
from pprint import pprint 
pprint(string)

# Close SSL verification
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import requests
# Write this line before creating pyVmomi session
requests.packages.urllib3.disable_warnings()

# export PYTHONHTTPSVERIFY=0
# python your_script
# or
# PYTHONHTTPSVERIFY=0 python your_script
    
if __name__ == '__main__':
    pass


# HTTPS/URL Encoding
from urllib.parse import parse_url, parse_qs
url = 'amount=1000&merchant=123456789&destination[account]=111111&destination[amount]=877'
print(parse_qs(url))



# I/O
import json
with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)
import pickle
pickle.load(open(file_path, 'rb'))
pickle.dump(data, open(file_path, 'wb'))


