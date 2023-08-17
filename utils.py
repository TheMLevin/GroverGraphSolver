import math
from functools import reduce

def angle(a,b):
    return 2 * math.asin(math.sqrt(a / b))

def in_binary(n, d):
    return n % 2**(d + 1) >= 2**d

def prod(lst):
    return reduce(lambda a, b: a * b, lst)