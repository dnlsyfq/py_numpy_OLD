from __future__ import print_function
import sys
from datetime import datetime
import numpy as np

size = int(sys.argv[1])

"""
def pythonsum(n):
    a = range(n)
    b = range(n)
    c = []

    for i in range(len(a)):
        a[i] = i ** 2
        b[i] = i ** 3
        c.append(a[i] + b[i])

    return c
"""

def pythonsum(n):
    a = []
    b = []
    for i in range(n):
        a.append(i ** 2)
        b.append(i ** 3)
    c = [x+y for x, y in list(zip(a,b))]
    return c


def numpysum(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    c = a + b
    return c.tolist()


start = datetime.now()
c = pythonsum(size)
delta = datetime.now() - start
print("Pythonsum: {}".format(delta.microseconds))
print("The last 2 elements of the sum: {}".format(c[-2:]))


start = datetime.now()
c = numpysum(size)
delta = datetime.now() - start
print("Numpy: {}".format(delta.microseconds))
print("The last 2 elements of the sum: {}".format(c[-2:]))
