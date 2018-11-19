import numpy as np
import time
import sys

print("Numpy vs list")
s = range(10)
print(sys.getsizeof(5) * len(s))

print("\nNumpy")
d = np.arange(10)
print(d.size * d.itemsize)

# access list

size = 10000

li = range(size)
li2 = range(size)

a1 = np.arange(size)
a2 = np.arange(size)

start = time.time()

result = [(x, y) for x, y in zip(li, li2)]
print((time.time() - start) * 1000)

start = time.time()

result2 = a1 + a2
