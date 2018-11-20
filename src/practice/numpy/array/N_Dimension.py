import numpy as np

n = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

for i in n:
    print(i)

print("Type", n.dtype)
print("Data", n.data)
print("Size", n.size)
print("Shape", n.shape)

print(n[1, 2])

y = n[:, 1]
print(y)

x = np.array([[[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]],
              [[9, 10, 11],
               [12, 13, 14],
               [15, 16, 17]],
              [[18, 19, 20],
               [21, 22, 23],
               [24, 25, 26]]])

print("Type", x.dtype)
print("Data", x.data)
print("Size", x.size)
print("Shape", x.shape)

print("Sum")
print(x.sum(0))
print("\n")
print(x.sum(0), x.sum(1), x.sum(2))
