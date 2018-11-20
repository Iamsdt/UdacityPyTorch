import numpy as np

dt = np.dtype('>i4')
print(dt)
print(dt.type)

dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2, 2))])
print(dt["name"])
print(dt["grades"])

x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dt)

print("Type: ", type(x))

b = x[1]
print(b)
