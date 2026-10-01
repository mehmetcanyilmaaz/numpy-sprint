#%% setup
import numpy as np
'''
array1 = np.array([1,2,3])

print(array1)
print(array1.shape)
print(array1.dtype)
print(array1.ndim)
'''

# %%

# exercise 1-3
print(np.__version__)

z = np.zeros(10)
print(z)

# %%

# exercise 4

z = np.zeros(10)
print("%d bytes" % z.nbytes)

# %%
#exercise 5

np.info(np.add)



# %% exec6
z = np.zeros(10)
z[4] = 1
print(z)

# %% exec7

z = np.arange(10, 50)
print(z)

# %% exec8

z = np.arange(1, 11)
z = z[::-1]
print(z)

# %% exec 9
z = np.arange(0, 9).reshape(3, 3)
print(z)
# %% exec10
z = np.array([1, 2, 0, 0, 4, 0])
nz = np.nonzero(z) #gives tuple of  --> common in numpy
print(nz)

# %% exec11
z = np.eye(3)
print(z)

# %% exec12
z = np.random.random((3, 3, 3))
print(z)

# %% exec13
z = np.random.random((10, 10))
zmin, zmax = z.min(), z.max()
print(zmin, zmax)

# %% exec14
z = np.random.random(30)
zmean = z.mean()
print(zmean)
# %% exec15
z = np.ones((10, 10))
z[1:-1, 1:-1] = 0
print(z)

# %% exec16
z = np.ones((8,8))
y = np.zeros((10,10))
y[1:-1, 1:-1] = z
print(y)

x = np.pad(z,1)
print(x)
# %% exec17
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1

# %% exec18
z = np.diag([1, 2, 3, 4], -1)
print(z)

# %% exec19

z = np.ones((8,8))
z[::2, ::2] = 0
z[1::2, 1::2] = 0
print(z)

# %% exec20
z = np.unravel_index(100, (6,7,8))
print(z)

# %% exec21
z = np.tile([[0, 1], [1, 0]], (4,4))
print(z)

# %% exec22
z = np.random.random((5,5))
z = (z - z.mean()) / z.std()
print(z)

# %% exec23
color = np.dtype([('r', np.ubyte), ('g', np.ubyte), ('b', np.ubyte), ('a', np.ubyte)])
z= np.zeros(3, dtype=color)
print(z)
# %% exec24
z = np.random.random((5,3))
x = np.random.random((3,2))
y = z @ x
print(y)

# %% exec25
z = np.array([1, 2, 3, 4, 5])
z[(z > 3) & (z < 8)] = -z[(z > 3) & (z < 8)]
print(z)

# %% exec26
# 9 and 10 bc np.sum's second argument is axis not start


# %% exec27

'''
Z**Z. --> elementwise power
2 << Z >> 2. --> bit shifts, << divides by 2^Z and >> multiplies by 4
Z <- Z --> parses as Z < -Z
1j*Z --> 1j is imaginary unit so result is a complex array
Z/1/1 --> divide twice by 1
Z<Z>Z  --> illegal
'''
# %% exec 28
# 0 nan hugegarbagenumber

# %% exec29
z = np.random.random(5)-0.5
x = np.abs(z)
y = np.ceil(x)
z = np.copysign(y,z)
print(z)


# %% exec30
z = np.array([1,2,3,4,5,6,7])
x = np.array([1,5,6,9,10])
print(np.intersect1d(x,z))
