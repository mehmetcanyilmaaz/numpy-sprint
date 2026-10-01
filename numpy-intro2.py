
import numpy as np

# %% exec31
a = np.array(0.)
b = np.array(0.)
with np.errstate(divide='ignore', invalid='ignore'):
    result = a / b

# only use it this way

# %% exec32
print(np.sqrt(-1) == np.emath.sqrt(-1))
print(np.sqrt(-1))
print(np.emath.sqrt(-1))

# %% exec33
print(np.datetime64('today'))


# %% exe34
print(np.arange('2016-07', '2016-08', dtype='datetime64[D]'))

# %% exec35
import numpy as np

A = np.random.random(10)
B = np.random.random(10)
np.add(A, B, out=B)
np.negative(np.divide(A, 2, out=A))
np.multiply(A, B, out=A)
print(A)
# %% exec36

A = np.random.uniform(1, 10, 5)
print(A)
np.floor(A, out=A)
print(A)


# %% exec37
A = np.zeros((5,5)) + np.arange(5)
print(A)

# %% exec38
def gen(): 
    for i in range(10): yield i

np.fromiter(gen(), dtype=int)


# %% exec39
np.random.uniform(0,1, 10)
np.linspace(0.1, 9.9, 10)
# %% exec40
A = np.random.random(10)
print(A)
np.sort(A)

# %% exec41
A = np.random.random(10) # what np.sum calls under the hood, saves a couple microsecs
np.add.reduce(A)

# %% exec42
A = np.random.random(1)
B = np.random.random(1)
print(A == B)
print(np.allclose(A, B))
print(np.array_equal(A, B))

# %% exec43
A = np.random.random(3)
print(A)
A.flags.writeable = False
A += 1
print(A)
# %% exec44
A = np.random.random((10,2))
print(A)
np.sqrt(A, out=A)
np.arctan(A, out=A)
print(A)


# %% exec45
A = np.random.random(10)
A[np.argmax(A)] = 0
print(A)

# %% exec46
x = np.random.uniform(0, 1, 3)
y = np.random.uniform(0, 1, 3)
z = np.meshgrid(x, y)
print("")
print(z)

# %% exec47
x = np.random.uniform(0, 1, 3)
y = np.random.uniform(0, 1, 3)
D = np.subtract.outer(x, y)   # or  X[:, None] - Y[None, :]
print(D)
C = 1.0 / D
print(C)
# %% exec48
import numpy as np

np.finfo(np.float32)
np.iinfo(np.int32)

# %% exec49

a = np.arange(1000)
np.set_printoptions(precision=2, suppress=True)
print(a)

# %% exec50
a = np.arange(100)
b = 5
np.subtract(a, b, out=a)
np.abs(a, out=a)
c = np.argmin(a)
print(a[c])

# %%
