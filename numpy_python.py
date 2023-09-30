import numpy as np
from timeit import timeit
# Set the dimensions of the matrices
M, N, K = 128, 128, 128

# Generate two random matrices with random values between 0 and 1
A = np.random.rand(M, N)
B = np.random.rand(N, K)

def matmul_python(A, B):
    C = np.dot(A, B)
    return C
secs = timeit(lambda: matmul_python(A, B), number=2)/2
gflops = ((2*M*N*K)/secs) / 1e9
print(gflops, "GFLOP/s")
