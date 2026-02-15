import ctypes
import numpy as np
import time

# Load library
lib = ctypes.cdll.LoadLibrary("./libmatrix.so")

# Setup convolution function
lib.cuda_convolve.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int
]

print("=== Python→CUDA Library Performance ===\n")
print("| M     | N | Time (seconds) |")
print("|-------|---|----------------|")

# Test different sizes
for M in [256, 512, 1024]:
    for N in [3, 5, 7]:
        input_data = np.random.rand(M * M).astype(np.float32) * 255
        filter_data = np.random.rand(N * N).astype(np.float32)
        output_data = np.zeros(M * M, dtype=np.float32)
        
        # Warmup run
        lib.cuda_convolve(input_data, filter_data, output_data, N, M)
        
        # Timed run
        start = time.time()
        lib.cuda_convolve(input_data, filter_data, output_data, N, M)
        end = time.time()
        
        elapsed = end - start
        print(f"| {M:5} | {N} | {elapsed:.6f}     |")

print("All tests completed!")