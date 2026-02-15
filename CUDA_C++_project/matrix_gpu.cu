#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }
  C[row * N + col] = sum;
  }
}

int main(int argc, char **argv) {
    // get the matrix size from user input (N) -- otherwise default to n=1024
    int N = (argc > 1) ? atoi(argv[1]) : 1024;

    // how much memory do we need?
    size_t size = N * N * sizeof(float);

    // allocate memory on the CPU (h)
    // malloc: memory allocation in C
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // fill the matrices (A, B) with random values between 0 and 1
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }

    // allocate memory on GPU
    // d_prefix means device memory
    // cuda malloc is memory allocation in GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // copy matrices from CPU to GPU bc GPU can't access CPU memory directly
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // setting up the parallel execution
    // each block has 16 x 16 or 256 threads
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);

    // now start our timers for how long the functions take
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // launch the kernel with gridDim num of blocks and blockDim num of threads
    // and run our matrix multiplication function
    matrixMultiplyGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    // wait for GPU to finish -- then see how long it took to run
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from CPU To GPU bc GPU can't access CPU memory directly!
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // print how long to took to execute
    printf("GPU execution time (N=%d): %f seconds\n", N, milliseconds/1000.0);

    // free up memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}



