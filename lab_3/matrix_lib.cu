#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N)
{
    // Declare shared memory tiles
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate row and column positions
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    //  create accumulator
    float Pvalue = 0.0;

    // Loop over tiles
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m)
    {
        // Load tile A into shared memory
        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        // Load tile B into shared memory
        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

// Wrapper function for Python to call
extern "C" void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N)
{
    size_t size = N * N * sizeof(float);

    // allocate some of the GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // copy our data from the CPU to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // initialize the grid with the block dimensions (using TILE_WIDTH)
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // start the kernel
    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // wait for GPU to run
    cudaDeviceSynchronize();

    // copy result from GPU to CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // clean up memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============ ADD YOUR CONVOLUTION CODE BELOW ============

// Convolution kernel
__global__ void convolve2D_GPU(float *input_image, float *filter, float *output_image, int N, int M)
{
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_row >= M || out_col >= M)
        return;

    float sum = 0.0f;

    for (int f_row = 0; f_row < N; f_row++)
    {
        for (int f_col = 0; f_col < N; f_col++)
        {
            int img_row = out_row + (f_row - N / 2);
            int img_col = out_col + (f_col - N / 2);

            float img_val;
            if (img_row < 0 || img_row >= M || img_col < 0 || img_col >= M)
            {
                img_val = 0.0f;
            }
            else
            {
                img_val = input_image[img_row * M + img_col];
            }

            float filt_val = filter[f_row * N + f_col];
            sum += img_val * filt_val;
        }
    }

    output_image[out_row * M + out_col] = sum;
}

// Wrapper function for Python to call
extern "C" void cuda_convolve(float *h_input, float *h_filter, float *h_output, int N, int M)
{
    float *d_input, *d_filter, *d_output;
    size_t image_size = M * M * sizeof(float);
    size_t filter_size = N * N * sizeof(float);

    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_filter, filter_size);
    cudaMalloc(&d_output, image_size);

    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((M + 15) / 16, (M + 15) / 16);

    convolve2D_GPU<<<gridSize, blockSize>>>(d_input, d_filter, d_output, N, M);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
}