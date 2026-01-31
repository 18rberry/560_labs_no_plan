#include <cuda_runtime.h>

// CUDA kernel (same as before)
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
    // Allocate device memory
    float *d_input, *d_filter, *d_output;
    size_t image_size = M * M * sizeof(float);
    size_t filter_size = N * N * sizeof(float);

    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_filter, filter_size);
    cudaMalloc(&d_output, image_size);

    // Copy to device
    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((M + 15) / 16, (M + 15) / 16);

    convolve2D_GPU<<<gridSize, blockSize>>>(d_input, d_filter, d_output, N, M);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
}