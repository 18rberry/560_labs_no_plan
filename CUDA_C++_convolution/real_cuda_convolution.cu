#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA kernel - runs on GPU
__global__ void convolve2D_GPU(float *input_image, float *filter, float *output_image, int N, int M)
{
    // Calculate which pixel THIS thread computes
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check - some threads might be out of bounds
    if (out_row >= M || out_col >= M)
        return;

    float sum = 0.0f;

    // Same convolution logic as CPU!
    for (int f_row = 0; f_row < N; f_row++)
    {
        for (int f_col = 0; f_col < N; f_col++)
        {

            int img_row = out_row + (f_row - N / 2);
            int img_col = out_col + (f_col - N / 2);

            float img_val;
            if (img_row < 0 || img_row >= M || img_col < 0 || img_col >= M)
            {
                img_val = 0.0f; // Zero padding
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

// CPU version (keep for comparison)
void convolve2D_CPU(float *input_image, float *filter, float *output_image, int N, int M)
{
    // Your existing CPU code here
    for (int out_row = 0; out_row < M; out_row++)
    {
        for (int out_col = 0; out_col < M; out_col++)
        {
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
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: %s <image_path> <filter_choice> [filter_size]\n", argv[0]);
        return 1;
    }

    const char *image_path = argv[1];
    const char *filter_choice = argv[2];
    int N = (argc > 3) ? atoi(argv[3]) : 3;

    // Load image
    int width, height, channels;
    unsigned char *img = stbi_load(image_path, &width, &height, &channels, 1);

    if (!img)
    {
        printf("Error: Could not load image\n");
        return 1;
    }

    int M = (width < height) ? width : height;

    // Allocate HOST memory - will be transferred to GPU later
    float *h_input = (float *)malloc(M * M * sizeof(float));
    float *h_filter = (float *)malloc(N * N * sizeof(float));
    float *h_output = (float *)malloc(M * M * sizeof(float));

    // Copy image to host input
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < M; col++)
        {
            h_input[row * M + col] = (float)img[row * width + col];
        }
    }

    // Set up filter (edge detection only for now)
    float edge_3x3[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    for (int i = 0; i < N * N; i++)
    {
        h_filter[i] = edge_3x3[i];
    }

    // Allocate DEVICE (GPU) memory
    float *d_input, *d_filter, *d_output;
    size_t image_size = M * M * sizeof(float);
    size_t filter_size = N * N * sizeof(float);

    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_filter, filter_size);
    cudaMalloc(&d_output, image_size);

    // Copy data from HOST to DEVICE
    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 blockSize(16, 16);                      // 16x16 threads per block
    dim3 gridSize((M + 15) / 16, (M + 15) / 16); // Enough blocks to cover M×M, rounded up

    // Launch kernel and time it
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // gridSize, blockSize initialized gridSize x blockSize blocks. This results in blockSize x gridSize x (thread/block) threads, must be
    // enough to cover however many pixels there are in the image
    convolve2D_GPU<<<gridSize, blockSize>>>(d_input, d_filter, d_output, N, M);
    cudaEventRecord(stop);

    // wait for all GPU threads to finish
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    printf("GPU Convolution (M=%d, N=%d): %.6f seconds\n", M, N, gpu_time / 1000.0f);

    // Copy result back from DEVICE to HOST
    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    // ADD THIS DEBUG CODE:
    printf("Checking first 10 output values:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("  h_output[%d] = %.2f\n", i, h_output[i]);
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Save output (with normalization)
    float min_val = h_output[0], max_val = h_output[0];
    for (int i = 0; i < M * M; i++)
    {
        if (h_output[i] < min_val)
            min_val = h_output[i];
        if (h_output[i] > max_val)
            max_val = h_output[i];
    }

    // ADD THIS TO DEBUG WEIRD BLACK IMAGE ERROR:
    printf("Output range: min=%.2f, max=%.2f\n", min_val, max_val);

    unsigned char *output_img = (unsigned char *)malloc(M * M);
    for (int i = 0; i < M * M; i++)
    {
        float normalized = (h_output[i] - min_val) / (max_val - min_val) * 255.0f;
        output_img[i] = (unsigned char)normalized;
    }

    stbi_write_png("output_gpu.png", M, M, 1, output_img, M);
    printf("Saved output_gpu.png (%dx%d)\n", M, M);

    // Cleanup
    free(h_input);
    free(h_filter);
    free(h_output);
    free(output_img);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    stbi_image_free(img);

    return 0;
}