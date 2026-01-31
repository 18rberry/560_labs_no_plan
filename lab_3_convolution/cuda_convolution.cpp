#include <vector>
#include <iostream>
#include <ctime>
#include <cstring>
using namespace std;
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void convolve2D(float *input_image, float *filter, float *output_image, int N, int M)
{
    // input_image: 2D integer array
    // filter/kernel: filter 2D integer array, IE 3x3
    // output_image: #2D array of output values
    // Assignment instructions: N x N filter matrix, M x M image matrix
    // For each output pixel...
    for (int out_row = 0; out_row < M; out_row++)
    {
        for (int out_col = 0; out_col < M; out_col++)
        {

            float sum = 0.0f;

            // Slide the N×N filter over this position
            for (int f_row = 0; f_row < N; f_row++)
            {
                for (int f_col = 0; f_col < N; f_col++)
                {

                    // Calculate which input pixel to read
                    // Row Major oder formula: 1D_index = (row x width + col)

                    // out_row is our center cell. Calculate to get the first top left cell to start at
                    int img_row = out_row + (f_row - N / 2);
                    int img_col = out_col + (f_col - N / 2);

                    // Get the values (1D indexing!)
                    float img_val;
                    // Edge case: check if img_row or img_col is OOB
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

            // Store result
            output_image[out_row * M + out_col] = sum;
        }
    }
}

// testing function

int main(int argc, char **argv)
{

    if (argc < 3)
    {
        printf("Usage: %s <image_path> <filter_choice> [filter_size]\n", argv[0]);
    }

    //  Get image path from command line
    const char *image_path = argv[1];
    const char *filter_choice = argv[2];
    int N = (argc > 3) ? atoi(argv[3]) : 3;

    // Note: Taking greyscale images via assignment instructions, each pixel = 1 unsigned number
    //  Load test images
    int width, height, channels;
    unsigned char *img = stbi_load(image_path, &width, &height, &channels, 1);

    if (!img)
    {
        printf("Error: Could not load image\n");
        return 1;
    }

    int M = (width < height) ? width : height; // crop the image to a square using ternary operator - guarantee square shape

    // convert our img array into input
    // allocate M x M memory for input
    float *input = (float *)malloc(M * M * sizeof(float));

    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < M; col++)
        {
            input[row * M + col] = img[row * width + col];
        }
    }

    // RIGHT AFTER you create the test pattern, save it
    unsigned char *input_debug = (unsigned char *)malloc(M * M);
    for (int i = 0; i < M * M; i++)
    {
        input_debug[i] = (unsigned char)input[i];
    }
    stbi_write_png("debug_input.png", M, M, 1, input_debug, M);
    printf("Saved debug_input.png - check if it's a centered white square!\n");
    free(input_debug);

    // Allocate arrays

    float *filter = (float *)malloc(N * N * sizeof(float));
    float *output = (float *)malloc(M * M * sizeof(float));

    // Edge detection filter
    float edge_filter_3x3[9] = {
        -1, -1, -1,
        -1, 8, -1,
        -1, -1, -1};

    float edge_filter_5x5[25] = {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 24, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1};
    float edge_filter_7x7[49] = {
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, 48, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1};
    float h_line_filter_3x3[9] = {
        -1, -1, -1,
        8, 8, 8,
        -1, -1, -1};
    float h_line_filter_5x5[25] = {
        -1, -1, -1, -1, -1,
        8, 8, 8, 8, 8,
        -1, -1, -1, -1, -1,
        8, 8, 8, 8, 8,
        -1, -1, -1, -1, -1};
    float h_line_filter_7x7[49] = {
        -1, -1, -1, -1, -1, -1, -1,
        8, 8, 8, 8, 8, 8, 8,
        -1, -1, -1, -1, -1, -1, -1,
        8, 8, 8, 8, 8, 8, 8,
        -1, -1, -1, -1, -1, -1, -1,
        8, 8, 8, 8, 8, 8, 8,
        -1, -1, -1, -1, -1, -1, -1};

    float v_line_filter_3x3[9] = {
        -1, 8, -1,
        -1, 8, -1,
        -1, 8, -1};

    float v_line_filter_5x5[25] = {
        -1, 8, -1, 8, -1,
        -1, 8, -1, 8, -1,
        -1, 8, -1, 8, -1,
        -1, 8, -1, 8, -1,
        -1, 8, -1, 8, -1};
    float v_line_filter_7x7[49] = {
        -1, 8, -1, 8, -1, 8, -1,
        -1, 8, -1, 8, -1, 8, -1,
        -1, 8, -1, 8, -1, 8, -1,
        -1, 8, -1, 8, -1, 8, -1,
        -1, 8, -1, 8, -1, 8, -1,
        -1, 8, -1, 8, -1, 8, -1,
        -1, 8, -1, 8, -1, 8, -1};

    // Then your loop
    for (int i = 0; i < N * N; i++)
    {
        if (strcmp(filter_choice, "edge") == 0)
        {
            if (N == 3)
                filter[i] = edge_filter_3x3[i];
            else if (N == 5)
                filter[i] = edge_filter_5x5[i];
            else if (N == 7)
                filter[i] = edge_filter_7x7[i];
        }
        else if (strcmp(filter_choice, "horizontal") == 0)
        {
            if (N == 3)
                filter[i] = h_line_filter_3x3[i];
            // Note: you'd need 5x5 and 7x7 versions for other sizes
        }
        else if (strcmp(filter_choice, "vertical") == 0)
        {
            if (N == 3)
                filter[i] = v_line_filter_3x3[i];
            // Note: you'd need 5x5 and 7x7 versions for other sizes
        }
    }

    // time convolution
    clock_t start = clock();

    // Run convolution
    convolve2D(input, filter, output, N, M);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU Convolution (M=%d, N=%d): %.6f seconds\n", M, N, elapsed);

    // Convert output back to unsigned char and save
    unsigned char *output_img = (unsigned char *)malloc(M * M);
    // Find min and max FIRST
    float min_val = output[0], max_val = output[0];
    for (int i = 0; i < M * M; i++)
    {
        if (output[i] < min_val)
            min_val = output[i];
        if (output[i] > max_val)
            max_val = output[i];
    }

    printf("Output range: min=%.2f, max=%.2f\n", min_val, max_val);

    // Remove normalization, use absolute scaling
    for (int i = 0; i < M * M; i++)
    {
        // Scale down and clamp
        float val = output[i]; // Adjust this divisor to taste
        if (val < 0)
            val = 0;
        if (val > 255)
            val = 255;
        output_img[i] = (unsigned char)val;
    }

    stbi_write_png("output.png", M, M, 1, output_img, M);
    printf("Saved output.png (%dx%d)\n", M, M);

    // Cleanup
    free(input);
    free(filter);
    free(output);
    free(output_img);
    stbi_image_free(img);

    return 0;
}
