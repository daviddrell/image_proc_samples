
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "resize.h"
#include <memory.h>
#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{

    // sample image resize with cuda npp built-in function

    printf("\nsetting up example resizing image in cuda using the nppi library function : nppiResize_32f_C1R_Ctx\n");

    // This example creates a pretend image of 2048x2048. In actual image processing applications you will have
    // an image that comes from a jpg or tif file and must be decoded into an array of RGB bytes.
    // From there, many image processing operations occure on a gray-scale version of this image
    // encoded as floating point, typically of values 0 to 1, or -1 to +1.
    // 
    // This is an example of re-scaling the size of the image in floating point format.
    // 
    // NVIDIA cuda comes with a libary of basic image processing functions which are accelerated with parallel opertions on the GPU,
    // that run on top of the cuda library.
    // 
    // One of this functions is nppiResize_32f_C1R_Ctx(). The file resize.h provides a simple entry point for an image resize function.
    // 
    // The file resize.cpp implements all the memory operations necessary to resize the provided image using nppiResize_32f_C1R_Ctx().
    // 


    // create a pretend image of 2048x2048. In actual image processing applications you will have provided a real image in floating point
    // If you re-implement with  a real image,
    // you will not need to do this cudaMallocHost step, you will just pass in a float* to a real image that has been malloc()-ed and 
    // initialized with actual pixel values
    int imageWidth = 2048;
    int imageHeight = 2048;
    float* imagePixels = nullptr;
    cudaError_t cudaStatus = cudaMallocHost(&imagePixels,sizeof(float) * imageWidth * imageHeight);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed!\n");
        return 1;
    }

    // the pretend image will be a black background with a single white diagonal line from upper-left to lower-right

    for (int x = 0; x < imageWidth; x++)
    {
        for (int y = 0; y < imageHeight; y++)
        {
            if (x == y)
                imagePixels[x + (y * imageWidth)] = 1;
            else
                imagePixels[x + (y * imageWidth)] = 0;
        }
    }



    // if you are using a real image, the program that creates this image should also allocate
    // the memory for the new scaled down image and pass in a float* to this new image space.
    // but since I am only demonstrating the scale fuction, I will create the memory here using cudaMallocHost()
    // since this program file is actually a .cu file. Normally, this would have been allocated using malloc() from an
    // actual c program and the point passed into the cuda function resize_Cuda.
    int imageSmallerWidth = 1024;
    int imageSmallerHeight = 1024;
    float* imageSmallerPixels = nullptr;
    cudaStatus = cudaMallocHost(&imageSmallerPixels, sizeof(float) * imageSmallerWidth * imageSmallerHeight);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed!\n");
        return 1;
    }

    memset(imageSmallerPixels, 0, (size_t)imageSmallerWidth * imageSmallerHeight * sizeof(float));

    // the call to the example implementation:

    cudaStatus =(cudaError_t) resize_Cuda(imagePixels, imageSmallerPixels, imageWidth, imageHeight, imageSmallerWidth, imageSmallerHeight);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "resize_Cuda failed!\n");
        return 1;
    }

    // lets see if the smaller version of the pretend image looks like the larger version
    float matchCount = 0;
    float shouldMatchCount = imageSmallerWidth * imageSmallerHeight;

    for (int x = 0; x < imageSmallerWidth; x++)
    {
        for (int y = 0; y < imageSmallerHeight; y++)
        {
            //printf("x=%d, y=%d, image=%f\n",x,y, imageSmallerPixels[x + (y * imageSmallerWidth)]);
            if (x == y)
            {
                if (imageSmallerPixels[x + (y * imageSmallerWidth)] > 0.95f)
                    matchCount += 1;
                if(imageSmallerPixels[x + (y * imageSmallerWidth)] != 1.0)
                    printf("expected 1.0 but got : x=%d, y=%d, image=%f\n",x,y, imageSmallerPixels[x + (y * imageSmallerWidth)]);
            }
            else
            {
                if (imageSmallerPixels[x + (y * imageSmallerWidth)] < 0.05f)
                    matchCount += 1;
                if (imageSmallerPixels[x + (y * imageSmallerWidth)] != 0.0)
                    printf("expected 0.0 but got : x=%d, y=%d, image=%f\n", x, y, imageSmallerPixels[x + (y * imageSmallerWidth)]);

            }

        }
    }

    float pixelMatchRate = matchCount / shouldMatchCount;

    if (pixelMatchRate > 0.9f)
    {
        printf("\nsuccess: running example resizing image in cuda, pixel match rate is %f\n\n",pixelMatchRate);
    }
    else
    {
        printf("\nfailure: running example resizing image in cuda, pixel match rate is %f\n\n", pixelMatchRate);
    }



    // default cuda project sample code:

    printf("setting up sample cuda kernel that is provided by nvidia: \n");

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    printf("success running sample cuda kernel that is provided by nvidia: \n");

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
