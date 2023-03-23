
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "resize.h"
#include <memory.h>
#include <stdio.h>

/*
   This is an example of re-scaling the size of the image in gray-scale floating point 
   format accelerated using cuda on a GPU.

   This example creates a simulated image of 2048×2048. In actual image processing applications 
   you will have an image that comes from a jpeg or tiff file and must be decoded,
   often into an array of RGB bytes or directly into a gray-scale format. 
   Many image processing operations occur on a gray-scale version of the image encoded 
   as floating point, typically of values 0 to 1, or -1 to +1.

   NVIDIA cuda comes with a library of basic image processing functions which are accelerated
   with parallel operations on the GPU, that run on top of the cuda library.

   One of these functions is nppiResize_32f_C1R_Ctx(). The file resize.cpp implements all the 
   memory operations necessary to resize an image using nppiResize_32f_C1R_Ctx().

   The file resize.h provides a simple entry point for an image resize function which can be 
   called from a c program with no knowledge of cuda programming.

*/


int main()
{
    printf("\nsetting up example resizing image in cuda using the nppi library function : nppiResize_32f_C1R_Ctx\n");

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
        printf("\nsuccess running example resizing image in cuda, pixel match rate is %f\n\n",pixelMatchRate);
    }
    else
    {
        printf("\nfailure: pixel match rate is %f\n\n", pixelMatchRate);
    }


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
