#include "resize.h"
#include "stdio.h"
#include <math.h>
#include <malloc.h>
#include <limits.h>
#include <float.h>
#include "memory.h"
#include "npp.h"


// This function is a cuda program compiled with the nvidia cuda compiler.
// Once built into a library, it can be called from a c program or any program that can call a c library.
// The calling program does not need to know anything about cuda programming.

int resize_Cuda(float* src, float* dst, __int32 src_width, __int32 src_height, __int32 dst_width, __int32 dst_height)
{
    float* dev_src = 0;
    float* dev_dst = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for the two images (one input, one output)    
    size_t memsizeSrc = (size_t)src_width * (size_t)src_height * (size_t)sizeof(float);
    cudaStatus = cudaMalloc((void**)&dev_src, memsizeSrc);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed src!");
        goto Error;
    }

    size_t memsizeDst = (size_t)dst_width * (size_t)dst_height * (size_t)sizeof(float);
    cudaStatus = cudaMalloc((void**)&dev_dst, memsizeDst);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed dst!");
        goto Error;
    }

    // set all destination values to zero
    cudaStatus = cudaMemset((void*)dev_dst, 0, memsizeDst);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
        goto Error;
    }

    // Copy input image from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_src, src, memsizeSrc, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    NppiRect srcROI;
    srcROI.x = 0;
    srcROI.y = 0;
    srcROI.width = src_width;
    srcROI.height = src_height;
    NppiSize srcSize;
    srcSize.width = src_width;
    srcSize.height = src_height;
    int srcStep = src_width * sizeof(float);

    NppiRect dstROI;
    dstROI.x = 0;
    dstROI.y = 0;
    dstROI.width = dst_width;
    dstROI.height = dst_height;
    NppiSize dstSize;
    dstSize.width = dst_width;
    dstSize.height = dst_height;
    int dstStep = dst_width * sizeof(float);

    int eInterpolation = NPPI_INTER_CUBIC;
    NppStreamContext nppStreamContext{};

    NppStatus status = nppiResize_32f_C1R_Ctx(dev_src, srcStep, srcSize,srcROI, dev_dst, dstStep, dstSize, dstROI, eInterpolation, nppStreamContext);

    fprintf(stderr, "NppStatus status = %d\n", status);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.

    cudaStatus = cudaMemcpy(dst, dev_dst, memsizeDst, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_src);
    cudaFree(dev_dst);

    return (int)cudaStatus;
}


