#ifndef RESIZE_H
#define RESIZE_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_CLIB_EXPORTS
#ifdef CUDA_CLIB_EXPORTS
#define CUDA_CLIB_API __declspec(dllexport) 
#else
#define CUDA_CLIB_API __declspec(dllimport) 
#endif


    
    /// <summary>
    /// resize_Cuda() implements a cuda image resize using the NVIDIA provided function nppiResize_32f_C1R_Ctx().
    /// from a normal c program, this function can be called with just the memory pointers for the images and
    /// the image sizes.
    /// 
    /// The memory for both the source and output images must be allocated and managed by the caller.
    /// The source image memory should be first initialized with the source image pixels.
    /// 
    /// The output memory must be allocated but does not need to be intialized to any values.
    /// The output memory will contain the scaled down image pixels after the call.
    /// 
    /// The scaling algorithm is hard-coded to NPPI_INTER_CUBIC. You could make this an additional parameter to the call.
    /// 
    /// </summary>
    /// <param name="src">pointer to source image in floating point pixels.</param>
    /// <param name="dst">pointer to malloc()-ed memory for the output image</param>
    /// <param name="src_width">pixel columns on source image</param>
    /// <param name="src_height">pixel rows on source image</param>
    /// <param name="dst_width">pixel colums on the output image</param>
    /// <param name="dst_height">pixel row on the output image</param>
    /// <returns></returns>
    CUDA_CLIB_API int resize_Cuda(float* src, float* dst, __int32 src_width, __int32 src_height, __int32 dst_width, __int32 dst_height);

#ifdef __cplusplus
}
#endif
#endif
