/** @file CUDA_vtkCudaVolumeMapper_renderAlgo.h
 *
 *  @brief Header file with definitions for different common CUDA functions for setting up and
 *         running the ray casting process regardless of its structure
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on May 6th, 2012
 *
 *  @note This is primarily an internal file used by the vtkCudaVolumeMapper and subclasses
 *        to manage the ray casting process.
 *
 */

#ifndef CUDA_VTKCUDAVOLUMEMAPPER_RENDERALGO_H
#define CUDA_VTKCUDAVOLUMEMAPPER_RENDERALGO_H

#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CUDA_containerOutputImageInformation.h"

/** @brief Converts a host memory buffer to a device memory buffer of a different type
 *
 *  @param hostBuffer A floating point buffer 
 *  @param deviceBuffer The area to store the resulting device buffer
 *  @param bufferSize The size of the buffer
 *
 *  @pre The host buffer pointer points to a buffer of size bufferSize
 *
 */
template<typename T, typename S>
void CUDA_castBuffer(T* hostBuffer, S** deviceBuffer, int bufferSize);

template<typename T>
void CUDA_allocBuffer(T* hostBuffer, T** deviceBuffer, int bufferSize);
void CUDA_deallocateMemory( void* memory );


/** @brief Loads the ZBuffer into a 2D texture for checking during the rendering process
 *
 *  @param zBuffer A floating point buffer 
 *  @param zBufferSizeX The size of the z buffer in the x direction
 *  @param zBufferSizeY The size of the z buffer in the y direction
 *
 *  @pre The zBuffer consists only of numbers between 0.0f and 1.0f inclusive
 *
 */
bool CUDA_vtkCudaVolumeMapper_renderAlgo_loadZBuffer(const float* zBuffer, const int zBufferSizeX,
                           const int zBufferSizeY, cudaStream_t* stream);
bool CUDA_vtkCudaVolumeMapper_renderAlgo_unloadZBuffer(cudaStream_t* stream);

/** @brief Loads an random image into a 2D CUDA array for de-artifacting
 *
 *  @param randomRayOffsets A 16x16 array (in 1 dimension, so 256 elements) of random numbers
 *
 *  @pre Each number in randomRayOffsets is between 0.0f and 1.0f inclusive
 *
 */
bool CUDA_vtkCudaVolumeMapper_renderAlgo_loadrandomRayOffsets(const float* randomRayOffsets,
                                cudaStream_t* stream);


/** @brief Unloads the random image from the 2D CUDA array set before
 *
 */
bool CUDA_vtkCudaVolumeMapper_renderAlgo_unloadrandomRayOffsets(cudaStream_t* stream);

#endif
