/** @file CUDA_vtkCudaVolumeMapper_renderAlgo.h
 *
 *  @brief Header file with definitions for different common CUDA functions for setting up and
 *         running the ray casting process regardless of its structure
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
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

/** @brief Loads the ZBuffer into a 2D texture for checking during the rendering process
 *
 *  @param zBuffer A floating point buffer 
 *  @param zBufferSizeX The size of the z buffer in the x direction
 *  @param zBufferSizeY The size of the z buffer in the y direction
 *
 *  @pre The zBuffer consists only of numbers between 0.0f and 1.0f inclusive
 *
 */
extern "C"
void CUDA_vtkCudaVolumeMapper_renderAlgo_loadZBuffer(const float* zBuffer, const int zBufferSizeX, const int zBufferSizeY);

/** @brief Loads an image into a 3D CUDA array which will be bound to a 3D texture for rendering
 *
 *  @param randomRayOffsets A 16x16 array (in 1 dimension, so 256 elements) of random numbers
 *
 *  @pre Each number in randomRayOffsets is between 0.0f and 1.0f inclusive
 *
 */
extern "C"
void CUDA_vtkCudaVolumeMapper_renderAlgo_loadrandomRayOffsets(const float* randomRayOffsets);

#endif
