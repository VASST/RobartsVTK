/** @file CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo.h
 *
 *  @brief Header file with definitions for different CUDA functions for setting up and running the ray casting process
 *			for digitally reconstruced radiography
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on August 13, 2014
 *
 *  @note This is primarily an internal file used by the vtkCudaDRRImageVolumeMapper to manage the ray casting process
 *
 */

#ifndef CUDA_VTKCUDADRRIMAGEVOLUMEMAPPER_RENDERALGO_H
#define CUDA_VTKCUDADRRIMAGEVOLUMEMAPPER_RENDERALGO_H
#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CUDA_containerOutputImageInformation.h"
struct cudaArray;

/** @brief Compute the image of the volume taking into account occluding isosurfaces returning it in a CUDA-OpenGL compatible texture
 *
 *  @param outputInfo Structure containing information for the rendering process describing the output image and how it is handled
 *  @param renderInfo Structure containing information for the rendering process taken primarily from the renderer, such as camera/shading properties
 *  @param volumeInfo Structure containing information for the rendering process taken primarily from the volume, such as dimensions and location in space
 *
 *  @pre The current frame is less than the number of frames, and is non-negative
 *  @pre CUDA-OpenGL interoperability is functional (ie. Only 1 OpenGL context which corresponds solely to the singular renderer/window)
 *
 */
bool CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_doRender(const cudaOutputImageInformation& outputInfo,
							 const cudaRendererInformation& rendererInfo,
							 const cudaVolumeInformation& volumeInfo,
							 const float CTIntercept, const float CTSlope, const float CTOffset,
							 cudaArray* frame,
							 cudaStream_t* stream);

/** @brief Changes the current volume to be rendered to this particular frame, used in 4D visualization
 *
 *  @param frame The frame (starting with 0) that you want to change the currently rendering volume to
 *
 *  @pre frame is less than the total number of frames and is non-negative
 *
 */
bool CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_changeFrame(const cudaArray* frame, cudaStream_t* stream);

/** @brief Deallocates the frames and clears the container (needed for ray caster deallocation)
 *
 */
void CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_clearImageArray(cudaArray** frame, cudaStream_t* stream);

/** @brief Loads an image into a 3D CUDA array which will be bound to a 3D texture for rendering
 *
 *  @param volumeInfo Structure containing information for the rendering process taken primarily from the volume, such as dimensions and location in space
 *  @param index The frame number of this image
 *
 *  @pre index is between 0 and 99 inclusive
 *
 */
bool CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_loadImageInfo(const float* imageData,
	const cudaVolumeInformation& volumeInfo, cudaArray** frame, cudaStream_t* stream);
#endif
