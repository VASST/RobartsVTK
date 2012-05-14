/** @file CUDA_vtkCuda2DVolumeMapper_renderAlgo.h
 *
 *  @brief Header file with definitions for different CUDA functions for setting up and running the ray casting process
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
 *  @note First documented on March 27, 2011
 *
 *  @note This is primarily an internal file used by the vtkCUDAVolumeMapper to manage the ray casting process
 *
 */

#ifndef CUDA_VTKCUDA2DVOLUMEMAPPER_RENDERALGO_H
#define CUDA_VTKCUDA2DVOLUMEMAPPER_RENDERALGO_H
#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CUDA_containerOutputImageInformation.h"
#include "CUDA_container2DTransferFunctionInformation.h"

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
extern "C"
bool CUDA_vtkCuda2DVolumeMapper_renderAlgo_doRender(const cudaOutputImageInformation& outputInfo,
							 const cudaRendererInformation& rendererInfo,
							 const cudaVolumeInformation& volumeInfo,
							 const cuda2DTransferFunctionInformation& transInfo );

/** @brief Changes the current volume to be rendered to this particular frame, used in 4D visualization
 *
 *  @param frame The frame (starting with 0) that you want to change the currently rendering volume to
 *
 *  @pre frame is less than the total number of frames and is non-negative
 *
 */
extern "C"
bool CUDA_vtkCuda2DVolumeMapper_renderAlgo_changeFrame(const int frame);

/** @brief Prepares the container for the frame at the initialization of the renderer
 *
 */
extern "C"
void CUDA_vtkCuda2DVolumeMapper_renderAlgo_initImageArray();

/** @brief Deallocates the frames and clears the container (needed for ray caster deallocation)
 *
 */
extern "C"
void CUDA_vtkCuda2DVolumeMapper_renderAlgo_clearImageArray();

/** @brief Loads the RGBA 2D transfer functions into texture memory
 *
 *  @param volumeInfo Structure containing information for the rendering process taken primarily from the volume, such as dimensions and location in space
 *  @param FunctionSize The size of each dimension of each transfer function
 *  @param redTF A floating point buffer containing the red transfer function
 *  @param greenTF A floating point buffer containing the green transfer function
 *  @param blueTF A floating point buffer containing the blue transfer function
 *  @param alphaTF A floating point buffer containing the opacity transfer function
 *
 *  @pre Each transfer function is square with the intensities separated by 1, and gradients by FunctionSize
 *
 */
extern "C"
bool CUDA_vtkCuda2DVolumeMapper_renderAlgo_loadTextures(const cuda2DTransferFunctionInformation& transInfo,
								  float* redTF, float* greenTF, float* blueTF, float* alphaTF);

/** @brief Loads an image into a 3D CUDA array which will be bound to a 3D texture for rendering
 *
 *  @param volumeInfo Structure containing information for the rendering process taken primarily from the volume, such as dimensions and location in space
 *  @param index The frame number of this image
 *
 *  @pre index is between 0 and 99 inclusive
 *
 */
extern "C"
bool CUDA_vtkCuda2DVolumeMapper_renderAlgo_loadImageInfo(const float* imageData, const cudaVolumeInformation& volumeInfo, const int index);

#endif
