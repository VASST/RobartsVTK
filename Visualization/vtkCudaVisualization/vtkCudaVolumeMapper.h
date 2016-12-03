/*=========================================================================

  Robarts Visualization Toolkit

  Copyright (c) 2016 Virtual Augmentation and Simulation for Surgery and Therapy, Robarts Research Institute

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
  TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
  CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.

=========================================================================*/

/**
 *  @file vtkCudaVolumeMapper.h
 *  @brief Header file defining a volume mapper (ray caster) using CUDA kernels for parallel ray calculation
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 29, 2011
 */

#ifndef __vtkCudaVolumeMapper_H
#define __vtkCudaVolumeMapper_H

// Local includes
#include "CUDA_containerOutputImageInformation.h"
#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CudaObject.h"
#include "vtkCudaVisualizationExport.h"

// VTK includes
#include <vtkSmartPointer.h>
#include <vtkVersion.h>
#include <vtkVolumeMapper.h>

// STL includes
#include <map>

class vtkCudaOutputImageInformationHandler;
class vtkCudaRendererInformationHandler;
class vtkCudaVolumeInformationHandler;
class vtkMatrix4x4;
class vtkPlane;
class vtkPlaneCollection;
class vtkPlanes;
class vtkRenderer;
class vtkTransform;
class vtkVolume;

// This is the maximum number of frames, may need to be set
#define VTKCUDAVOLUMEMAPPER_UPPER_BOUND 30

/** @brief vtkCudaVolumeMapper is a volume mapper, taking a set of 3D image data objects, volume and renderer as input and creates a 2D ray casted projection of the scene which is then displayed to screen
 *
 */
class vtkCudaVisualizationExport vtkCudaVolumeMapper : public vtkVolumeMapper, public CudaObject
{
public:
  vtkTypeMacro(vtkCudaVolumeMapper, vtkVolumeMapper);

  /** @brief Sets the 3D image data for the first frame in the 4D sequence
   *  @param image The 3D image data.
   *  @pre All dataset being rendered are the same size, anatomy, patient and modality
   */
  virtual void SetInputData(vtkImageData* image);

  /** @brief Sets the 3D image data for a particular frame in the 4D sequence
   *  @param image The 3D image data.
   *  @param frame The desired frame number when this data is rendered
   *  @pre All dataset being rendered are the same size, anatomy, patient and modality
   */
  void SetInputData(vtkImageData* image, int frame);
  virtual void SetInputInternal(vtkImageData* image, int frame) = 0;

  /** @brief Sets the 3D image data for the first frame in the 4D sequence */
  virtual vtkImageData* GetInput();

  /** @brief Gets the 3D image data for a particular frame in the 4D sequence */
  vtkImageData* GetInput(int frame);

  /** @brief Clears all the frames in the 4D sequence */
  void ClearInput();
  virtual void ClearInputInternal() = 0;

  /** @brief Uses the provided renderer and volume to render the image data at the current frame
   *  @note This is an internal method used primarily by the rendering pipeline
   */
  void Render(vtkRenderer* renderer, vtkVolume* volume);

  /** @brief Perform specific rendering process
   *  @note This is an internal method used primarily by the raycasting hierarchy structure
   */
  virtual void InternalRender(vtkRenderer* ren, vtkVolume* vol,
                              const cudaRendererInformation& rendererInfo,
                              const cudaVolumeInformation& volumeInfo,
                              const cudaOutputImageInformation& outputInfo) = 0;

  /** @brief Sets how the image is displayed which is passed to the output image information handler
   *  @param scaleFactor The factor by which the screen is under sampled in each direction (must be equal or greater than 1.0f, where 1.0f means full sampling)
   */
  void SetRenderOutputScaleFactor(float scaleFactor);

  /** @brief Set the strength and sensitivity parameters of the non-photo realistic shading model which is given to the renderer information handler
   *  @param darkness Floating point between 0.0f and 1.0f inclusive, where 0.0f means no shading, and 1.0f means maximal shading
   *  @param a The shading start value
   *  @param b The shading stop value
   */
  void SetCelShadingConstants(float darkness, float a, float b);

  /** @brief Set the strength and sensitivity parameters of the non-photo realistic shading model which is given to the renderer information handler
   *  @param darkness Floating point between 0.0f and 1.0f inclusive, where 0.0f means no shading, and 1.0f means maximal shading
   *  @param a The shading start value
   *  @param b The shading stop value
   */
  void SetDistanceShadingConstants(float darkness, float a, float b);

  /** @brief Changes the next frame to be rendered to the provided frame
   *  @param frame The next frame to be rendered
   *  @pre frame is a non-negative integer less than the total number of frames
   */
  void ChangeFrame(int frame);
  virtual void ChangeFrameInternal(int frame) = 0;

  /** @brief Gets the current frame being rendered
   *  @post frame is a non-negative integer less than the total number of frames
   */
  int GetCurrentFrame();

  /** @brief Changes the next frame to be rendered to the next frame in the 4D sequence
   *  (modulo the number of frames, so if this is called on the last frame, the next frame is the first frame)
   */
  void AdvanceFrame();

  /** @brief Changes the total number of frames being rendered (the number of 3D frames in the 4D sequence)
   *  @param n The requested number of frames in the 4D sequence
   *  @pre n is a non-negative integer less than 100
   */
  void SetNumberOfFrames(int n);

  /** @brief Fetches the total number of frames being rendered (the number of 3D frames in the 4D sequence) */
  int GetNumberOfFrames();

  /** @brief Gets a 2D image data consisting of the output of the most current render
   *  @pre The volume mapper is currently rendering to vtkImageData (using the UseImageDataRenderering method), else this method returns NULL
   */
  vtkImageData* GetOutput();

  // Description:
  // Specify Keyhole planes to be applied when the data is mapped
  // (at most 6 Keyhole planes can be specified).
  void AddKeyholePlane(vtkPlane* plane);
  void RemoveKeyholePlane(vtkPlane* plane);
  void RemoveAllKeyholePlanes();

  // Description:
  // Get/Set the vtkPlaneCollection which specifies the
  // Keyhole planes.
  virtual void SetKeyholePlanes(vtkPlaneCollection*);
  vtkGetObjectMacro(KeyholePlanes, vtkPlaneCollection);

  // Description:
  // An alternative way to set Keyhole planes: use up to six planes found
  // in the supplied instance of the implicit function vtkPlanes.
  void SetKeyholePlanes(vtkPlanes* planes);

  /** @brief Sets the displaying type to display to screen using CUDA-OpenGL interoperability, which is fast, but not always supported
   *  @pre The rendering environment is amenable to this form of rendering (aka, no multiple OpenGL contexts with isosurface object inclusion)
   */
  void UseCUDAOpenGLInteroperability();

  /** @brief Sets the displaying type to display to screen using VTK's ray cast helper classes which is more generally supported */
  void UseFullVTKCompatibility();

  /** @brief Sets the displaying type to not display to screen, but to save the image in a vtkImageData object which can be fetched using GetOutput()
   *  @todo Support this option
   */
  void UseImageDataRendering();

  void SetImageFlipped(bool b);
  bool GetImageFlipped();

  void SetTint(double RGBA[4]);
  void GetTint(double RGBA[4]);
  void SetTint(float RGBA[4]);
  void GetTint(float RGBA[4]);
  void SetTint(unsigned char RGBA[4]);
  void GetTint(unsigned char RGBA[4]);

protected:
  /// Using the mapper's volume and renderer objects, check for updates and reconstruct the appropriate matrices based on them,
  ///   sending them off to the renderer information handler afterwards
  void ComputeMatrices();

  virtual void Reinitialize(bool withData = false);
  virtual void Deinitialize(bool withData = false);

protected:
  vtkCudaVolumeMapper();
  virtual ~vtkCudaVolumeMapper();

protected:
  vtkSmartPointer<vtkCudaVolumeInformationHandler>      VolumeInfoHandler;    // The handler for any volume/transfer function information
  vtkSmartPointer<vtkCudaRendererInformationHandler>    RendererInfoHandler;  // The handler for any renderer/camera/geometry/clipping information
  vtkSmartPointer<vtkCudaOutputImageInformationHandler> OutputInfoHandler;    // The handler for any output image housing/display information

  vtkPlaneCollection*                   KeyholePlanes;

  //modified time variables used to minimize setup
  vtkMTimeType                          RendererModifiedTime;  // The last time the renderer object was modified
  vtkMTimeType                          VolumeModifiedTime;    // The last time the volume object was modified
  int                                   CurrentFrame;               // The current frame being rendered
  int                                   FrameCount;                 // The total number of frames housed by the mapper

  vtkSmartPointer<vtkMatrix4x4>         ViewToVoxelsMatrix;         // Matrix used as temporary storage for the view to voxels transformation
  vtkSmartPointer<vtkMatrix4x4>         WorldToVoxelsMatrix;        // Matrix used as temporary storage for the voxels to view transformation

  vtkSmartPointer<vtkTransform>         PerspectiveTransform;       // the perspective transform used by the current camera
  vtkSmartPointer<vtkTransform>         VoxelsTransform;            // the user defined volume transform used to modify position, orientation, etc...
  vtkSmartPointer<vtkTransform>         VoxelsToViewTransform;      // the VoxelsToView transformation used to speed the process of switching/recalculating matrices
  vtkSmartPointer<vtkTransform>         NextVoxelsToViewTransform;  // the next VoxelsToView transformation used to speed the process of switching/recalculating matrices

  bool                                  CanRender;                  // Boolean to describe whether it is safe to render
  bool                                  ImageFlipped;               // Boolean to describe whether the output image is flipped

  std::map<int, vtkImageData*>          InputImages;
};

#endif
