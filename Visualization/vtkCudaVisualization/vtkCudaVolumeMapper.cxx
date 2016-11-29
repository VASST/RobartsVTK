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

/** @file vtkCudaVolumeMapper.cxx
 *
 *  @brief Header file defining a volume mapper (ray caster) using CUDA kernels for parallel ray calculation
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 29, 2011
 *
 */

// Local includes
#include "CUDA_containerOutputImageInformation.h"
#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include "vtkCudaOutputImageInformationHandler.h"
#include "vtkCudaRendererInformationHandler.h"
#include "vtkCudaVolumeInformationHandler.h"
#include "vtkCudaVolumeMapper.h"

// VTK includes
#include <vtkCamera.h>
#include <vtkImageData.h>
#include <vtkMatrix4x4.h>
#include <vtkObjectFactory.h>
#include <vtkPlane.h>
#include <vtkPlaneCollection.h>
#include <vtkPlanes.h>
#include <vtkRenderer.h>
#include <vtkTransform.h>
#include <vtkVersion.h>
#include <vtkVolume.h>

// STL includes
#include <algorithm>

namespace
{
  const float RANDOM_RAY_OFFSETS[256] =
  {
    0.70554, 0.53342, 0.57951, 0.28956, 0.30194, 0.77474, 0.01401, 0.76072,
    0.81449, 0.70903, 0.04535, 0.41403, 0.86261, 0.79048, 0.37353, 0.96195,
    0.87144, 0.05623, 0.94955, 0.36401, 0.52486, 0.76711, 0.0535, 0.59245,
    0.4687, 0.29816, 0.62269, 0.64782, 0.26379, 0.27934, 0.8298, 0.8246,
    0.58916, 0.98609, 0.91096, 0.22686, 0.69511, 0.98, 0.24393, 0.53387,
    0.10636, 0.99941, 0.67617, 0.0157, 0.57518, 0.10005, 0.10302, 0.79888,
    0.28448, 0.04564, 0.29577, 0.38201, 0.30097, 0.94857, 0.97982, 0.40137,
    0.27827, 0.16044, 0.16282, 0.64658, 0.41007, 0.41276, 0.71273, 0.3262,
    0.63317, 0.20756, 0.18601, 0.58335, 0.08071, 0.45797, 0.90572, 0.26136,
    0.78521, 0.3789, 0.28966, 0.91937, 0.63174, 0.62764, 0.42845, 0.09797,
    0.56104, 0.69448, 0.91371, 0.83481, 0.02262, 0.54336, 0.91616, 0.43026,
    0.67794, 0.50245, 0.51373, 0.46298, 0.35347, 0.40483, 0.26973, 0.05559,
    0.24384, 0.97907, 0.06091, 0.39029, 0.36499, 0.48989, 0.15566, 0.47445,
    0.25726, 0.62875, 0.54207, 0.1563, 0.93854, 0.65449, 0.50608, 0.39047,
    0.10737, 0.78399, 0.45964, 0.75368, 0.59609, 0.83273, 0.01875, 0.21036,
    0.07395, 0.10545, 0.33169, 0.12824, 0.00024, 0.53679, 0.65705, 0.54401,
    0.82741, 0.08189, 0.19192, 0.67891, 0.4542, 0.35702, 0.14998, 0.70439,
    0.92878, 0.53021, 0.08964, 0.75772, 0.40184, 0.46187, 0.49216, 0.20762,
    0.32973, 0.09542, 0.58979, 0.16987, 0.92761, 0.09792, 0.44386, 0.27294,
    0.87254, 0.75068, 0.27294, 0.67364, 0.25662, 0.08989, 0.03095, 0.32271,
    0.79012, 0.29725, 0.23528, 0.48047, 0.2546, 0.3406, 0.04493, 0.48242,
    0.20601, 0.86453, 0.58862, 0.7549, 0.92788, 0.33101, 0.54294, 0.08069,
    0.63437, 0.41003, 0.96042, 0.11462, 0.92344, 0.6202, 0.34772, 0.14924,
    0.47997, 0.2194, 0.99373, 0.13042, 0.02888, 0.34539, 0.54766, 0.92295,
    0.53824, 0.40642, 0.84724, 0.82622, 0.67242, 0.72189, 0.99677, 0.3398,
    0.49521, 0.41296, 0.69528, 0.17908, 0.42291, 0.54317, 0.81466, 0.54091,
    0.42753, 0.50906, 0.22778, 0.61918, 0.48983, 0.68081, 0.8866, 0.37051,
    0.30249, 0.29286, 0.15031, 0.52982, 0.22326, 0.58452, 0.36345, 0.87597,
    0.47801, 0.19063, 0.68406, 0.74741, 0.61393, 0.78213, 0.16174, 0.80777,
    0.20261, 0.95676, 0.06585, 0.06152, 0.79319, 0.3796, 0.46358, 0.11954,
    0.11547, 0.17377, 0.04811, 0.71481, 0.53302, 0.561, 0.21673, 0.468,
    0.74635, 0.75231, 0.39893, 0.90309, 0.746, 0.08855, 0.63457, 0.71302
  };
}

//----------------------------------------------------------------------------

vtkCxxSetObjectMacro(vtkCudaVolumeMapper, KeyholePlanes, vtkPlaneCollection);

//----------------------------------------------------------------------------
vtkCudaVolumeMapper::vtkCudaVolumeMapper()
  : VolumeInfoHandler(vtkSmartPointer<vtkCudaVolumeInformationHandler>::New())
  , RendererInfoHandler(vtkSmartPointer<vtkCudaRendererInformationHandler>::New())
  , OutputInfoHandler(vtkSmartPointer<vtkCudaOutputImageInformationHandler>::New())
  , KeyholePlanes(nullptr)
  , RendererModifiedTimestamp(0)
  , VolumeModifiedTimestamp(0)
  , CurrentFrame(0)
  , FrameCount(1)
  , ViewToVoxelsMatrix(vtkSmartPointer<vtkMatrix4x4>::New())
  , WorldToVoxelsMatrix(vtkSmartPointer<vtkMatrix4x4>::New())
  , PerspectiveTransform(vtkSmartPointer<vtkTransform>::New())
  , VoxelsTransform(vtkSmartPointer<vtkTransform>::New())
  , VoxelsToViewTransform(vtkSmartPointer<vtkTransform>::New())
  , NextVoxelsToViewTransform(vtkSmartPointer<vtkTransform>::New())
  , CanRender(false)
  , ImageFlipped(false)
{
  this->Reinitialize();
}

//----------------------------------------------------------------------------
vtkCudaVolumeMapper::~vtkCudaVolumeMapper()
{
  this->Deinitialize();
  if (this->KeyholePlanes != nullptr)
  {
    this->KeyholePlanes->Delete();
    this->KeyholePlanes = nullptr;
  }
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::Deinitialize(bool withData /*= false*/)
{
  CUDA_vtkCudaVolumeMapper_renderAlgo_unloadrandomRayOffsets(this->GetStream());
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::Reinitialize(bool withData /*= false*/)
{
  this->VolumeInfoHandler->ReplicateObject(this, withData);
  this->RendererInfoHandler->ReplicateObject(this, withData);
  this->OutputInfoHandler->ReplicateObject(this, withData);

  //initialize the random ray denoising buffer
  CUDA_vtkCudaVolumeMapper_renderAlgo_loadrandomRayOffsets(RANDOM_RAY_OFFSETS, this->GetStream());

  //re-copy the image data if any
  if (withData)
  {
    for (std::map<int, vtkImageData*>::iterator it = this->InputImages.begin(); it != this->InputImages.end(); it++)
    {
      this->SetInputInternal(it->second, it->first);
    }
  }
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::SetNumberOfFrames(int n)
{
  if (n > 0 && n <= VTKCUDAVOLUMEMAPPER_UPPER_BOUND)
  {
    this->FrameCount = n;
  }
}

//----------------------------------------------------------------------------
int vtkCudaVolumeMapper::GetNumberOfFrames()
{
  return this->FrameCount;
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::SetInputData(vtkImageData* input)
{
  //set information at this level
  this->vtkVolumeMapper::SetInputData(input);
  this->VolumeInfoHandler->SetInputData(input, 0);
  this->InputImages.insert(std::pair<int, vtkImageData*>(0, input));

  //pass down to subclass
  this->SetInputInternal(input, 0);
  if (this->CurrentFrame == 0) { this->ChangeFrame(0); }
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::SetInputData(vtkImageData* input, int index)
{
  //check for consistency
  if (index < 0 || !(index < this->FrameCount)) { return; }

  //set information at this level
  this->vtkVolumeMapper::SetInputData(input);
  this->VolumeInfoHandler->SetInputData(input, index);
  this->InputImages.insert(std::pair<int, vtkImageData*>(index, input));

  //pass down to subclass
  this->SetInputInternal(input, index);
  if (this->CurrentFrame == 0) { this->ChangeFrame(0); }
}

//----------------------------------------------------------------------------
vtkImageData* vtkCudaVolumeMapper::GetInput()
{
  return GetInput(0);
}

//----------------------------------------------------------------------------
vtkImageData* vtkCudaVolumeMapper::GetInput(int frame)
{
  if (this->InputImages.find(frame) != this->InputImages.end())
  {
    return this->InputImages[frame];
  }
  return 0;
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::ClearInput()
{
  //clear information at this class level
  this->VolumeInfoHandler->ClearInput();
  this->InputImages.clear();

  //pass down to subclass
  this->ClearInputInternal();
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::SetCelShadingConstants(float darkness, float a, float b)
{
  this->RendererInfoHandler->SetCelShadingConstants(darkness, a, b);
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::SetDistanceShadingConstants(float darkness, float a, float b)
{
  this->RendererInfoHandler->SetDistanceShadingConstants(darkness, a, b);
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::SetRenderOutputScaleFactor(float scaleFactor)
{
  this->OutputInfoHandler->SetRenderOutputScaleFactor(scaleFactor);
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::ChangeFrame(int frame)
{
  if (frame >= 0 && frame < this->FrameCount)
  {
    this->ChangeFrameInternal(frame);
    this->CurrentFrame = frame;
  }
}

//----------------------------------------------------------------------------
int vtkCudaVolumeMapper::GetCurrentFrame()
{
  return this->CurrentFrame;
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::AdvanceFrame()
{
  this->ChangeFrame((this->CurrentFrame + 1) % this->FrameCount);
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::UseCUDAOpenGLInteroperability()
{
  this->OutputInfoHandler->SetRenderType(0);
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::UseFullVTKCompatibility()
{
  this->OutputInfoHandler->SetRenderType(1);
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::UseImageDataRendering()
{
  this->OutputInfoHandler->SetRenderType(2);
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::SetImageFlipped(bool b)
{
  this->OutputInfoHandler->SetImageFlipped(b);
}

//----------------------------------------------------------------------------
bool vtkCudaVolumeMapper::GetImageFlipped()
{
  return this->OutputInfoHandler->GetImageFlipped();
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::Render(vtkRenderer* renderer, vtkVolume* volume)
{
  //prepare the 3 main information handlers
  if (volume != this->VolumeInfoHandler->GetVolume()) { this->VolumeInfoHandler->SetVolume(volume); }
  this->VolumeInfoHandler->Update();
  this->RendererInfoHandler->SetRenderer(renderer);
  this->OutputInfoHandler->SetRenderer(renderer);
  this->ComputeMatrices();
  this->RendererInfoHandler->LoadZBuffer();
  this->RendererInfoHandler->SetClippingPlanes(this->ClippingPlanes);
  this->RendererInfoHandler->SetKeyholePlanes(this->KeyholePlanes);
  this->OutputInfoHandler->Prepare();

  //pass the actual rendering process to the subclass
  if (CanRender)
  {
    vtkErrorMacro("Error propogation in rendering - cause error flag previously set - MARKER 3");
  }
  else
  {
    try
    {
      this->InternalRender(renderer, volume,
                           this->RendererInfoHandler->GetRendererInfo(),
                           this->VolumeInfoHandler->GetVolumeInfo(),
                           this->OutputInfoHandler->GetOutputImageInfo());
    }
    catch (...)
    {
      CanRender = true;
      vtkErrorMacro("Internal rendering error - cause unknown - MARKER 2");
    }
  }

  //display the rendered results
  this->OutputInfoHandler->Display(volume, renderer);

  return;
}

//----------------------------------------------------------------------------
vtkImageData* vtkCudaVolumeMapper::GetOutput()
{
  return this->OutputInfoHandler->GetCurrentImageData();
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::ComputeMatrices()
{
  // Get the renderer and the volume from the information handlers
  vtkRenderer* ren = this->RendererInfoHandler->GetRenderer();
  vtkVolume* vol = this->VolumeInfoHandler->GetVolume();
  bool flag = false;

  if (ren->GetMTime() > this->RendererModifiedTimestamp)
  {
    this->RendererModifiedTimestamp = ren->GetMTime();
    flag = true;

    // Get the camera from the renderer
    vtkCamera* cam = ren->GetActiveCamera();

    // Get the aspect ratio from the renderer. This is needed for the
    // computation of the perspective matrix
    ren->ComputeAspect();
    double* aspect = ren->GetAspect();

    // Keep track of the projection matrix - we'll need it in a couple of places
    // Get the projection matrix. The method is called perspective, but
    // the matrix is valid for perspective and parallel viewing transforms.
    // Don't replace this with the GetCompositePerspectiveTransformMatrix
    // because that turns off stereo rendering!!!
    this->PerspectiveTransform->Identity();
    this->PerspectiveTransform->Concatenate(cam->GetProjectionTransformMatrix(aspect[0] / aspect[1], 0.0, 1.0));
    this->PerspectiveTransform->Concatenate(cam->GetViewTransformMatrix());
  }

  if (vol->GetMTime() > this->VolumeModifiedTimestamp)
  {
    this->VolumeModifiedTimestamp = vol->GetMTime();
    flag = true;

    //get the input origin, spacing and extents
    vtkImageData* img = this->VolumeInfoHandler->GetInputData();
    double inputOrigin[3];
    double inputSpacing[3];
    int inputExtent[6];
    img->GetOrigin(inputOrigin);
    img->GetSpacing(inputSpacing);
    img->GetExtent(inputExtent);

    // Compute the origin of the extent the volume origin is at voxel (0,0,0)
    // but we want to consider (0,0,0) in voxels to be at
    // (inputExtent[0], inputExtent[2], inputExtent[4]).
    double extentOrigin[3];
    extentOrigin[0] = inputOrigin[0] + inputExtent[0] * inputSpacing[0];
    extentOrigin[1] = inputOrigin[1] + inputExtent[2] * inputSpacing[1];
    extentOrigin[2] = inputOrigin[2] + inputExtent[4] * inputSpacing[2];

    // Create a transform that will account for the scaling and translation of
    // the scalar data. The is the volume to voxels matrix.
    this->VoxelsTransform->Identity();
    this->VoxelsTransform->Translate(extentOrigin[0], extentOrigin[1], extentOrigin[2]);
    this->VoxelsTransform->Scale(inputSpacing[0], inputSpacing[1], inputSpacing[2]);

    // Get the volume matrix. This is a volume to world matrix right now.
    // We'll need to invert it, translate by the origin and scale by the
    // spacing to change it to a world to voxels matrix.
    if (vol->GetUserMatrix() != NULL)
    {
      this->VoxelsToViewTransform->SetMatrix(vol->GetUserMatrix());
    }
    else
    {
      this->VoxelsToViewTransform->Identity();
    }

    // Now concatenate the volume's matrix with this scalar data matrix (sending the result off as the voxels to world matrix)
    this->VoxelsToViewTransform->PreMultiply();
    this->VoxelsToViewTransform->Concatenate(this->VoxelsTransform->GetMatrix());
    this->RendererInfoHandler->SetVoxelsToWorldMatrix(VoxelsToViewTransform->GetMatrix());

    // Invert the transform (sending the result off as the world to voxels matrix)
    this->WorldToVoxelsMatrix->DeepCopy(this->VoxelsToViewTransform->GetMatrix());
    this->WorldToVoxelsMatrix->Invert();
    this->RendererInfoHandler->SetWorldToVoxelsMatrix(this->WorldToVoxelsMatrix);

  }

  if (flag)
  {
    this->Modified();

    // Compute the voxels to view transform by concatenating the
    // voxels to world matrix with the projection matrix (world to view)
    this->NextVoxelsToViewTransform->DeepCopy(this->VoxelsToViewTransform);
    this->NextVoxelsToViewTransform->PostMultiply();
    this->NextVoxelsToViewTransform->Concatenate(this->PerspectiveTransform->GetMatrix());

    this->ViewToVoxelsMatrix->DeepCopy(this->NextVoxelsToViewTransform->GetMatrix());
    this->ViewToVoxelsMatrix->Invert();

    //load into the renderer information via the handler
    this->RendererInfoHandler->SetViewToVoxelsMatrix(this->ViewToVoxelsMatrix);
  }
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::AddKeyholePlane(vtkPlane* plane)
{
  if (this->KeyholePlanes == NULL)
  {
    this->KeyholePlanes = vtkPlaneCollection::New();
    this->KeyholePlanes->Register(this);
    this->KeyholePlanes->Delete();
  }

  this->KeyholePlanes->AddItem(plane);
  this->Modified();
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::RemoveKeyholePlane(vtkPlane* plane)
{
  if (this->KeyholePlanes == NULL) { vtkErrorMacro("Cannot remove Keyhole plane: mapper has none"); }
  this->KeyholePlanes->RemoveItem(plane);
  this->Modified();
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::RemoveAllKeyholePlanes()
{
  if (this->KeyholePlanes) { this->KeyholePlanes->RemoveAllItems(); }
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::SetKeyholePlanes(vtkPlanes* planes)
{
  vtkPlane* plane;
  if (!planes) { return; }

  int numPlanes = planes->GetNumberOfPlanes();

  this->RemoveAllKeyholePlanes();
  for (int i = 0; i < numPlanes && i < 6; i++)
  {
    plane = vtkPlane::New();
    planes->GetPlane(i, plane);
    this->AddKeyholePlane(plane);
    plane->Delete();
  }
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::SetTint(double RGBA[4])
{
  unsigned char RGBAuc[4];
  RGBAuc[0] = std::min(255.0, std::max(0.0, 255.0 * RGBA[0]));
  RGBAuc[1] = std::min(255.0, std::max(0.0, 255.0 * RGBA[1]));
  RGBAuc[2] = std::min(255.0, std::max(0.0, 255.0 * RGBA[2]));
  RGBAuc[3] = std::min(255.0, std::max(0.0, 255.0 * RGBA[3]));
  this->OutputInfoHandler->SetTint(RGBAuc);
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::GetTint(double RGBA[4])
{
  unsigned char RGBAuc[4];
  this->OutputInfoHandler->GetTint(RGBAuc);
  RGBA[0] = (double) RGBAuc[0] / 255.0;
  RGBA[1] = (double) RGBAuc[1] / 255.0;
  RGBA[2] = (double) RGBAuc[2] / 255.0;
  RGBA[3] = (double) RGBAuc[3] / 255.0;
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::SetTint(float RGBA[4])
{
  unsigned char RGBAuc[4];
  RGBAuc[0] = std::min(255.0f, std::max(0.0f, 255.0f * RGBA[0]));
  RGBAuc[1] = std::min(255.0f, std::max(0.0f, 255.0f * RGBA[1]));
  RGBAuc[2] = std::min(255.0f, std::max(0.0f, 255.0f * RGBA[2]));
  RGBAuc[3] = std::min(255.0f, std::max(0.0f, 255.0f * RGBA[3]));
  this->OutputInfoHandler->SetTint(RGBAuc);
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::GetTint(float RGBA[4])
{
  unsigned char RGBAuc[4];
  this->OutputInfoHandler->GetTint(RGBAuc);
  RGBA[0] = (float) RGBAuc[0] / 255.0f;
  RGBA[1] = (float) RGBAuc[1] / 255.0f;
  RGBA[2] = (float) RGBAuc[2] / 255.0f;
  RGBA[3] = (float) RGBAuc[3] / 255.0f;
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::SetTint(unsigned char RGBA[4])
{
  this->OutputInfoHandler->SetTint(RGBA);
}

//----------------------------------------------------------------------------
void vtkCudaVolumeMapper::GetTint(unsigned char RGBA[4])
{
  this->OutputInfoHandler->GetTint(RGBA);
}