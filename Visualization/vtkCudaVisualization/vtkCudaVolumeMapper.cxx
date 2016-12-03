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
    0.70554f, 0.53342f, 0.57951f, 0.28956f, 0.30194f, 0.77474f, 0.01401f, 0.76072f,
    0.81449f, 0.70903f, 0.04535f, 0.41403f, 0.86261f, 0.79048f, 0.37353f, 0.96195f,
    0.87144f, 0.05623f, 0.94955f, 0.36401f, 0.52486f, 0.76711f, 0.0535f, 0.59245f,
    0.4687f, 0.29816f, 0.62269f, 0.64782f, 0.26379f, 0.27934f, 0.8298f, 0.8246f,
    0.58916f, 0.98609f, 0.91096f, 0.22686f, 0.69511f, 0.98f, 0.24393f, 0.53387f,
    0.10636f, 0.99941f, 0.67617f, 0.0157f, 0.57518f, 0.10005f, 0.10302f, 0.79888f,
    0.28448f, 0.04564f, 0.29577f, 0.38201f, 0.30097f, 0.94857f, 0.97982f, 0.40137f,
    0.27827f, 0.16044f, 0.16282f, 0.64658f, 0.41007f, 0.41276f, 0.71273f, 0.3262f,
    0.63317f, 0.20756f, 0.18601f, 0.58335f, 0.08071f, 0.45797f, 0.90572f, 0.26136f,
    0.78521f, 0.3789f, 0.28966f, 0.91937f, 0.63174f, 0.62764f, 0.42845f, 0.09797f,
    0.56104f, 0.69448f, 0.91371f, 0.83481f, 0.02262f, 0.54336f, 0.91616f, 0.43026f,
    0.67794f, 0.50245f, 0.51373f, 0.46298f, 0.35347f, 0.40483f, 0.26973f, 0.05559f,
    0.24384f, 0.97907f, 0.06091f, 0.39029f, 0.36499f, 0.48989f, 0.15566f, 0.47445f,
    0.25726f, 0.62875f, 0.54207f, 0.1563f, 0.93854f, 0.65449f, 0.50608f, 0.39047f,
    0.10737f, 0.78399f, 0.45964f, 0.75368f, 0.59609f, 0.83273f, 0.01875f, 0.21036f,
    0.07395f, 0.10545f, 0.33169f, 0.12824f, 0.00024f, 0.53679f, 0.65705f, 0.54401f,
    0.82741f, 0.08189f, 0.19192f, 0.67891f, 0.4542f, 0.35702f, 0.14998f, 0.70439f,
    0.92878f, 0.53021f, 0.08964f, 0.75772f, 0.40184f, 0.46187f, 0.49216f, 0.20762f,
    0.32973f, 0.09542f, 0.58979f, 0.16987f, 0.92761f, 0.09792f, 0.44386f, 0.27294f,
    0.87254f, 0.75068f, 0.27294f, 0.67364f, 0.25662f, 0.08989f, 0.03095f, 0.32271f,
    0.79012f, 0.29725f, 0.23528f, 0.48047f, 0.2546f, 0.3406f, 0.04493f, 0.48242f,
    0.20601f, 0.86453f, 0.58862f, 0.7549f, 0.92788f, 0.33101f, 0.54294f, 0.08069f,
    0.63437f, 0.41003f, 0.96042f, 0.11462f, 0.92344f, 0.6202f, 0.34772f, 0.14924f,
    0.47997f, 0.2194f, 0.99373f, 0.13042f, 0.02888f, 0.34539f, 0.54766f, 0.92295f,
    0.53824f, 0.40642f, 0.84724f, 0.82622f, 0.67242f, 0.72189f, 0.99677f, 0.3398f,
    0.49521f, 0.41296f, 0.69528f, 0.17908f, 0.42291f, 0.54317f, 0.81466f, 0.54091f,
    0.42753f, 0.50906f, 0.22778f, 0.61918f, 0.48983f, 0.68081f, 0.8866f, 0.37051f,
    0.30249f, 0.29286f, 0.15031f, 0.52982f, 0.22326f, 0.58452f, 0.36345f, 0.87597f,
    0.47801f, 0.19063f, 0.68406f, 0.74741f, 0.61393f, 0.78213f, 0.16174f, 0.80777f,
    0.20261f, 0.95676f, 0.06585f, 0.06152f, 0.79319f, 0.3796f, 0.46358f, 0.11954f,
    0.11547f, 0.17377f, 0.04811f, 0.71481f, 0.53302f, 0.561f, 0.21673f, 0.468f,
    0.74635f, 0.75231f, 0.39893f, 0.90309f, 0.746f, 0.08855f, 0.63457f, 0.71302f
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
  , RendererModifiedTime(0)
  , VolumeModifiedTime(0)
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

  if (ren->GetMTime() > this->RendererModifiedTime)
  {
    this->RendererModifiedTime = ren->GetMTime();
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

  if (vol->GetMTime() > this->VolumeModifiedTime)
  {
    this->VolumeModifiedTime = vol->GetMTime();
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