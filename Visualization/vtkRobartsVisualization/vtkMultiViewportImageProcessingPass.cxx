/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkMultiViewportImageProcessingPass.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// Local includes
#include "vtkMultiViewportImageProcessingPass.h"

// VTK includes
#include <vtk_glew.h>
#include <vtkCamera.h>
#if VTK_MAJOR_VERSION >= 8
  #include <vtkOpenGLFramebufferObject.h>
  typedef vtkOpenGLFramebufferObject RobartsVTKFrameBufferObject;
#else
  #include <vtkFrameBufferObject.h>
  typedef vtkFrameBufferObject RobartsVTKFrameBufferObject;
#endif
#include <vtkImageExtractComponents.h>
#include <vtkImageImport.h>
#include <vtkMath.h>
#include <vtkObjectFactory.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkPNGWriter.h>
#include <vtkPixelBufferObject.h>
#include <vtkRenderState.h>
#include <vtkRenderer.h>
#include <vtkTextureObject.h>

// STL includes
#include <cassert>

// ----------------------------------------------------------------------------
vtkCxxSetObjectMacro(vtkMultiViewportImageProcessingPass, DelegatePass, vtkRenderPass);

// ----------------------------------------------------------------------------
vtkMultiViewportImageProcessingPass::vtkMultiViewportImageProcessingPass()
{
  this->DelegatePass = 0;
}

// ----------------------------------------------------------------------------
vtkMultiViewportImageProcessingPass::~vtkMultiViewportImageProcessingPass()
{
  if (this->DelegatePass != 0)
  {
    this->DelegatePass->Delete();
  }
}

// ----------------------------------------------------------------------------
void vtkMultiViewportImageProcessingPass::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "DelegatePass:";
  if (this->DelegatePass != 0)
  {
    this->DelegatePass->PrintSelf(os, indent);
  }
  else
  {
    os << "(none)" << endl;
  }
}
// ----------------------------------------------------------------------------
// Description:
// Render delegate with a image of different dimensions than the
// original one.
// \pre s_exists: s!=0
// \pre fbo_exists: fbo!=0
// \pre fbo_has_context: fbo->GetContext()!=0
// \pre target_exists: target!=0
// \pre target_has_context: target->GetContext()!=0
void vtkMultiViewportImageProcessingPass::RenderDelegate(const vtkRenderState* s,
    int width,
    int height,
    int newWidth,
    int newHeight,
    RobartsVTKFrameBufferObject* fbo,
    vtkTextureObject* target)
{
  assert("pre: s_exists" && s != 0);
  assert("pre: fbo_exists" && fbo != 0);
  assert("pre: fbo_has_context" && fbo->GetContext() != 0);
  assert("pre: target_exists" && target != 0);
  assert("pre: target_has_context" && target->GetContext() != 0);

#ifdef VTK_IMAGE_PROCESSING_PASS_DEBUG
  cout << "width=" << width << endl;
  cout << "height=" << height << endl;
  cout << "newWidth=" << newWidth << endl;
  cout << "newHeight=" << newHeight << endl;
#endif

  vtkRenderer* r = s->GetRenderer();
  vtkRenderState s2(r);
  s2.SetPropArrayAndCount(s->GetPropArray(), s->GetPropArrayCount());

  // Adapt camera to new window size
  vtkCamera* savedCamera = r->GetActiveCamera();
  savedCamera->Register(this);
  vtkCamera* newCamera = vtkCamera::New();
  newCamera->DeepCopy(savedCamera);

#ifdef VTK_IMAGE_PROCESSING_PASS_DEBUG
  cout << "old camera params=";
  savedCamera->Print(cout);
  cout << "new camera params=";
  newCamera->Print(cout);
#endif
  r->SetActiveCamera(newCamera);

  if (newCamera->GetParallelProjection())
  {
    newCamera->SetParallelScale(
      newCamera->GetParallelScale()*newHeight / static_cast<double>(height));
  }
  else
  {
    double large;
    double small;
    if (newCamera->GetUseHorizontalViewAngle())
    {
      large = newWidth;
      small = width;
    }
    else
    {
      large = newHeight;
      small = height;

    }
    double angle = vtkMath::RadiansFromDegrees(newCamera->GetViewAngle());

#ifdef VTK_IMAGE_PROCESSING_PASS_DEBUG
    cout << "old angle =" << angle << " rad=" << vtkMath::DegreesFromRadians(angle) << " deg" << endl;
#endif

    angle = 2.0 * atan(tan(angle / 2.0) * large / static_cast<double>(small));

#ifdef VTK_IMAGE_PROCESSING_PASS_DEBUG
    cout << "new angle =" << angle << " rad=" << vtkMath::DegreesFromRadians(angle) << " deg" << endl;
#endif

    newCamera->SetViewAngle(vtkMath::DegreesFromRadians(angle));
  }

  s2.SetFrameBuffer(fbo);

  if (target->GetWidth() != static_cast<unsigned int>(newWidth) ||
      target->GetHeight() != static_cast<unsigned int>(newHeight))
  {
    target->Create2D(newWidth, newHeight, 4, VTK_UNSIGNED_CHAR, false);
  }

#if VTK_MAJOR_VERSION >= 8
  fbo->AddColorAttachment(GL_FRAMEBUFFER, 0, target);
#else
  fbo->SetNumberOfRenderTargets(1);
  fbo->SetColorBuffer(0, target);
#endif

  // because the same FBO can be used in another pass but with several color
  // buffers, force this pass to use 1, to avoid side effects from the
  // render of the previous frame.
#if VTK_MAJOR_VERSION >= 8
  fbo->ActivateDrawBuffer(0);
  fbo->StartNonOrtho(newWidth, newHeight);
#else
  fbo->SetActiveBuffer(0);
  fbo->StartNonOrtho(newWidth, newHeight, false);
#endif

  GLint saved_viewport[4];
  glGetIntegerv(GL_VIEWPORT, saved_viewport);
  GLboolean saved_scissor_test;
  glGetBooleanv(GL_SCISSOR_TEST, &saved_scissor_test);
  GLint saved_scissor_box[4];
  glGetIntegerv(GL_SCISSOR_BOX, saved_scissor_box);

  // 2. Delegate render in FBO
  glEnable(GL_DEPTH_TEST);
  this->DelegatePass->Render(&s2);
  this->NumberOfRenderedProps +=
    this->DelegatePass->GetNumberOfRenderedProps();

  fbo->UnBind();

#ifdef VTK_IMAGE_PROCESSING_PASS_DEBUG
  vtkPixelBufferObject* pbo = target->Download();

  unsigned int dims[2];
  vtkIdType continuousInc[3];

  dims[0] = static_cast<unsigned int>(newWidth);
  dims[1] = static_cast<unsigned int>(newHeight);
  continuousInc[0] = 0;
  continuousInc[1] = 0;

  unsigned char* buffer = new unsigned char[newWidth * newHeight * 4];
  bool status = pbo->Download2D(VTK_UNSIGNED_CHAR, buffer, dims, 4, continuousInc);
  assert("check" && status);
  pbo->Delete();
  //glReadPixels(0, 0, newWidth, newHeight, GL_RGBA, GL_UNSIGNED_BYTE, buffer);

  vtkImageImport* importer = vtkImageImport::New();
  importer->CopyImportVoidPointer(buffer, 4 * width * height * sizeof(unsigned char));
  importer->SetDataScalarTypeToUnsignedChar();
  importer->SetNumberOfScalarComponents(4);
  importer->SetWholeExtent(0, width - 1, 0, height - 1, 0, 0);
  importer->SetDataExtentToWholeExtent();
  delete[] buffer;

  vtkImageExtractComponents* rgbatoRgb = vtkImageExtractComponents::New();
  rgbatoRgb->SetInputConnection(importer->GetOutputPort());
  rgbatoRgb->SetComponents(0, 1, 2);

  vtkPNGWriter* writer = vtkPNGWriter::New();
  writer->SetFileName("mvp_pass.png");
  writer->SetInputConnection(rgbatoRgb->GetOutputPort());
  writer->Write();

  importer->Delete();
  rgbatoRgb->Delete();
  writer->Delete();
#endif

  glViewport(saved_viewport[0], saved_viewport[1], saved_viewport[2],
             saved_viewport[3]);
  glScissor(saved_scissor_box[0], saved_scissor_box[1], saved_scissor_box[2],
            saved_scissor_box[3]);

  newCamera->Delete();
  r->SetActiveCamera(savedCamera);
  savedCamera->UnRegister(this);
}

// ----------------------------------------------------------------------------
// Description:
// Release graphics resources and ask components to release their own
// resources.
// \pre w_exists: w!=0
void vtkMultiViewportImageProcessingPass::ReleaseGraphicsResources(vtkWindow* w)
{
  assert("pre: w_exists" && w != 0);
  if (this->DelegatePass != 0)
  {
    this->DelegatePass->ReleaseGraphicsResources(w);
  }
}
