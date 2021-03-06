/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkMultiViewportImageProcessingPass.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   vtkMultiViewportImageProcessingPass
 * @brief   Convenient class for post-processing passes.
 * render pass.
 *
 * Abstract class with some convenient methods frequently used in subclasses.
 * This class was a modification to the original vtkImageProcessingPass class to support
 * rendering in multiple view-port.
 *
 *
 * @sa
 * vtkRenderPass vtkGaussianBlurPass vtkSobelGradientMagnitudePass
*/

#ifndef vtkMultiViewportImageProcessingPass_h
#define vtkMultiViewportImageProcessingPass_h

#include "vtkRobartsVisualizationExport.h" // For export macro

// VTK includes
#include <vtkRenderPass.h>
#include <vtkVersionMacros.h>

class vtkOpenGLRenderWindow;
class vtkDepthPeelingPassLayerList; // Pimpl
#if VTK_MAJOR_VERSION >= 8
  class vtkOpenGLFramebufferObject;
  typedef vtkOpenGLFramebufferObject RobartsVTKFrameBufferObject;
#else
  class vtkFrameBufferObject;
  typedef vtkFrameBufferObject RobartsVTKFrameBufferObject;
#endif
class vtkTextureObject;

class vtkRobartsVisualizationExport vtkMultiViewportImageProcessingPass : public vtkRenderPass
{
public:
  vtkTypeMacro(vtkMultiViewportImageProcessingPass, vtkRenderPass);
  void PrintSelf(ostream& os, vtkIndent indent);

  /**
   * Release graphics resources and ask components to release their own
   * resources.
   * \pre w_exists: w!=0
   */
  void ReleaseGraphicsResources(vtkWindow* w);

  //@{
  /**
   * Delegate for rendering the image to be processed.
   * If it is NULL, nothing will be rendered and a warning will be emitted.
   * It is usually set to a vtkCameraPass or to a post-processing pass.
   * Initial value is a NULL pointer.
   */
  vtkGetObjectMacro(DelegatePass, vtkRenderPass);
  virtual void SetDelegatePass(vtkRenderPass* delegatePass);
  //@}

protected:
  /**
   * Default constructor. DelegatePass is set to NULL.
   */
  vtkMultiViewportImageProcessingPass();

  /**
   * Destructor.
   */
  virtual ~vtkMultiViewportImageProcessingPass();

  /**
   * Render delegate with a image of different dimensions than the
   * original one.
   * \pre s_exists: s!=0
   * \pre fbo_exists: fbo!=0
   * \pre fbo_has_context: fbo->GetContext()!=0
   * \pre target_exists: target!=0
   * \pre target_has_context: target->GetContext()!=0
   */
  void RenderDelegate(const vtkRenderState* s,
                      int width,
                      int height,
                      int newWidth,
                      int newHeight,
                      RobartsVTKFrameBufferObject* fbo,
                      vtkTextureObject* target);

  vtkRenderPass* DelegatePass;

private:
  vtkMultiViewportImageProcessingPass(const vtkMultiViewportImageProcessingPass&) VTK_DELETE_FUNCTION;
  void operator=(const vtkMultiViewportImageProcessingPass&) VTK_DELETE_FUNCTION;
};

#endif
