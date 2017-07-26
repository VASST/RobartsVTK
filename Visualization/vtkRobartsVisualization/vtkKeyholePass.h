/*==========================================================================

  Copyright (c) 2017 Uditha L. Jayarathne, ujayarat@robarts.ca

  Use, modification and redistribution of the software, in source or
  binary forms, are permitted provided that the following terms and
  conditions are met:

  1) Redistribution of the source code, in verbatim or modified
  form, must retain the above copyright notice, this license,
  the following disclaimer, and any notices that refer to this
  license and/or the following disclaimer.

  2) Redistribution in binary form must include the above copyright
  notice, a copy of this license and the following disclaimer
  in the documentation or with other materials provided with the
  distribution.

  3) Modified copies of the source code must be clearly marked as such,
  and must not be misrepresented as verbatim copies of the source code.

  THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE SOFTWARE "AS IS"
  WITHOUT EXPRESSED OR IMPLIED WARRANTY INCLUDING, BUT NOT LIMITED TO,
  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE.  IN NO EVENT SHALL ANY COPYRIGHT HOLDER OR OTHER PARTY WHO MAY
  MODIFY AND/OR REDISTRIBUTE THE SOFTWARE UNDER THE TERMS OF THIS LICENSE
  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, LOSS OF DATA OR DATA BECOMING INACCURATE
  OR LOSS OF PROFIT OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF
  THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGES.
  =========================================================================*/

#ifndef vtkKeyholePass_h
#define vtkKeyholePass_h

#include "vtkRobartsVisualizationExport.h"

#include <vtkVersionMacros.h>
#include "vtkActorCollection.h"
#include "vtkImageData.h"
#include "vtkMultiViewportImageProcessingPass.h" // Use this instead of vtkImageProcessingPass.h 
//  that comes with VTK if multiple view-ports are used.
#include "vtkOpenGLHelper.h"
#include "vtkPixelBufferObject.h"
#include "vtkRenderer.h"
#include "vtkTexture.h"

#if VTK_MAJOR_VERSION >= 8
  class vtkOpenGLFramebufferObject;
  typedef vtkOpenGLFramebufferObject RobartsVTKFrameBufferObject;
#else
  class vtkFrameBufferObject;
  typedef vtkFrameBufferObject RobartsVTKFrameBufferObject;
#endif
class vtkOpenGLRenderWindow;
class vtkShader2;
class vtkShaderProgram2;
class vtkTextureObject;

// To be able to dump intermediate passes into png images for debugging.
//#define VTK_KEYHOLE_PASS_DEBUG

class vtkRobartsVisualizationExport vtkKeyholePass : public vtkMultiViewportImageProcessingPass
{
public:
  enum RenderingMode
  {
    MODE_NO_KEYHOLE,
    MODE_ALPHA_BLENDING,
    MODE_KEYHOLE
  };
  enum vtkKeyholePass_Texture_Index
  {
    BACKGROUND,
    BACKGROUND_EDGEMAP,
    MASK
  };

  static vtkKeyholePass* New();
  vtkTypeMacro(vtkKeyholePass, vtkMultiViewportImageProcessingPass);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Perform rendering according to a render state \p s.
  // \pre s_exists: s!=0
  virtual void Render(const vtkRenderState* s);

  // Description:
  // Release graphics resources and ask components to release their own
  // resources.
  // \pre w_exists: w!=0
  void ReleaseGraphicsResources(vtkWindow* w);

  // Description:
  // Use an image as the mask
  void UseMaskImage(bool t);

  // Description:
  // Set mask parameters
  void SetLeftKeyholeParameters(int x, int y, int r, double g);
  void SetRightKeyholeParameters(int x, int y, int r, double g);

  // Description:
  // Set whether the keyhole has hard edges/soft(blurred) edges.
  // By default it is set to false.
  void SetHardKeyholeEdges(bool t);

  // Description:
  // Set the background color
  // By default is is set to blue
  void SetBackgroundColor(double r, double g, double b);

  // Description:
  // Set visualization mode: 0 - no keyhole, 1- alpha blending, 2- additive blending, 3 - keyhole
  // By default this is set to no keyhole
  void SetVisualizationMode(RenderingMode _mode);

  // Description:
  // Set alpha value for blending. The default value is 0.5
  void SetAlpha(double _alpha);

  // Description:
  // Set d1 value for the opacity function
  void SetD1(double _d1);

protected:
  // Description:
  // Graphics resources.
  RobartsVTKFrameBufferObject*  FrameBufferObject;
  vtkTextureObject*             Pass1; // render target for the volume
  vtkTextureObject*             Pass2; // render target for the horizontal pass
  vtkPixelBufferObject*         LeftPixelBufferObject;
  vtkPixelBufferObject*         RightPixelBufferObject;
  vtkPixelBufferObject*         MaskPixelBufferObject;
  vtkTextureObject*             LeftTextureObject;
  vtkTextureObject*             RightTextureObject;
  vtkTextureObject*             MaskTextureObject;
  vtkTextureObject*             ForegroundGradientTextureObject;
  vtkTextureObject*             GX;
  vtkTextureObject*             GY;
  vtkOpenGLHelper*              KeyholeProgram;
  vtkOpenGLHelper*              GradientProgram1;
  vtkOpenGLHelper*              KeyholeShader;

  std::string                   FragmentShaderSource;
  std::string                   VertexShaderSource;

  int                           xL;
  int                           xR;
  int                           yL;
  int                           yR;
  RenderingMode                 Mode; // 0 - no keyhole, 1 - alpha blending, 3 - with keyhole
  int                           Radius;
  int                           Components;
  double                        D1;
  unsigned int                  Dimensions[2];
  float                         Gamma;
  bool                          AllowHardEdges;
  bool                          MaskImageAvailable;

  int                           ViewPortWidth;
  int                           ViewPortHeight;
  int                           ViewPortX;
  int                           ViewPortY;

  double                        BackgroundRed;
  double                        BackgroundGreen;
  double                        BackgroundBlue;
  double                        BackgroundAlpha;

  bool                          Supported;
  bool                          SupportProbed;
  bool                          Stereo;

protected:
  // Default constructor. DelegatePass is set to NULL.
  vtkKeyholePass();
  virtual ~vtkKeyholePass();

  void LoadShaders(std::string, std::string); // Load Shader programs from file.
  void GetForegroudGradient(vtkRenderer*); // perform sobel pass on foreground texture and save the results to foreground_grad_to
  void UpdateLeftTextureObject(vtkOpenGLRenderWindow*);  // convenience method to update texture object when new data is available.
  void UpdateRightTextureObject(vtkOpenGLRenderWindow*);  // convenience method to update texture object when new data is available.
  void SetupDrawBuffers(vtkRenderer*); // convenience method to set up appropriate drawbuffers.
  void CopyToFrameBuffer(int, int,
                         int, int,
                         int, int,
                         int, int,
                         vtkTextureObject*,
                         vtkShaderProgram*, vtkOpenGLVertexArrayObject*);  // This is a convenience method to update the framebuffer. vtkTextureObject has a similar method.
  // However, it has a minor bug in setting glViewPort. This method fixes that issue.
  void ProbeSupport(const vtkRenderState*);  // Probe for support
  int ReadTextures(vtkRenderer*);

private:
  vtkKeyholePass(const vtkKeyholePass&);  // Not implemented.
  void operator=(const vtkKeyholePass&);  // Not implemented.
};

#endif // vtkKeyholePass_h