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

#include "vtkActorCollection.h"
#include "vtkImageData.h"
#include "vtkMultiViewportImageProcessingPass.h" // Use this instead of vtkImageProcessingPass.h 
									//  that comes with VTK if multiple view-ports are used.
#include "vtkOpenGLHelper.h"
#include "vtkPixelBufferObject.h"
#include "vtkRenderer.h"
#include "vtkTexture.h"

class vtkFrameBufferObject;
class vtkOpenGLRenderWindow;
class vtkShader2;
class vtkShaderProgram2;
class vtkTextureObject;

// To be able to dump intermediate passes into png images for debugging.
//#define VTK_KEYHOLE_PASS_DEBUG

class vtkRobartsVisualizationExport vtkKeyholePass : public vtkMultiViewportImageProcessingPass
{
public:

  enum vtkKeyholePass_Texture_Index
  {
    BACKGROUND,
    BACKGROUND_EDGEMAP,
    MASK
  };

  static vtkKeyholePass *New();
  vtkTypeMacro(vtkKeyholePass, vtkMultiViewportImageProcessingPass);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Perform rendering according to a render state \p s.
  // \pre s_exists: s!=0
  virtual void Render(const vtkRenderState *s);

  // Description:
  // Release graphics resources and ask components to release their own
  // resources.
  // \pre w_exists: w!=0
  void ReleaseGraphicsResources(vtkWindow *w);

  // Description:
  // Use an image as the mask
  void UseMaskImage( bool t )
  {
    this->mask_img_available = t;
  }

  // Description:
  // Set mask parameters
  void SetLeftKeyholeParameters(int x, int y , int r, double g)
  {
    this->xL = x;
    this->yL = y;
    this->radius = r;
    this->gamma = static_cast<float>(g);
  }

  void SetRightKeyholeParameters(int x, int y, int r, double g)
  {
	  this->xR = x;
	  this->yR = y;
	  this->radius = r;
	  this->gamma = static_cast<float>(g);
  }

  // Description:
  // Set whether the keyhole has hard edges/soft(blurred) edges.
  // By default it is set to false.
  void SetHardKeyholeEdges(bool t)
  {
    this->allow_hard_edges = t;
  }

  // Description:
  // Set the background color
  // By default is is set to blue
  void SetBackgroundColor(double r, double g, double b)
  {
	  this->background_r = r;
	  this->background_g = g;
	  this->background_b = b;
  }

  // Description:
  // Set visualization mode: 0 - no keyhole, 1- alpha blending, 2- additive blending, 3 - keyhole
  // By default this is set to no keyhole
  void SetVisualizationMode(int _mode)
  {
	  this->mode = _mode;
  }

  // Description:
  // Set alpha value for blending. The default value is 0.5
  void SetAlphaValue(double _alpha)
  {
	  this->alpha = _alpha;
  }

protected:
  // Description:
  // Graphics resources.
  vtkFrameBufferObject* FrameBufferObject;
  vtkTextureObject* Pass1; // render target for the volume
  vtkTextureObject* Pass2; // render target for the horizontal pass
  vtkPixelBufferObject *leftPixelBufferObject, *rightPixelBufferObject;
  vtkPixelBufferObject* MaskPixelBufferObject;
  vtkTextureObject *leftTextureObject, *rightTextureObject;
  vtkTextureObject* MaskTextureObject;
  vtkTextureObject* ForegroundGradientTextureObject;
  vtkTextureObject* GX;
  vtkTextureObject* GY;
  vtkOpenGLHelper* KeyholeProgram;
  vtkOpenGLHelper* GradientProgram1;
  vtkOpenGLHelper* KeyholeShader; // keyhole shader

  std::string FragmentShaderSource;
  std::string VertexShaderSource;

  int xL, xR;
  int yL, yR;
  int mode; // 0 - no keyhole, 1 - alpha blending, 3 - with keyhole
  int radius;
  int components;
  unsigned int dimensions[2];
  float gamma;
  bool allow_hard_edges;
  bool mask_img_available;

  int viewPortWidth, viewPortHeight;
  int viewPortX, viewPortY;

  double background_r, background_g, background_b;
  double alpha;

  bool Supported;
  bool SupportProbed;
  bool stereo;

private:
  // Description:
  // Default constructor. DelegatePass is set to NULL.
  vtkKeyholePass();

  // Description:
  // Destructor.
  virtual ~vtkKeyholePass();

  vtkKeyholePass(const vtkKeyholePass&);  // Not implemented.
  void operator=(const vtkKeyholePass&);  // Not implemented.
  void LoadShaders(std::string, std::string); // Load Shader programs from file.
  void GetForegroudGradient(vtkRenderer *);// perform sobel pass on foreground texture and save the results to foreground_grad_to
  void UpdateLeftTextureObject(vtkOpenGLRenderWindow *); // convenience method to update texture object when new data is available.
  void UpdateRightTextureObject(vtkOpenGLRenderWindow *); // convenience method to update texture object when new data is available.
  void SetupDrawBuffers(vtkRenderer *);// convenience method to set up appropriate drawbuffers.
  void CopyToFrameBuffer(int, int,
	  int, int,
	  int, int,
	  int, int,
	  vtkTextureObject *,
	  vtkShaderProgram*, vtkOpenGLVertexArrayObject *); // This is a convenience method to update the framebuffer. vtkTextureObject has a similar method. 
														// However, it has a minor bug in setting glViewPort. This method fixes that issue. 
  void ProbeSupport(const vtkRenderState *); // Probe for support
  void ReadTextures(vtkRenderer *);
};

#endif // vtkKeyholePass_h