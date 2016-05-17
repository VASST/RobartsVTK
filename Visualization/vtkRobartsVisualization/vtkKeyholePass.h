/*==========================================================================

  Copyright (c) 2016 Uditha L. Jayarathne, ujayarat@robarts.ca

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

#include "vtkRobartsVisualizationModule.h"

#include "vtkRenderingOpenGL2Module.h" // For export macro
#include "vtkImageProcessingPass.h"
#include "vtkImageData.h"
#include "vtkRenderer.h"
#include "vtkTexture.h"
#include "vtkPixelBufferObject.h"
#include "vtkOpenGLHelper.h"
#include "vtkTexture.h"
#include "vtkActorCollection.h"

class vtkOpenGLRenderWindow;
class vtkShaderProgram2;
class vtkShader2;
class vtkFrameBufferObject;
class vtkTextureObject;

class VTKROBARTSVISUALIZATION_EXPORT vtkKeyholePass : public vtkImageProcessingPass
{
public:

  enum class vtkKeyholePass_Texture_Index {BACKGROUND,
      BACKGROUND_EDGEMAP,
      MASK
                                          };

  static vtkKeyholePass *New();
  vtkTypeMacro(vtkKeyholePass, vtkImageProcessingPass);
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
  void UseMaskImage( bool t ){
	  this->mask_img_available = t;
  }

  // Description:
  // Set mask parameters
  void SetKeyholeParameters(int x, int y , int r, double g)
  {
    this->x0 = x;
    this->y0 = y;
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

protected:
  // Description:
  // Default constructor. DelegatePass is set to NULL.
  vtkKeyholePass();

  // Description:
  // Destructor.
  virtual ~vtkKeyholePass();

  // Description:
  // Graphics resources.
  vtkFrameBufferObject *FrameBufferObject;
  vtkTextureObject *Pass1; // render target for the volume
  vtkTextureObject *Pass2; // render target for the horizontal pass
  vtkTexture *foregroundTex, *maskTex;
  vtkPixelBufferObject *foreground_pbo, *mask_pbo;
  vtkTextureObject *foreground_to, *mask_to, *foreground_grad_to, *GX, *GY;
  vtkOpenGLHelper *KeyholeProgram, *gradientProgram1, *gradientProgram2; // keyhole shader

  std::string frag_shader_src, ver_shader_src;

  bool Supported;
  bool SupportProbed;

private:
  vtkKeyholePass(const vtkKeyholePass&);  // Not implemented.
  void operator=(const vtkKeyholePass&);  // Not implemented.
  void LoadShaders(std::string, std::string); // Load Shader programs from file.
  void GetForegroudGradient(vtkRenderer *);// perform sobel pass on foreground texture and save the results to foreground_grad_to

  int x0, y0, radius;
  int components;
  unsigned int dimensions[2];
  float gamma;
  bool allow_hard_edges;
  bool mask_img_available;

};

#endif // vtkKeyholePass_h