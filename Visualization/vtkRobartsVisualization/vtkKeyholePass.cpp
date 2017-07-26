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

#include "RobartsVTKConfigure.h"

// Local includes
#include "vtkKeyholePass.h"

// VTK includes
#include <vtkObjectFactory.h>
#include <vtkOpenGLError.h>
#include <vtkOpenGLHelper.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOpenGLShaderCache.h>
#include <vtkOpenGLTexture.h>
#include <vtkOpenGLVertexArrayObject.h>
#include <vtkProperty.h>
#include <vtkRenderState.h>
#include <vtkRenderer.h>
#include <vtkShaderProgram.h>
#include <vtkTextureObject.h>
#include <vtkTextureUnitManager.h>
#if VTK_MAJOR_VERSION >= 8
  #include <vtkOpenGLFramebufferObject.h>
  typedef vtkOpenGLFramebufferObject RobartsVTKFrameBufferObject;
#else
  #include <vtkFrameBufferObject.h>
  typedef vtkFrameBufferObject RobartsVTKFrameBufferObject;
#endif
#include <vtkCamera.h>
#include <vtkImageExtractComponents.h>
#include <vtkImageImport.h>
#include <vtkMath.h>
#include <vtkPNGWriter.h>
#include <vtkPixelBufferObject.h>
#include <vtkPixelBufferObject.h>

#include <cassert>
#include <sstream>

vtkStandardNewMacro(vtkKeyholePass);

//----------------------------------------------------------------------------------------------------
vtkKeyholePass::vtkKeyholePass()
  : FrameBufferObject(NULL),
    Pass1(NULL),
    Pass2(NULL),
    LeftPixelBufferObject(NULL),
    RightPixelBufferObject(NULL),
    MaskPixelBufferObject(NULL),
    LeftTextureObject(NULL),
    RightTextureObject(NULL),
    MaskTextureObject(NULL),
    GX(NULL),
    GY(NULL),
    ForegroundGradientTextureObject(NULL),
    KeyholeProgram(NULL),
    GradientProgram1(NULL),
    KeyholeShader(NULL),
    Supported(false),
    SupportProbed(false),
    AllowHardEdges(false),
    MaskImageAvailable(false),
    BackgroundRed(0),
    BackgroundGreen(0),
    BackgroundBlue(128),
    Mode(MODE_NO_KEYHOLE),
    BackgroundAlpha(0.5),
    D1(0.0)
{
}

//----------------------------------------------------------------------------------------------------
vtkKeyholePass::~vtkKeyholePass()
{
  if (this->FrameBufferObject != NULL)
  {
    vtkErrorMacro( << "FrameBufferObject should have been deleted in ReleaseGraphicsResources().");
  }
  if (this->Pass1 != NULL)
  {
    vtkErrorMacro( << "Pass1 should have been deleted in ReleaseGraphicsResources().");
  }
  if (this->Pass2 != NULL)
  {
    vtkErrorMacro( << "Pass2 should have been deleted in ReleaseGraphicsResources().");
  }
}

//----------------------------------------------------------------------------------------------------
void vtkKeyholePass::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkMultiViewportImageProcessingPass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------------------------------
// Description:
// Perform rendering according to a render state \p s.
// \pre s_exist s!=0
void vtkKeyholePass::Render(const vtkRenderState* s)
{
  assert("pre: s_exists" && s != NULL);

  vtkOpenGLClearErrorMacro();

  this->NumberOfRenderedProps = 0;

  vtkRenderer* r = s->GetRenderer();

  // find out if we should stereo render
  this->Stereo = r->GetRenderWindow()->GetStereoRender() == 1;

  vtkOpenGLRenderWindow* renwin = vtkOpenGLRenderWindow::SafeDownCast(r->GetRenderWindow());
  if (this->DelegatePass == NULL)
  {
    vtkWarningMacro( << "no delegate.");
  }
  else
  {
    if (!this->SupportProbed)
    {
      this->ProbeSupport(s);
    }

    // If not supported.
    if (!this->Supported)
    {

      this->DelegatePass->Render(s);
      this->NumberOfRenderedProps += this->DelegatePass->GetNumberOfRenderedProps();

      return;
    }

    // Read image data into pixelBuffers.
    if (this->ReadTextures(r) == -1)
    {
      return;
    }

    // Create Left & Right texture objects .
    if (this->LeftTextureObject->GetWidth() != static_cast<unsigned int>(this->Dimensions[0]) ||
        this->LeftTextureObject->GetHeight() != static_cast<unsigned int>(this->Dimensions[1]))
    {
      this->LeftTextureObject->Create2D(this->Dimensions[0], this->Dimensions[1], this->Components,
                                        this->LeftPixelBufferObject, false);

      if (this->Stereo && (this->RightTextureObject->GetWidth() != static_cast<unsigned int>(this->Dimensions[0]) ||
                           this->RightTextureObject->GetHeight() != static_cast<unsigned int>(this->Dimensions[1])))
      {
        this->RightTextureObject->Create2D(this->Dimensions[0], this->Dimensions[1], this->Components,
                                           this->RightPixelBufferObject, false);
      }
    }
    else
    {
      UpdateLeftTextureObject(renwin);

      if (this->Stereo)
      {
        UpdateRightTextureObject(renwin);
      }
    }

    // Create mask texture object.
    if (this->MaskImageAvailable)
    {
      if (this->MaskTextureObject->GetHeight() != static_cast<unsigned int>(this->Dimensions[0]) ||
          this->MaskTextureObject->GetWidth() != static_cast<unsigned int>(this->Dimensions[1]))

        this->MaskTextureObject->Create2D(this->Dimensions[0], this->Dimensions[1], this->Components,
                                          this->MaskPixelBufferObject, false);
    }

    // 1. Create a new render state with FBO.
    // Get viewport dimensions

    // Save State
    GLint saved_viewport[4];
    glGetIntegerv(GL_VIEWPORT, saved_viewport);

    r->GetTiledSizeAndOrigin(&this->ViewPortWidth, &this->ViewPortHeight,
                             &this->ViewPortX, &this->ViewPortY);

    // Get ViewPort Size, not the window size
    int width;
    int height;
    width = this->ViewPortWidth;
    height = this->ViewPortHeight;

    // Remove background texture
    r->SetTexturedBackground(false);
    // Now set a black background.
    r->SetBackground(this->BackgroundRed, this->BackgroundGreen, this->BackgroundBlue);

    // First pass
    this->RenderDelegate(s, width, height, width, height, this->FrameBufferObject,
                         this->Pass1);

#ifdef VTK_KEYHOLE_PASS_DEBUG
    // Save the output of the first pass to a file for debugging
    vtkPixelBufferObject* pbo = this->Pass1->Download();

    vtkIdType increments[2];
    increments[0] = 0;
    increments[1] = 0;

    this->Dimensions[0] = width;
    this->Dimensions[1] = height;

    unsigned char* openglRawData = new unsigned char[4 * width * height];
    bool status = pbo->Download2D(VTK_UNSIGNED_CHAR, openglRawData, this->Dimensions, 4, increments);
    assert("check" && status);
    pbo->Delete();

    vtkImageImport* importer = vtkImageImport::New();
    importer->CopyImportVoidPointer(openglRawData, 4 * width * height * sizeof(unsigned char));
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents(4);
    importer->SetWholeExtent(0, width - 1, 0, height - 1, 0, 0);
    importer->SetDataExtentToWholeExtent();
    delete[] openglRawData;

    vtkImageExtractComponents* rgbatoRgb = vtkImageExtractComponents::New();
    rgbatoRgb->SetInputConnection(importer->GetOutputPort());
    rgbatoRgb->SetComponents(0, 1, 2);

    vtkPNGWriter* writer = vtkPNGWriter::New();
    writer->SetFileName("KeyholePass1.png");
    writer->SetInputConnection(rgbatoRgb->GetOutputPort());
    writer->Write();

    importer->Delete();
    rgbatoRgb->Delete();
    writer->Delete();
#endif

    // 2. Sobel pass and save to texture
    GetForegroudGradient(r);

    ///------------------------------------------

    // Same FBO, but new colour attachment (new TO).
    // Pass2 is our final composed scene
    if (this->Pass2 == NULL)
    {
      this->Pass2 = vtkTextureObject::New();
      this->Pass2->SetContext(this->FrameBufferObject->GetContext());
    }

    if (this->Pass2->GetWidth() != static_cast<unsigned int>(width) ||
        this->Pass2->GetHeight() != static_cast<unsigned int>(height))
    {
      this->Pass2->Create2D(static_cast<unsigned int>(width),
                            static_cast<unsigned int>(height), 4,
                            VTK_UNSIGNED_CHAR, false);
    }

    this->FrameBufferObject->Bind();
    this->FrameBufferObject->SetColorBuffer(0, this->Pass2);
    this->FrameBufferObject->Start(width, height, false);

    // Now use the shader to do composition.
    if (this->KeyholeProgram == NULL)
    {
      this->KeyholeProgram = new vtkOpenGLHelper;
      // build the shader source code
      std::string data_dir = std::string(SHADER_DIRECTORY);
      LoadShaders(data_dir + "vtkKeyhole.fs", data_dir + "vtkKeyhole.vs");

      std::string GSSource;
      // compile and bind it if needed
      vtkShaderProgram* newShader =
        renwin->GetShaderCache()->ReadyShaderProgram(
          this->VertexShaderSource.c_str(),
          this->FragmentShaderSource.c_str(),
          GSSource.c_str());

      // if the shader changed reinitialize the VAO
      if (newShader != this->KeyholeProgram->Program)
      {
        this->KeyholeProgram->Program = newShader;
        this->KeyholeProgram->VAO->ShaderProgramChanged(); // reset the VAO as the shader has changed
      }

      this->KeyholeProgram->ShaderSourceTime.Modified();
    }
    else
    {
      renwin->GetShaderCache()->ReadyShaderProgram(this->KeyholeProgram->Program);
    }

    if (this->KeyholeProgram->Program->GetCompiled() != true)
    {
      vtkErrorMacro("Couldn't build the shader program. At this point , it can be an error in a shader or a driver bug.");

      //restore state
      this->FrameBufferObject->UnBind();
      return;
    }

    this->Pass1->Activate();
    int texture0 = this->Pass1->GetTextureUnit();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    this->KeyholeProgram->Program->SetUniformi("_volume", texture0);

    if (this->MaskImageAvailable)
    {
      this->MaskTextureObject->Activate();
      int texture1 = this->MaskTextureObject->GetTextureUnit();
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      this->KeyholeProgram->Program->SetUniformi("_mask", texture1);
    }

    int texture2;
    if (r->GetActiveCamera()->GetLeftEye())
    {
      this->LeftTextureObject->Activate();
      texture2 = this->LeftTextureObject->GetTextureUnit();
    }
    else
    {
      this->RightTextureObject->Activate();
      texture2 = this->RightTextureObject->GetTextureUnit();
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    this->KeyholeProgram->Program->SetUniformi("_foreground", texture2);

    this->ForegroundGradientTextureObject->Activate();
    int texture3 = this->ForegroundGradientTextureObject->GetTextureUnit();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    this->KeyholeProgram->Program->SetUniformi("_foreground_grad", texture3);

    if (r->GetActiveCamera()->GetLeftEye())
    {
      this->KeyholeProgram->Program->SetUniformf("x0", static_cast<float>((this->xL * 1.0) / width));
      this->KeyholeProgram->Program->SetUniformf("y0", static_cast<float>((this->yL * 1.0) / height));
    }
    else
    {
      this->KeyholeProgram->Program->SetUniformf("x0", static_cast<float>((this->xR * 1.0) / width));
      this->KeyholeProgram->Program->SetUniformf("y0", static_cast<float>((this->yR * 1.0) / height));
    }

    this->KeyholeProgram->Program->SetUniformi("mode", this->Mode);
    this->KeyholeProgram->Program->SetUniformf("alpha", this->BackgroundAlpha);
    this->KeyholeProgram->Program->SetUniformf("radius", static_cast<float>((this->Radius * 1.0) / width));
    this->KeyholeProgram->Program->SetUniformf("aspect_ratio", static_cast<float>(width * 1.0 / height));
    this->KeyholeProgram->Program->SetUniformf("gamma", this->Gamma);
    this->KeyholeProgram->Program->SetUniformf("d1", static_cast<float>(this->D1 * 1.0 / width));
    this->KeyholeProgram->Program->SetUniformi("use_mask_texture", 0);
    this->KeyholeProgram->Program->SetUniformi("use_hard_edges", static_cast<int>(this->AllowHardEdges));

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

#ifdef VTK_KEYHOLE_PASS_DEBUG

    this->FrameBufferObject->RenderQuad(0, width - 1, 0, height - 1,
                                        this->KeyholeProgram->Program,
                                        this->KeyholeProgram->VAO);
    this->Pass1->Deactivate();

    // Save the output of the first pass to a file for debugging
    pbo = this->Pass2->Download();

    openglRawData = new unsigned char[4 * width * height];

    status = pbo->Download2D(VTK_UNSIGNED_CHAR, openglRawData, this->Dimensions, 4, increments);
    assert("check" && status);
    pbo->Delete();

    importer = vtkImageImport::New();
    importer->CopyImportVoidPointer(openglRawData, 4 * width * height * sizeof(unsigned char));
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents(4);
    importer->SetWholeExtent(0, width - 1, 0, height - 1, 0, 0);
    importer->SetDataExtentToWholeExtent();
    delete[] openglRawData;

    rgbatoRgb = vtkImageExtractComponents::New();
    rgbatoRgb->SetInputConnection(importer->GetOutputPort());
    rgbatoRgb->SetComponents(0, 1, 2);

    writer = vtkPNGWriter::New();
    writer->SetFileName("KeyholePass2.png");
    writer->SetInputConnection(rgbatoRgb->GetOutputPort());
    writer->Write();

    importer->Delete();
    rgbatoRgb->Delete();
    writer->Delete();
#endif
    // Render in the original FBO
    this->FrameBufferObject->UnBind();

    this->Pass2->Activate();

    // Render in the correct viewport
    glViewport(this->ViewPortX, this->ViewPortY, this->ViewPortWidth, this->ViewPortHeight);

    // Copy Pass2 to a FBO
    this->CopyToFrameBuffer(0, 0, width - 1, height - 1,
                            0, 0, width, height,
                            this->Pass2,
                            this->KeyholeProgram->Program,
                            this->KeyholeProgram->VAO);

    // Restore state
    glViewport(saved_viewport[0], saved_viewport[1], saved_viewport[2], saved_viewport[3]);

    this->Pass2->Deactivate();

    if (r->GetActiveCamera()->GetLeftEye())
    {
      this->LeftTextureObject->Deactivate();
    }
    else
    {
      this->RightTextureObject->Deactivate();
    }

    this->MaskTextureObject->Deactivate();
    this->ForegroundGradientTextureObject->Deactivate();
  }

  vtkOpenGLCheckErrorMacro("Failed after Render");
}

//------------------------------------------------------------------------------------------------------------
// Description:
// Compute gradient texture of the foreground and save it to a texture object
void vtkKeyholePass::GetForegroudGradient(vtkRenderer* r)
{
  vtkOpenGLRenderWindow* renwin = vtkOpenGLRenderWindow::SafeDownCast(r->GetRenderWindow());
  int* vsize = r->GetSize();
  int width = vsize[0];
  int height = vsize[1];

  GLint saved_viewport[4];
  glGetIntegerv(GL_VIEWPORT, saved_viewport);

  // Create new TOs and set FBO color attachments
  if (this->GX == NULL)
  {
    this->GX = vtkTextureObject::New();
    this->GX->SetContext(renwin);
  }
  if (this->GX->GetWidth() != static_cast<unsigned int>(width) ||
      this->GX->GetHeight() != static_cast<unsigned int>(height))
  {
    this->GX->Create2D(width, height, 4, VTK_UNSIGNED_CHAR, false);
  }

  if (this->GY == NULL)
  {
    this->GY = vtkTextureObject::New();
    this->GY->SetContext(renwin);
  }
  if (this->GY->GetWidth() != static_cast<unsigned int>(width) ||
      this->GY->GetHeight() != static_cast<unsigned int>(height))
  {
    this->GY->Create2D(width, height, 4, VTK_UNSIGNED_CHAR, false);
  }

  this->FrameBufferObject->Bind();
  this->FrameBufferObject->SetNumberOfRenderTargets(2);
  this->FrameBufferObject->SetColorBuffer(0, this->GX);
  this->FrameBufferObject->SetColorBuffer(1, this->GY);
  unsigned int indices[2] = { 0, 1 };
  this->FrameBufferObject->SetActiveBuffers(2, indices);
  this->FrameBufferObject->Start(width, height, false);

  // Set the shader program for the first pass of GX and GY
  if (this->GradientProgram1 == NULL)
  {
    this->GradientProgram1 = new vtkOpenGLHelper;
    // build the shader source code
    std::string data_dir = std::string(SHADER_DIRECTORY);
    LoadShaders(data_dir + "gradientMagPass1.fs", data_dir + "vtkKeyhole.vs");

    std::string GSSource;
    // compile and bind it if needed
    vtkShaderProgram* newShader =
      renwin->GetShaderCache()->ReadyShaderProgram(
        this->VertexShaderSource.c_str(),
        this->FragmentShaderSource.c_str(),
        GSSource.c_str());

    // if the shader changed reinitialize the VAO
    if (newShader != this->GradientProgram1->Program)
    {
      this->GradientProgram1->Program = newShader;
      this->GradientProgram1->VAO->ShaderProgramChanged(); // reset the VAO as the shader has changed
    }

    this->GradientProgram1->ShaderSourceTime.Modified();
  }
  else
  {
    renwin->GetShaderCache()->ReadyShaderProgram(this->GradientProgram1->Program);
  }

  if (this->GradientProgram1->Program->GetCompiled() != true)
  {
    vtkErrorMacro("Couldn't build the shader (Gradient 1) shader program. At this point it can be an error in the shader or a driver bug.");
    return;
  }

  int sourceID;
  if (r->GetActiveCamera()->GetLeftEye())
  {
    this->LeftTextureObject->Activate();
    sourceID = this->LeftTextureObject->GetTextureUnit();
  }
  else
  {
    this->RightTextureObject->Activate();
    sourceID = this->RightTextureObject->GetTextureUnit();
  }

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  this->GradientProgram1->Program->SetUniformi("source", sourceID);

  float fvalue = static_cast<float>(1.0 / width);
  this->GradientProgram1->Program->SetUniformf("stepSize", fvalue);


  // Save viewport state and render at (0, 0, width, height)
  //GLint saved_viewport[4];
  //glGetIntegerv(GL_VIEWPORT, saved_viewport);
  //glViewport(0, 0, width, height);

  this->FrameBufferObject->RenderQuad(0, width - 1, 0, height - 1,
                                      this->GradientProgram1->Program,
                                      this->GradientProgram1->VAO);

  this->FrameBufferObject->UnBind();

  // Restore viewport
  glViewport(saved_viewport[0], saved_viewport[1], saved_viewport[2],
             saved_viewport[3]);

  if (r->GetActiveCamera()->GetLeftEye())
  {
    this->LeftTextureObject->Deactivate();
  }
  else
  {
    this->RightTextureObject->Deactivate();
  }

#ifdef VTK_KEYHOLE_PASS_DEBUG
  // Save the output for debugging
  vtkPixelBufferObject* pbo = GX->Download();

  unsigned int dims[2] = { width, height };

  unsigned char* openglRawData = new unsigned char[4 * dims[0] * dims[1]];
  vtkIdType incs[2];
  incs[0] = 0;
  incs[1] = 0;
  bool status = pbo->Download2D(VTK_UNSIGNED_CHAR, openglRawData, dims, 4, incs);
  assert("check" && status);
  pbo->Delete();

  vtkImageImport* importer = vtkImageImport::New();
  importer->CopyImportVoidPointer(openglRawData, 4 * dims[0] * dims[1] * sizeof(unsigned char));
  importer->SetDataScalarTypeToUnsignedChar();
  importer->SetNumberOfScalarComponents(4);
  importer->SetWholeExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, 0);
  importer->SetDataExtentToWholeExtent();
  delete[] openglRawData;

  vtkImageExtractComponents* rgbatoRgb = vtkImageExtractComponents::New();
  rgbatoRgb->SetInputConnection(importer->GetOutputPort());
  rgbatoRgb->SetComponents(0, 1, 2);

  vtkPNGWriter* writer = vtkPNGWriter::New();
  writer->SetFileName("Foreground_Grad_Pass1.png");
  writer->SetInputConnection(rgbatoRgb->GetOutputPort());
  writer->Write();

  importer->Delete();
  writer->Delete();
  rgbatoRgb->Delete();
#endif

  if (this->ForegroundGradientTextureObject == NULL)
  {
    this->ForegroundGradientTextureObject = vtkTextureObject::New();
    this->ForegroundGradientTextureObject->SetContext(renwin);
  }

  if (this->ForegroundGradientTextureObject->GetWidth() != static_cast<unsigned int>(width) ||
      this->ForegroundGradientTextureObject->GetHeight() != static_cast<unsigned int>(height))
  {
    this->ForegroundGradientTextureObject->Create2D(static_cast<unsigned int>(width),
        static_cast<unsigned int>(height), 4,
        VTK_UNSIGNED_CHAR, false);
  }

  // Now bind foreground_grad_to to the FBO
  this->FrameBufferObject->Bind();
  this->FrameBufferObject->SetNumberOfRenderTargets(1);
  this->FrameBufferObject->SetColorBuffer(0, ForegroundGradientTextureObject);
  this->FrameBufferObject->SetActiveBuffer(0);
  this->FrameBufferObject->Start(width, height, false);

  // Set the shader program for the second pass of GX and GY
  if (this->KeyholeShader == NULL)
  {
    this->KeyholeShader = new vtkOpenGLHelper;
    // build the shader source code
    std::string data_dir = std::string(SHADER_DIRECTORY);
    LoadShaders(data_dir + "gradientMagPass2.fs", data_dir + "vtkKeyhole.vs");

    std::string GSSource;
    // compile and bind it if needed
    vtkShaderProgram* newShader2 =
      renwin->GetShaderCache()->ReadyShaderProgram(
        this->VertexShaderSource.c_str(),
        this->FragmentShaderSource.c_str(),
        GSSource.c_str());

    // if the shader changed reinitialize the VAO
    if (newShader2 != this->KeyholeShader->Program)
    {
      this->KeyholeShader->Program = newShader2;
      this->KeyholeShader->VAO->ShaderProgramChanged(); // reset the VAO as the shader has changed
    }

    this->KeyholeShader->ShaderSourceTime.Modified();
  }
  else
  {
    renwin->GetShaderCache()->ReadyShaderProgram(this->KeyholeShader->Program);
  }

  if (this->KeyholeShader->Program->GetCompiled() != true)
  {
    vtkErrorMacro("Couldn't build the shader (Gradient 2) shader program. At this point it can be an error in the shader or a driver bug.");
    return;
  }

  // Set GX and GY as source
  this->GX->Activate();
  int id0 = this->GX->GetTextureUnit();
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  this->GY->Activate();
  int id1 = this->GY->GetTextureUnit();
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  this->KeyholeShader->Program->SetUniformi("gx1", id0);
  this->KeyholeShader->Program->SetUniformi("gy1", id1);

  fvalue = static_cast<float>(1.0 / height);
  this->KeyholeShader->Program->SetUniformf("stepSize", fvalue);

  // Save viewport state and render at (0, 0, width, height)
  //glGetIntegerv(GL_VIEWPORT, saved_viewport);
  //glViewport(0, 0, width, height);

  this->FrameBufferObject->RenderQuad(0, width - 1, 0, height - 1,
                                      this->KeyholeShader->Program,
                                      this->KeyholeShader->VAO);
  this->FrameBufferObject->UnBind();
  this->GX->Deactivate();
  this->GY->Deactivate();

  // Restore viewport
  glViewport(saved_viewport[0], saved_viewport[1], saved_viewport[2],
             saved_viewport[3]);

#ifdef VTK_KEYHOLE_PASS_DEBUG
  // Save the output for debugging
  pbo = this->ForegroundGradientTextureObject->Download();

  openglRawData = new unsigned char[4 * dims[0] * dims[1]];
  status = pbo->Download2D(VTK_UNSIGNED_CHAR, openglRawData, dims, 4, incs);
  assert("check" && status);
  pbo->Delete();

  importer = vtkImageImport::New();
  importer->CopyImportVoidPointer(openglRawData, 4 * dims[0] * dims[1] * sizeof(unsigned char));
  importer->SetDataScalarTypeToUnsignedChar();
  importer->SetNumberOfScalarComponents(4);
  importer->SetWholeExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, 0);
  importer->SetDataExtentToWholeExtent();
  delete[] openglRawData;

  rgbatoRgb = vtkImageExtractComponents::New();
  rgbatoRgb->SetInputConnection(importer->GetOutputPort());
  rgbatoRgb->SetComponents(0, 1, 2);

  writer = vtkPNGWriter::New();
  writer->SetFileName("Foreground_Grad_Pass2.png");
  writer->SetInputConnection(rgbatoRgb->GetOutputPort());
  writer->Write();

  importer->Delete();
  writer->Delete();
  rgbatoRgb->Delete();
#endif

}

//-----------------------------------------------------------------------------------------------------
void vtkKeyholePass::UpdateLeftTextureObject(vtkOpenGLRenderWindow* renwin)
{
  int vtktype = this->LeftPixelBufferObject->GetType();
  GLenum type = this->LeftTextureObject->GetDefaultDataType(vtktype);

  GLenum internalFormat = this->LeftTextureObject->GetInternalFormat(vtktype,
                          this->Components,
                          false);
  GLenum format = this->LeftTextureObject->GetFormat(vtktype,
                  this->Components,
                  false);

  int width = this->LeftTextureObject->GetWidth();
  int height = this->LeftTextureObject->GetHeight();

  renwin->ActivateTexture(this->LeftTextureObject);
  this->LeftTextureObject->Bind();

  this->LeftPixelBufferObject->Bind(vtkPixelBufferObject::UNPACKED_BUFFER);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  glTexImage2D(GL_TEXTURE_2D,
               0,
               internalFormat,
               static_cast<GLsizei>(width),
               static_cast<GLsizei>(height),
               0,
               format,
               type,
               0);

  vtkOpenGLCheckErrorMacro("failed at glTexImage2D");

  this->LeftPixelBufferObject->UnBind();
  this->LeftTextureObject->Deactivate();
}

//----------------------------------------------------------------------------------------------------
void vtkKeyholePass::UpdateRightTextureObject(vtkOpenGLRenderWindow* renwin)
{
  int vtktype = this->RightPixelBufferObject->GetType();
  GLenum type = this->RightTextureObject->GetDefaultDataType(vtktype);

  GLenum internalFormat = this->RightTextureObject->GetInternalFormat(vtktype,
                          this->Components,
                          false);
  GLenum format = this->RightTextureObject->GetFormat(vtktype,
                  this->Components,
                  false);

  int width = this->RightTextureObject->GetWidth();
  int height = this->RightTextureObject->GetHeight();

  renwin->ActivateTexture(this->RightTextureObject);
  this->RightTextureObject->Bind();

  this->RightPixelBufferObject->Bind(vtkPixelBufferObject::UNPACKED_BUFFER);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  glTexImage2D(GL_TEXTURE_2D,
               0,
               internalFormat,
               static_cast<GLsizei>(width),
               static_cast<GLsizei>(height),
               0,
               format,
               type,
               0);

  vtkOpenGLCheckErrorMacro("failed at glTexImage2D");

  this->RightPixelBufferObject->UnBind();
  this->RightTextureObject->Deactivate();
}

//-----------------------------------------------------------------------------------------------------
void vtkKeyholePass::CopyToFrameBuffer(
  int srcXmin, int srcYmin,
  int srcXmax, int srcYmax,
  int dstXmin, int dstYmin,
  int dstXmax, int dstYmax,
  vtkTextureObject* to,
  vtkShaderProgram* program, vtkOpenGLVertexArrayObject* vao)
{
  assert("pre: positive_srcXmin" && srcXmin >= 0);
  assert("pre: max_srcXmax" &&
         static_cast<unsigned int>(srcXmax) < to->GetWidth());
  assert("pre: increasing_x" && srcXmin <= srcXmax);
  assert("pre: positive_srcYmin" && srcYmin >= 0);
  assert("pre: max_srcYmax" &&
         static_cast<unsigned int>(srcYmax) < to->GetHeight());
  assert("pre: increasing_y" && srcYmin <= srcYmax);
  assert("pre: positive_dstXmin" && dstXmin >= 0);
  assert("pre: positive_dstYmin" && dstYmin >= 0);

  float minXTexCoord = static_cast<float>(
                         static_cast<double>(srcXmin + 0.5) / to->GetWidth());
  float minYTexCoord = static_cast<float>(
                         static_cast<double>(srcYmin + 0.5) / to->GetHeight());

  float maxXTexCoord = static_cast<float>(
                         static_cast<double>(srcXmax + 0.5) / to->GetWidth());
  float maxYTexCoord = static_cast<float>(
                         static_cast<double>(srcYmax + 0.5) / to->GetHeight());

  float tcoords[] =
  {
    minXTexCoord, minYTexCoord,
    maxXTexCoord, minYTexCoord,
    maxXTexCoord, maxYTexCoord,
    minXTexCoord, maxYTexCoord
  };

  float dstSizeX = static_cast<float>(dstXmin + srcXmax - srcXmin);
  float dstSizeY = static_cast<float>(dstYmin + srcYmax - srcYmin);

  float verts[] =
  {
    2.0f * dstXmin / dstSizeX - 1.0f, 2.0f * dstYmin / dstSizeY - 1.0f, 0.0f,
    2.0f * (dstXmax + 1.0f) / dstSizeX - 1.0f, 2.0f * dstYmin / dstSizeY - 1.0f, 0.0f,
    2.0f * (dstXmax + 1.0f) / dstSizeX - 1.0f, 2.0f * (dstYmax + 1.0f) / dstSizeY - 1.0f, 0.0f,
    2.0f * dstXmin / dstSizeX - 1.0f, 2.0f * (dstYmax + 1.0f) / dstSizeY - 1.0f, 0.0f
  };

  to->CopyToFrameBuffer(tcoords, verts, program, vao);
}

//-----------------------------------------------------------------------------------------------------
void vtkKeyholePass::ProbeSupport(const vtkRenderState* s)
{
  vtkRenderer* r = s->GetRenderer();
  vtkOpenGLRenderWindow* renwin = vtkOpenGLRenderWindow::SafeDownCast(r->GetRenderWindow());

  // Test hardware support. If not supported, just render the delegate.
  bool supported = RobartsVTKFrameBufferObject::IsSupported(renwin);

  if (!supported)
  {
    vtkErrorMacro("FBOs are not supported by the context. Cannot render keyhole. ");
  }
  if (supported)
  {
    supported = vtkTextureObject::IsSupported(renwin);

    if (!supported)
    {
      vtkErrorMacro("Texture objects are not supported by the context. Cannot render keyhole. ");
    }
  }
  if (supported)
  {
    RobartsVTKFrameBufferObject* fbo = RobartsVTKFrameBufferObject::SafeDownCast(s->GetFrameBuffer());
    // FBO extension is supported. Is the specific FBO format supported?
    if (fbo == 0)
    {
      // get the viewport dimensions
      r->GetTiledSizeAndOrigin(&this->ViewPortWidth, &this->ViewPortHeight, &this->ViewPortX, &this->ViewPortY);
      if (this->FrameBufferObject == NULL)
      {
        this->FrameBufferObject = RobartsVTKFrameBufferObject::New();
        this->FrameBufferObject->SetContext(renwin);

        /* Drawbuffers need to be set-up to be stereo here. vtkCameraPass assumes that FBO is setup appropriately. */
        this->SetupDrawBuffers(r);
      }
    }
    else
    {
      this->FrameBufferObject = fbo;
    }

    if (this->Pass1 == NULL)
    {
      this->Pass1 = vtkTextureObject::New();
      this->Pass1->SetContext(renwin);
    }

    if (this->LeftTextureObject == NULL)
    {
      this->LeftTextureObject = vtkTextureObject::New();
      this->LeftTextureObject->SetContext(renwin);
    }
    if (this->RightTextureObject == NULL)
    {
      this->RightTextureObject = vtkTextureObject::New();
      this->RightTextureObject->SetContext(renwin);
    }
    if (this->MaskTextureObject == NULL)
    {
      this->MaskTextureObject = vtkTextureObject::New();
      this->MaskTextureObject->SetContext(renwin);
    }
    if (this->LeftPixelBufferObject == NULL)
    {
      this->LeftPixelBufferObject = vtkPixelBufferObject::New();
      this->LeftPixelBufferObject->SetContext(renwin);
    }
    if (this->RightPixelBufferObject == NULL)
    {
      this->RightPixelBufferObject = vtkPixelBufferObject::New();
      this->RightPixelBufferObject->SetContext(renwin);
    }
    if (this->MaskPixelBufferObject == NULL)
    {
      this->MaskPixelBufferObject = vtkPixelBufferObject::New();
      this->MaskPixelBufferObject->SetContext(renwin);
    }
    if (this->ForegroundGradientTextureObject == NULL)
    {
      this->ForegroundGradientTextureObject = vtkTextureObject::New();
      this->ForegroundGradientTextureObject->SetContext(renwin);
    }

    this->Pass1->Create2D(64, 64, 4, VTK_UNSIGNED_CHAR, false);
    this->FrameBufferObject->SetColorBuffer(0, this->Pass1);
    this->FrameBufferObject->SetNumberOfRenderTargets(1);
    this->FrameBufferObject->SetActiveBuffer(0);
    this->FrameBufferObject->SetDepthBufferNeeded(true);

    supported = this->FrameBufferObject->StartNonOrtho(64, 64, false);

    if (!supported)
    {
      vtkErrorMacro("The requested FBO format is not supported by the context. Cannot render keyhole. ");
      this->FrameBufferObject->UnBind();
    }
    else
    {
      this->FrameBufferObject->UnBind();
    }
  }

  this->Supported = supported;
  this->SupportProbed = true;
}

//-----------------------------------------------------------------------------------------------------
int vtkKeyholePass::ReadTextures(vtkRenderer* r)
{
  //Do this for the background and the mask.

  vtkPropCollection* props = r->GetViewProps();
  int numActors = props->GetNumberOfItems();
  props->InitTraversal();

  vtkImageData* imgData;
  unsigned char* dataPtr;
  int img_size[3], maxRegularActorIDx, backgroundIDx, maskIDx;
  vtkIdType increments[2];
  increments[0] = 0;
  increments[1] = 0;

  if (this->Stereo)
  {
    maxRegularActorIDx = this->MaskImageAvailable ? numActors - 3 : numActors - 2;
    backgroundIDx = this->MaskImageAvailable ? numActors - 3 : numActors - 2;
  }
  else
  {
    maxRegularActorIDx = this->MaskImageAvailable ? numActors - 2 : numActors - 1;
    backgroundIDx = this->MaskImageAvailable ? numActors - 2 : numActors - 1;
  }

  maskIDx = this->MaskImageAvailable ? numActors - 1 : numActors + 1;

  for (int i = 0; i < numActors; i++)
  {
    if (i < maxRegularActorIDx)
    {
      // Disregard the first set of actors
      props->GetNextProp();
    }
    else if (i == backgroundIDx)
    {
      vtkActor* leftImageActor = vtkActor::SafeDownCast(props->GetNextProp()); // actors->GetNextActor();
      vtkTexture* leftImageTexture = leftImageActor->GetTexture();

      imgData = leftImageTexture->GetInput();
      imgData->GetDimensions(img_size);

      if (img_size[0] > 0 && img_size[1] > 0 && img_size[2] > 0) // If texture is initialized
      {

        this->Components = imgData->GetNumberOfScalarComponents();
        dataPtr = (unsigned char*)imgData->GetScalarPointer();

        this->Dimensions[0] = img_size[0];
        this->Dimensions[1] = img_size[1];

        // Upload imagedata to pixel buffer object
        bool success = this->LeftPixelBufferObject->Upload2D(VTK_UNSIGNED_CHAR,
                       dataPtr, this->Dimensions,
                       this->Components, increments);

        if (this->Stereo)
        {
          // Get right iamge texture
          vtkActor* rightImageActor = vtkActor::SafeDownCast(props->GetNextProp());
          vtkTexture* rightImageTexture = rightImageActor->GetTexture();
          imgData = rightImageTexture->GetInput();

          dataPtr = (unsigned char*)imgData->GetScalarPointer();

          success = this->RightPixelBufferObject->Upload2D(VTK_UNSIGNED_CHAR,
                    dataPtr, this->Dimensions,
                    this->Components, increments);

          // Now increment i so that mask texture can be acquired
          i++;
        }

#ifdef VTK_KEYHOLE_PASS_DEBUG2
        vtkPNGWriter* writer = vtkPNGWriter::New();
        writer->SetInputData(imgData);
        writer->SetFileName("KeyholePass0.png");
        writer->Write();

#endif
      }
      else
      {
        std::cerr << "[vtkKeyholePass] Background texture is not initialized" << std::endl;
        return -1;
      }
    }
    else if (i == maskIDx && this->MaskImageAvailable)
    {
      vtkActor* maskActor = vtkActor::SafeDownCast(props->GetNextProp());; // = actors->GetNextActor();
      vtkTexture* MaskTexture = maskActor->GetTexture();


      vtkImageData* imgData = MaskTexture->GetInput();
      imgData->GetDimensions(img_size);

      if (img_size[0] > 0 && img_size[1] > 0 && img_size[2] > 0) // if texture is initialized
      {
        imgData->GetDimensions(img_size);
        this->Components = imgData->GetNumberOfScalarComponents();
        dataPtr = (unsigned char*)imgData->GetScalarPointer();

        this->Dimensions[0] = img_size[0];
        this->Dimensions[1] = img_size[1];

        // Upload imagedata to pixel buffer object
        this->MaskPixelBufferObject->Upload2D(VTK_UNSIGNED_CHAR,
                                              dataPtr, this->Dimensions,
                                              this->Components, increments);
      }
      else
      {
        std::cerr << "[vtkKeyholePass] Mask texture is not initialized" << std::endl;
        return -1;
      }
    }
  }

  return 0;
}

//-----------------------------------------------------------------------------------------------------
void vtkKeyholePass::SetupDrawBuffers(vtkRenderer* ren)
{
  vtkCamera* camera = ren->GetActiveCamera();

  vtkOpenGLRenderWindow* win = vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow());
  win->MakeCurrent();

  // find out if we should stereo render
  bool stereo = win->GetStereoRender() == 1;

  ren->GetTiledSizeAndOrigin(&this->ViewPortWidth, &this->ViewPortHeight,
                             &this->ViewPortX, &this->ViewPortY + 1);

  // if were on a stereo renderer draw to special parts of screen
  if (stereo)
  {
    switch (win->GetStereoType()) // User left-right buffers for stereo
    {
      case VTK_STEREO_CRYSTAL_EYES:
      case VTK_STEREO_DRESDEN:
      case VTK_STEREO_SPLITVIEWPORT_HORIZONTAL:
      case VTK_STEREO_CHECKERBOARD:
      case VTK_STEREO_INTERLACED:
      case VTK_STEREO_RED_BLUE:
      case VTK_STEREO_ANAGLYPH:
        if (camera->GetLeftEye())
        {
          if (win->GetDoubleBuffer())
          {
            glDrawBuffer(static_cast<GLenum>(win->GetBackLeftBuffer()));
            glReadBuffer(static_cast<GLenum>(win->GetBackLeftBuffer()));
          }
          else
          {
            glDrawBuffer(static_cast<GLenum>(win->GetFrontLeftBuffer()));
            glReadBuffer(static_cast<GLenum>(win->GetFrontLeftBuffer()));
          }
        }
        else
        {
          if (win->GetDoubleBuffer())
          {
            glDrawBuffer(static_cast<GLenum>(win->GetBackRightBuffer()));
            glReadBuffer(static_cast<GLenum>(win->GetBackRightBuffer()));
          }
          else
          {
            glDrawBuffer(static_cast<GLenum>(win->GetFrontRightBuffer()));
            glReadBuffer(static_cast<GLenum>(win->GetFrontRightBuffer()));
          }
        }
        break;
      case VTK_STEREO_LEFT:
        camera->SetLeftEye(1);
        break;
      case VTK_STEREO_RIGHT:
        camera->SetLeftEye(0);
        break;
      default:
        break;
    }
  }
  else
  {
    if (win->GetDoubleBuffer())
    {
      glDrawBuffer(static_cast<GLenum>(win->GetBackBuffer()));

      // Reading back buffer means back left. see OpenGL spec.
      // because one can write to two buffers at a time but can only read from
      // one buffer at a time.
      glReadBuffer(static_cast<GLenum>(win->GetBackBuffer()));
    }
    else
    {
      glDrawBuffer(static_cast<GLenum>(win->GetFrontBuffer()));

      // Reading front buffer means front left. see OpenGL spec.
      // because one can write to two buffers at a time but can only read from
      // one buffer at a time.
      glReadBuffer(static_cast<GLenum>(win->GetFrontBuffer()));
    }
  }

  vtkOpenGLCheckErrorMacro("failed after restore context");
}

//-----------------------------------------------------------------------------------------------------
// Description:
// Release graphics resources and ask components to release their own
// resource.
// \pre w_exist: w!=0
void vtkKeyholePass::ReleaseGraphicsResources(vtkWindow* w)
{

  assert("pre: w_exists" && w != NULL);

  vtkMultiViewportImageProcessingPass::ReleaseGraphicsResources(w);

  if (this->KeyholeProgram != NULL)
  {
    this->KeyholeProgram->Program->ReleaseGraphicsResources(w);
  }

  if (this->FrameBufferObject != NULL)
  {
    this->FrameBufferObject->Delete();
    this->FrameBufferObject = 0;
  }

  if (this->Pass1 != NULL)
  {
    this->Pass1->Delete();
    this->Pass1 = 0;
  }

  if (this->Pass2 != NULL)
  {
    this->Pass2->Delete();
    this->Pass2 = 0;
  }

  if (this->LeftPixelBufferObject != NULL)
  {
    this->LeftPixelBufferObject->Delete();
    this->LeftPixelBufferObject = 0;
  }

  if (this->RightPixelBufferObject != NULL)
  {
    this->RightPixelBufferObject->Delete();
    this->RightPixelBufferObject = 0;
  }

  if (this->LeftTextureObject != NULL)
  {
    this->LeftTextureObject->Delete();
    this->LeftTextureObject = 0;
  }

  if (this->GX != NULL)
  {
    this->GX->Delete();
    this->GX = 0;
  }

  if (this->GY != NULL)
  {
    this->GY->Delete();
    this->GY = 0;
  }

  if (this->ForegroundGradientTextureObject != NULL)
  {
    this->ForegroundGradientTextureObject->Delete();
    this->ForegroundGradientTextureObject = 0;
  }

  if (this->MaskPixelBufferObject != NULL)
  {
    this->MaskPixelBufferObject->Delete();
    this->MaskPixelBufferObject = 0;
  }

  if (this->MaskTextureObject != NULL)
  {
    this->MaskTextureObject->Delete();
    this->MaskTextureObject = 0;
  }
}

//----------------------------------------------------------------------------
void vtkKeyholePass::UseMaskImage(bool t)
{
  this->MaskImageAvailable = t;
}

//----------------------------------------------------------------------------
void vtkKeyholePass::SetLeftKeyholeParameters(int x, int y, int r, double g)
{
  this->xL = x;
  this->yL = y;
  this->Radius = r;
  this->Gamma = static_cast<float>(g);
}

//----------------------------------------------------------------------------
void vtkKeyholePass::SetRightKeyholeParameters(int x, int y, int r, double g)
{
  this->xR = x;
  this->yR = y;
  this->Radius = r;
  this->Gamma = static_cast<float>(g);
}

//----------------------------------------------------------------------------
void vtkKeyholePass::SetHardKeyholeEdges(bool t)
{
  this->AllowHardEdges = t;
}

//----------------------------------------------------------------------------
void vtkKeyholePass::SetBackgroundColor(double r, double g, double b)
{
  this->BackgroundRed = r;
  this->BackgroundGreen = g;
  this->BackgroundBlue = b;
}

//----------------------------------------------------------------------------
void vtkKeyholePass::SetVisualizationMode(RenderingMode _mode)
{
  this->Mode = _mode;
}

//----------------------------------------------------------------------------
void vtkKeyholePass::SetAlpha(double _alpha)
{
  this->BackgroundAlpha = _alpha;
}

//----------------------------------------------------------------------------
void vtkKeyholePass::SetD1(double _d1)
{
  this->D1 = _d1;
}

//----------------------------------------------------------------------------------------------------------
// Description:
// Load Vertex and Fragment Shaders from files.
void vtkKeyholePass::LoadShaders(std::string fs_path, std::string vs_path)
{
  std::ifstream fShaderFile, vShaderFile;
  fShaderFile.exceptions(std::ifstream::badbit);
  vShaderFile.exceptions(std::ifstream::badbit);

  // Read Fragment Shader File
  try
  {
    // Open file
    fShaderFile.open(fs_path);
    std::stringstream fShaderStream;
    fShaderStream << fShaderFile.rdbuf();

    fShaderFile.close();

    this->FragmentShaderSource = fShaderStream.str();

  }
  catch (std::ifstream::failure e)
  {
    std::cerr << "ERROR:: FS FILE READING ERROR. " << std::endl;
  }

  // Read Vertex Shader File
  try
  {
    // Open file
    vShaderFile.open(vs_path);
    std::stringstream vShaderStream;
    vShaderStream << vShaderFile.rdbuf();

    vShaderFile.close();

    this->VertexShaderSource = vShaderStream.str();

  }
  catch (std::ifstream::failure e)
  {
    std::cerr << "ERROR:: VS FILE READING ERROR. " << std::endl;
  }

  return;
}