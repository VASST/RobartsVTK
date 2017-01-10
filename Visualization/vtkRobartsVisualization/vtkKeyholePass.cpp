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

#include "RobartsVTKConfigure.h"

#include "vtkFrameBufferObject.h"
#include "vtkKeyholePass.h"
#include "vtkObjectFactory.h"
#include "vtkOpenGLError.h"
#include "vtkOpenGLHelper.h"
#include "vtkOpenGLRenderWindow.h"
#include "vtkOpenGLRenderWindow.h"
#include "vtkOpenGLShaderCache.h"
#include "vtkOpenGLTexture.h"
#include "vtkOpenGLVertexArrayObject.h"
#include "vtkProperty.h"
#include "vtkRenderState.h"
#include "vtkRenderer.h"
#include "vtkShaderProgram.h"
#include "vtkTextureObject.h"
#include "vtkTextureUnitManager.h"

// To be able to dump intermediate passes into png images for debugging.
//#define VTK_KEYHOLE_PASS_DEBUG

#include "vtkCamera.h"
#include "vtkImageExtractComponents.h"
#include "vtkImageImport.h"
#include "vtkMath.h"
#include "vtkPNGWriter.h"
#include "vtkPixelBufferObject.h"
#include "vtkPixelBufferObject.h"

#include <cassert>
#include <sstream>

vtkStandardNewMacro(vtkKeyholePass);

//----------------------------------------------------------------------------------------------------
vtkKeyholePass::vtkKeyholePass()
  : FrameBufferObject(NULL),
    Pass1(NULL),
    Pass2(NULL),
    ForegroundPixelBufferObject(NULL),
    MaskPixelBufferObject(NULL),
    ForegroundTextureObject(NULL),
    MaskTextureObject(NULL),
    GX(NULL),
    GY(NULL),
    ForegroundGradientTextureObject(NULL),
    KeyholeProgram(NULL),
    GradientProgram1(NULL),
    KeyholeShader(NULL),
    Supported(false),
    SupportProbed(false),
    allow_hard_edges(false),
    mask_img_available(false)
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
  vtkImageProcessingPass::PrintSelf(os, indent);
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
  vtkOpenGLRenderWindow* renwin = vtkOpenGLRenderWindow::SafeDownCast(r->GetRenderWindow());
  if (this->DelegatePass != NULL)
  {
    if (!this->SupportProbed)
    {
      this->SupportProbed = true;

      // Test hardware support. If not supported, just render the delegate.
      bool supported = vtkFrameBufferObject::IsSupported(renwin);

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

        // FBO extension is supported. Is the specific FBO format supported?
        if (this->FrameBufferObject == NULL)
        {
          this->FrameBufferObject = vtkFrameBufferObject::New();
          this->FrameBufferObject->SetContext(renwin);
        }
        if (this->Pass1 == NULL)
        {
          this->Pass1 = vtkTextureObject::New();
          this->Pass1->SetContext(renwin);
        }
        if (this->ForegroundTextureObject == NULL)
        {
          this->ForegroundTextureObject = vtkTextureObject::New();
          this->ForegroundTextureObject->SetContext(renwin);
        }
        if (this->MaskTextureObject == NULL)
        {
          this->MaskTextureObject = vtkTextureObject::New();
          this->MaskTextureObject->SetContext(renwin);
        }
        if (this->ForegroundPixelBufferObject == NULL)
        {
          this->ForegroundPixelBufferObject = vtkPixelBufferObject::New();
          this->ForegroundPixelBufferObject->SetContext(renwin);
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
        }
        else
        {
          this->FrameBufferObject->UnBind();
        }
      }

      this->Supported = supported;
    }

    // If not supported.
    if (!this->Supported)
    {

      this->DelegatePass->Render(s);
      this->NumberOfRenderedProps += this->DelegatePass->GetNumberOfRenderedProps();

      return;
    }

    // Read image data into pixelBuffers. Do this for the background and the mask.
    vtkPropCollection* props = r->GetViewProps();
    int numActors = props->GetNumberOfItems();
    props->InitTraversal();

    vtkImageData* imgData;
    unsigned char* dataPtr;
    int img_size[3];
    vtkIdType increments[2];
    increments[0] = 0;
    increments[1] = 0;

    for (int i = 0; i < numActors; i++)
    {
      if (i == 0)
      {
        // Discard the first actor
        props->GetNextProp();
      }
	  else if (i == 1)
	  {
		  vtkActor* foregroundActor = vtkActor::SafeDownCast(props->GetNextProp()); // actors->GetNextActor();
		  this->ForegroundTexture = foregroundActor->GetTexture();

		  imgData = this->ForegroundTexture->GetInput();
		  imgData->GetDimensions(img_size);

		  if (img_size[0] > 0 && img_size[1] > 0 && img_size[2] >0) // If texture is initialized
		  {

			  this->components = imgData->GetNumberOfScalarComponents();
			  dataPtr = (unsigned char*)imgData->GetScalarPointer();

			  this->dimensions[0] = img_size[0];
			  this->dimensions[1] = img_size[1];

			  // Upload imagedata to pixel buffer object
			  bool success = this->ForegroundPixelBufferObject->Upload2D(VTK_UNSIGNED_CHAR,
				  dataPtr, this->dimensions,
				  this->components, increments);

#ifdef VTK_KEYHOLE_PASS_DEBUG2
			  vtkPNGWriter *writer =  vtkPNGWriter::New();
			  writer->SetInputData(imgData);
			  writer->SetFileName("KeyholePass0.png");
			  writer->Write();

#endif
		  }
		  else
		  {
			  std::cerr << "[vtkKeyholePass] Background texture is not initialized" << std::endl;
			  return;
		  }
      }
      else if (i == 2)
      {
        vtkActor* maskActor = vtkActor::SafeDownCast(props->GetNextProp());; // = actors->GetNextActor();
        this->MaskTexture = maskActor->GetTexture();


        vtkImageData* imgData = this->MaskTexture->GetInput();
		imgData->GetDimensions(img_size);

		if (img_size[0] > 0 && img_size[1] > 0 && img_size[2] >0) // if texture is initialized
		{
			imgData->GetDimensions(img_size);
			this->components = imgData->GetNumberOfScalarComponents();
			dataPtr = (unsigned char*)imgData->GetScalarPointer();

			this->dimensions[0] = img_size[0];
			this->dimensions[1] = img_size[1];

			// Upload imagedata to pixel buffer object
			this->MaskPixelBufferObject->Upload2D(VTK_UNSIGNED_CHAR,
				dataPtr, this->dimensions,
				this->components, increments);
		}
		else
		{
			std::cerr << "[vtkKeyholePass] Mask texture is not initialized" << std::endl;
			return;
		}
      }
    }

    // Create foreground texture object .
    if (this->ForegroundTextureObject->GetWidth() != static_cast<unsigned int>(this->dimensions[0]) ||
        this->ForegroundTextureObject->GetHeight() != static_cast<unsigned int>(this->dimensions[1]))
    {
		this->ForegroundTextureObject->Create2D(this->dimensions[0], this->dimensions[1], this->components,
                                              this->ForegroundPixelBufferObject, false);
    }
	else
	{
		UpdateTextureObject(renwin);
	}

    // Create mask texture object.
    if (this->mask_img_available)
    {
      if (this->MaskTextureObject->GetHeight() != static_cast<unsigned int>(this->dimensions[0]) ||
          this->MaskTextureObject->GetWidth() != static_cast<unsigned int>(this->dimensions[1]))

        this->MaskTextureObject->Create2D(this->dimensions[0], this->dimensions[1], this->components,
                                          this->MaskPixelBufferObject, false);
    }

    // 1. Create a new render state with FBO.
    int width;
    int height;
    int size[2];
    s->GetWindowSize(size);
    width = size[0];
    height  = size[1];

    if (this->Pass1 == NULL)
    {
      this->Pass1 = vtkTextureObject::New();
      this->Pass1->SetContext(renwin);
    }

    if (this->FrameBufferObject == NULL)
    {
      this->FrameBufferObject = vtkFrameBufferObject::New();
      this->FrameBufferObject->SetContext(renwin);
    }

    // Remove background texture
    r->SetTexturedBackground(false);
    // Now set a black background.
    r->SetBackground(0, 0, 0);

    // First pass
    this->RenderDelegate(s, width, height, width, height, this->FrameBufferObject,
                         this->Pass1);

#ifdef VTK_KEYHOLE_PASS_DEBUG
    // Save the output of the first pass to a file for debugging
	vtkPixelBufferObject* pbo = this->ForegroundTextureObject->Download();

    unsigned char* openglRawData = new unsigned char[4 * width * height];
    bool status = pbo->Download2D(VTK_UNSIGNED_CHAR, openglRawData, this->dimensions, 4, increments);
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

    if (this->mask_img_available)
    {
      this->MaskTextureObject->Activate();
      int texture1 = this->MaskTextureObject->GetTextureUnit();
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      this->KeyholeProgram->Program->SetUniformi("_mask", texture1);
    }

    this->ForegroundTextureObject->Activate();
    int texture2 = this->ForegroundTextureObject->GetTextureUnit();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    this->KeyholeProgram->Program->SetUniformi("_foreground", texture2);

    this->ForegroundGradientTextureObject->Activate();
    int texture3 = this->ForegroundGradientTextureObject->GetTextureUnit();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    this->KeyholeProgram->Program->SetUniformi("_foreground_grad", texture3);

    this->KeyholeProgram->Program->SetUniformf("th", 0.04);
    this->KeyholeProgram->Program->SetUniformf("x0", static_cast<float>((this->x0 * 1.0) / width));
    this->KeyholeProgram->Program->SetUniformf("y0", static_cast<float>((this->y0 * 1.0) / height));
    this->KeyholeProgram->Program->SetUniformf("radius", static_cast<float>((this->radius * 1.0) / width));
    this->KeyholeProgram->Program->SetUniformf("aspect_ratio", static_cast<float>(width * 1.0 / height));
    this->KeyholeProgram->Program->SetUniformf("gamma",  this->gamma);
    this->KeyholeProgram->Program->SetUniformi("use_mask_texture", 0);
    this->KeyholeProgram->Program->SetUniformi("use_hard_edges", static_cast<int>(this->allow_hard_edges));

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    this->FrameBufferObject->RenderQuad(0, width - 1, 0,  height - 1,
                                        this->KeyholeProgram->Program,
                                        this->KeyholeProgram->VAO);
    this->Pass1->Deactivate();

#ifdef VTK_KEYHOLE_PASS_DEBUG
    // Save the output of the first pass to a file for debugging
    pbo = this->Pass2->Download();

    openglRawData = new unsigned char[4 * width * height];

    status = pbo->Download2D(VTK_UNSIGNED_CHAR, openglRawData, this->dimensions, 4, increments);
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
    writer->SetFileName("KeyholePass12.png");
    writer->SetInputConnection(rgbatoRgb->GetOutputPort());
    writer->Write();

    importer->Delete();
    rgbatoRgb->Delete();
    writer->Delete();
#endif
    this->FrameBufferObject->UnBind();

    this->Pass2->Activate();

    this->Pass2->CopyToFrameBuffer(0, 0, width - 1, height - 1,
                                   0, 0, width, height,
                                   this->KeyholeProgram->Program,
                                   this->KeyholeProgram->VAO);
    this->Pass2->Deactivate();

	this->ForegroundTextureObject->Deactivate();
	this->MaskTextureObject->Deactivate();
	this->ForegroundGradientTextureObject->Deactivate();
  }
  else
  {
    vtkWarningMacro( << "no delegate.");
  }

  vtkOpenGLCheckErrorMacro("Failed after Render");
}

//------------------------------------------------------------------------------------------------------------
// Description:
// Compute gradient texture of the foreground and save it to a texture object
void vtkKeyholePass::GetForegroudGradient(vtkRenderer* r)
{
  vtkOpenGLRenderWindow* renwin = vtkOpenGLRenderWindow::SafeDownCast(r->GetRenderWindow());
  int* size = renwin->GetSize();
  int width = size[0];
  int height  = size[1];

  /*const int extraPixels = 1; // one on each side
  int w = width+2*extraPixels;
  int h = height+2*extraPixels;*/

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

  this->FrameBufferObject->SetNumberOfRenderTargets(2);
  this->FrameBufferObject->SetColorBuffer(0, this->GX);
  this->FrameBufferObject->SetColorBuffer(1, this->GY);
  unsigned int indices[2] = {0, 1};
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

  this->ForegroundTextureObject->Activate();
  int sourceID = this->ForegroundTextureObject->GetTextureUnit();
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  this->GradientProgram1->Program->SetUniformi("source", sourceID);

  float fvalue = static_cast<float>(1.0 / width);
  this->GradientProgram1->Program->SetUniformf("stepSize", fvalue);

  this->FrameBufferObject->RenderQuad(0, width - 1, 0, height - 1,
                                      this->GradientProgram1->Program,
                                      this->GradientProgram1->VAO);
  this->ForegroundTextureObject->Deactivate();
#ifdef VTK_KEYHOLE_PASS_DEBUG
  // Save the output for debugging
  vtkPixelBufferObject* pbo = GX->Download();

  unsigned int dims[2] = {size[0], size[1]};

  unsigned char* openglRawData = new unsigned char[4 * dims[0]*dims[1]];
  vtkIdType incs[2];
  incs[0] = 0;
  incs[1] = 0;
  bool status = pbo->Download2D(VTK_UNSIGNED_CHAR, openglRawData, dims, 4, incs);
  assert("check" && status);
  pbo->Delete();

  vtkImageImport* importer = vtkImageImport::New();
  importer->CopyImportVoidPointer(openglRawData, 4 * dims[0]*dims[1]*sizeof(unsigned char));
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

  this->FrameBufferObject->UnBind();
  // Now bind foreground_grad_to to the FBO
  this->FrameBufferObject->SetNumberOfRenderTargets(1);
  this->FrameBufferObject->SetColorBuffer(0, ForegroundGradientTextureObject);
  this->FrameBufferObject->SetActiveBuffer(0);
  this->FrameBufferObject->Start(width, height, false);

  // Set the shader program for the first pass of GX and GY
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

  this->FrameBufferObject->RenderQuad(0, width - 1, 0, height - 1,
                                      this->KeyholeShader->Program,
                                      this->KeyholeShader->VAO);
  this->GX->Deactivate();
  this->GY->Deactivate();

#ifdef VTK_KEYHOLE_PASS_DEBUG
  // Save the output for debugging
  pbo = this->ForegroundGradientTextureObject->Download();

  openglRawData = new unsigned char[4 * dims[0]*dims[1]];
  status = pbo->Download2D(VTK_UNSIGNED_CHAR, openglRawData, dims, 4, incs);
  assert("check" && status);
  pbo->Delete();

  importer = vtkImageImport::New();
  importer->CopyImportVoidPointer(openglRawData, 4 * dims[0]*dims[1]*sizeof(unsigned char));
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
void vtkKeyholePass::UpdateTextureObject(vtkOpenGLRenderWindow *renwin)
{
	int vtktype = this->ForegroundPixelBufferObject->GetType();
	GLenum type = this->ForegroundTextureObject->GetDefaultDataType(vtktype);

	GLenum internalFormat = this->ForegroundTextureObject->GetInternalFormat(vtktype,
		this->components,
		false);
	GLenum format = this->ForegroundTextureObject->GetFormat(vtktype,
		this->components,
		false);

	int width = this->ForegroundTextureObject->GetWidth();
	int height = this->ForegroundTextureObject->GetHeight();

	renwin->ActivateTexture(this->ForegroundTextureObject);
	this->ForegroundTextureObject->Bind();

	this->ForegroundPixelBufferObject->Bind(vtkPixelBufferObject::UNPACKED_BUFFER);
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

	this->ForegroundPixelBufferObject->UnBind();
	this->ForegroundTextureObject->Deactivate();
}

//-----------------------------------------------------------------------------------------------------
// Description:
// Release graphics resources and ask components to release their own
// resource.
// \pre w_exist: w!=0
void vtkKeyholePass::ReleaseGraphicsResources(vtkWindow* w)
{

  assert("pre: w_exists" && w != NULL);

  vtkImageProcessingPass::ReleaseGraphicsResources(w);

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
    this->Pass1  = 0;
  }

  if (this->Pass2 != NULL)
  {
    this->Pass2->Delete();
    this->Pass2 = 0;
  }

  if (this->ForegroundPixelBufferObject != NULL)
  {
    this->ForegroundPixelBufferObject->Delete();
    this->ForegroundPixelBufferObject = 0;
  }

  if (this->ForegroundTextureObject != NULL)
  {
    this->ForegroundTextureObject->Delete();
    this->ForegroundTextureObject = 0;
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