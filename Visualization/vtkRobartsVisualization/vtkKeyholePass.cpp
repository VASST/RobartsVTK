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

#include "vtkKeyholePass.h"
#include "vtkObjectFactory.h"
#include <cassert>
#include "vtkRenderState.h"
#include "vtkRenderer.h"
#include "vtkFrameBufferObject.h"
#include "vtkTextureObject.h"
#include "vtkShaderProgram.h"
#include "vtkOpenGLShaderCache.h"
#include "vtkOpenGLRenderWindow.h"
#include "vtkOpenGLVertexArrayObject.h"
#include "vtkOpenGLRenderWindow.h"
#include "vtkTextureUnitManager.h"
#include "vtkOpenGLError.h"
#include "vtkOpenGLTexture.h"
#include "vtkOpenGLHelper.h"
#include "vtkProperty.h"


// To be able to dump intermediate passes into png images for debugging.
//#define VTK_KEYHOLE_PASS_DEBUG

#include "vtkPNGWriter.h"
#include "vtkImageImport.h"
#include "vtkPixelBufferObject.h"
#include "vtkPixelBufferObject.h"
#include "vtkImageExtractComponents.h"
#include "vtkCamera.h"
#include "vtkMath.h"

vtkStandardNewMacro(vtkKeyholePass);

//----------------------------------------------------------------------------------------------------
vtkKeyholePass::vtkKeyholePass()
{

  this->FrameBufferObject = 0;
  this->Pass1 = 0;
  this->Pass2 = 0;
  this->foreground_pbo = 0;
  this->mask_pbo = 0;
  this->foreground_to = 0;
  this->mask_to = 0;
  this->GX = 0;
  this->GY = 0;
  this->foreground_grad_to = 0;
  this->KeyholeProgram = 0;
  this->gradientProgram1 = 0;
  this->gradientProgram2 = 0;
  this->Supported = false;
  this->SupportProbed = false;
  this->allow_hard_edges = false;
  this->mask_img_available = false;
}

//----------------------------------------------------------------------------------------------------
vtkKeyholePass::~vtkKeyholePass()
{
  if(this->FrameBufferObject!=0)
  {
    vtkErrorMacro(<<"FrameBufferObject should have been deleted in ReleaseGraphicsResources().");
  }
  if(this->Pass1!=0)
  {
    vtkErrorMacro(<<"Pass1 should have been deleted in ReleaseGraphicsResources().");
  }
  if(this->Pass2!=0)
  {
    vtkErrorMacro(<<"Pass2 should have been deleted in ReleaseGraphicsResources().");
  }
}

//----------------------------------------------------------------------------------------------------
void vtkKeyholePass::PrintSelf(ostream &os, vtkIndent indent)
{
  vtkImageProcessingPass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------------------------------
// Description:
// Perform rendering according to a render state \p s.
// \pre s_exist s!=0
void vtkKeyholePass::Render(const vtkRenderState *s)
{
  assert("pre: s_exists" && s!=0);

  vtkOpenGLClearErrorMacro();

  this->NumberOfRenderedProps = 0;

  vtkRenderer *r = s->GetRenderer();
  vtkOpenGLRenderWindow *renwin = vtkOpenGLRenderWindow::SafeDownCast(r->GetRenderWindow());
  if(this->DelegatePass!=0)
  {

    if(!this->SupportProbed)
    {
      this->SupportProbed = true;

      // Test hardware support. If not supported, just render the delegate.
      bool supported = vtkFrameBufferObject::IsSupported( renwin );

      if(!supported)
      {
        vtkErrorMacro("FBOs are not supported by the context. Cannot render keyhole. ");
      }
      if(supported)
      {
        supported = vtkTextureObject::IsSupported( renwin );

        if(!supported)
        {
          vtkErrorMacro("Texture objects are not supported by the context. Cannot render keyhole. ");
        }
      }

      if(supported)
      {

        // FBO extension is supported. Is the specific FBO format supported?
        if(this->FrameBufferObject==0)
        {
          this->FrameBufferObject = vtkFrameBufferObject::New();
          this->FrameBufferObject->SetContext(renwin);
        }
        if(this->Pass1==0)
        {
          this->Pass1 = vtkTextureObject::New();
          this->Pass1->SetContext( renwin );
        }
        if( this->foreground_to == 0 )
        {
          this->foreground_to = vtkTextureObject::New();
          this->foreground_to->SetContext( renwin );
        }
        if( this->mask_to == 0 )
        {
          this->mask_to = vtkTextureObject::New();
          this->mask_to->SetContext( renwin );
        }
        if( this->foreground_pbo == 0)
        {
          this->foreground_pbo = vtkPixelBufferObject::New();
          this->foreground_pbo->SetContext( renwin );
        }
        if( this->mask_pbo == 0)
        {
          this->mask_pbo = vtkPixelBufferObject::New();
          this->mask_pbo->SetContext( renwin );
        }
        if( this->foreground_grad_to == 0)
        {
          this->foreground_grad_to = vtkTextureObject::New();
          this->foreground_grad_to->SetContext( renwin );
        }

        this->Pass1->Create2D(64, 64, 4, VTK_UNSIGNED_CHAR, false);
        this->FrameBufferObject->SetColorBuffer(0, this->Pass1);
        this->FrameBufferObject->SetNumberOfRenderTargets(1);
        this->FrameBufferObject->SetActiveBuffer(0);
        this->FrameBufferObject->SetDepthBufferNeeded( true );

        supported = this->FrameBufferObject->StartNonOrtho(64, 64, false);

        if(!supported)
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
    if(!this->Supported)
    {

      this->DelegatePass->Render(s);
      this->NumberOfRenderedProps += this->DelegatePass->GetNumberOfRenderedProps();

      return;
    }

    // Read image data into pixelBuffers. Do this for the background and the mask.
	vtkPropCollection *props = r->GetViewProps();
	int numActors = props->GetNumberOfItems();
	props->InitTraversal();
	
	vtkImageData *imgData;
	unsigned char *dataPtr;
	int img_size[3];
	vtkIdType increments[2];
	increments[0] = 0;
	increments[1] = 0;

	for(int i=0; i<numActors; i++){

		if( i == 0 ){
			// Discard the first actor
			props->GetNextProp();
		}
		else if( i == 1){

			vtkActor * foregroundActor = vtkActor::SafeDownCast(props->GetNextProp());// actors->GetNextActor();
			this->foregroundTex = foregroundActor->GetTexture();

				
			imgData = this->foregroundTex->GetInput();

			imgData->GetDimensions(img_size);
			this->components = imgData->GetNumberOfScalarComponents();
			dataPtr = (unsigned char*)imgData->GetScalarPointer();

			this->dimensions[0] = img_size[0]; 
			this->dimensions[1] = img_size[1];
			
			// Upload imagedata to pixel buffer object
			this->foreground_pbo->Upload2D(VTK_UNSIGNED_CHAR,
										   dataPtr, this->dimensions,
										   this->components, increments);
		}
		else if( i == 2){

			vtkActor * maskActor = vtkActor::SafeDownCast(props->GetNextProp());;// = actors->GetNextActor();
			this->maskTex = maskActor->GetTexture();


			vtkImageData *imgData = this->maskTex->GetInput();
			imgData->GetDimensions(img_size);
			this->components = imgData->GetNumberOfScalarComponents();
			dataPtr = (unsigned char*)imgData->GetScalarPointer();

			this->dimensions[0] = img_size[0]; 
			this->dimensions[1] = img_size[1];

			// Upload imagedata to pixel buffer object
			this->mask_pbo->Upload2D(VTK_UNSIGNED_CHAR,
										   dataPtr, this->dimensions,
										   this->components, increments);
		}
	}

	


	// Create foreground texture object .
	if(this->foreground_to->GetWidth() != static_cast<unsigned int>(this->dimensions[0]) ||
				this->foreground_to->GetHeight() != static_cast<unsigned int>(this->dimensions[1]))
	{
			  this->foreground_to->Create2D(this->dimensions[0], this->dimensions[1], this->components,
											this->foreground_pbo, false);
	}

	// Create mask texture object.
	if( this->mask_img_available ){
		if( this->mask_to->GetHeight() != static_cast<unsigned int>(this->dimensions[0]) ||
				this->mask_to->GetWidth() != static_cast<unsigned int>(this->dimensions[1]))

				this->mask_to->Create2D(this->dimensions[0], this->dimensions[1], this->components,
												this->mask_pbo, false);
	}

    // 1. Create a new render state with FBO.
    int width;
    int height;
    int size[2];
    s->GetWindowSize(size);
    width = size[0];
    height  = size[1];

    if( this->Pass1 == 0 )
    {
      this->Pass1 = vtkTextureObject::New();
      this->Pass1->SetContext( renwin );
    }

    if( this->FrameBufferObject==0)
    {
      this->FrameBufferObject = vtkFrameBufferObject::New();
      this->FrameBufferObject->SetContext( renwin );
    }
		
    // Remove background texture
    r->SetTexturedBackground( false );
    // Now set a black background.
    r->SetBackground(0, 0, 0);

    // First pass
    this->RenderDelegate(s, width, height, width, height, this->FrameBufferObject,
                         this->Pass1 );

#ifdef VTK_KEYHOLE_PASS_DEBUG
    // Save the output of the first pass to a file for debugging
	vtkPixelBufferObject *pbo = this->foreground_to->Download();

    unsigned char * openglRawData = new unsigned char[4*width*height];
    bool status = pbo->Download2D(VTK_UNSIGNED_CHAR, openglRawData, this->dimensions, 4, increments);
    assert("check" && status );
    pbo->Delete();

    vtkImageImport *importer = vtkImageImport::New();
    importer->CopyImportVoidPointer(openglRawData, 4*width*height*sizeof(unsigned char));
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents(4);
    importer->SetWholeExtent(0, width-1, 0, height-1, 0, 0);
    importer->SetDataExtentToWholeExtent();
    delete[] openglRawData;

    vtkImageExtractComponents *rgbatoRgb = vtkImageExtractComponents::New();
    rgbatoRgb->SetInputConnection( importer->GetOutputPort() );
    rgbatoRgb->SetComponents(0, 1, 2);

    vtkPNGWriter *writer = vtkPNGWriter::New();
    writer->SetFileName("KeyholePass1.png");
    writer->SetInputConnection( rgbatoRgb->GetOutputPort() );
    writer->Write();

    importer->Delete();
    rgbatoRgb->Delete();
    writer->Delete();
#endif

    // 2. Sobel pass and save to texture
    GetForegroudGradient( r );

    ///------------------------------------------

    // Same FBO, but new colour attachment (new TO).
    // Pass2 is our final composited scene
    if(this->Pass2==0)
    {
      this->Pass2 = vtkTextureObject::New();
      this->Pass2->SetContext( this->FrameBufferObject->GetContext() );
    }

    if( this->Pass2->GetWidth() != static_cast<unsigned int>(width) ||
        this->Pass2->GetHeight() != static_cast<unsigned int>(height))
    {
      this->Pass2->Create2D(static_cast<unsigned int>(width),
                            static_cast<unsigned int>(height), 4,
                            VTK_UNSIGNED_CHAR, false);
    }

    this->FrameBufferObject->SetColorBuffer(0, this->Pass2);
    this->FrameBufferObject->Start(width, height, false);

    // Now use the shader to do composting
    if( this->KeyholeProgram == 0)
    {
      this->KeyholeProgram = new vtkOpenGLHelper;
      // build the shader source code
      LoadShaders("./vtkKeyhole.fs", "./vtkKeyhole.vs");

      std::string GSSource;
      // compile and bind it if needed
      vtkShaderProgram *newShader =
        renwin->GetShaderCache()->ReadyShaderProgram(
          this->ver_shader_src.c_str(),
          this->frag_shader_src.c_str(),
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

    if(this->KeyholeProgram->Program->GetCompiled() != true)
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

	if( this->mask_img_available ){
		this->mask_to->Activate();
		int texture1 = this->mask_to->GetTextureUnit();
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		this->KeyholeProgram->Program->SetUniformi("_mask", texture1);
	}

    this->foreground_to->Activate();
    int texture2 = this->foreground_to->GetTextureUnit();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    this->KeyholeProgram->Program->SetUniformi("_foreground", texture2);

    this->foreground_grad_to->Activate();
    int texture3 = this->foreground_grad_to->GetTextureUnit();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    this->KeyholeProgram->Program->SetUniformi("_foreground_grad", texture3);

    this->KeyholeProgram->Program->SetUniformf("th", 0.04);
    this->KeyholeProgram->Program->SetUniformf("x0", static_cast<float>((this->x0*1.0)/width));
    this->KeyholeProgram->Program->SetUniformf("y0", static_cast<float>((this->y0*1.0)/height));
    this->KeyholeProgram->Program->SetUniformf("radius", static_cast<float>((this->radius*1.0)/width));
	this->KeyholeProgram->Program->SetUniformi("width", width);
	this->KeyholeProgram->Program->SetUniformi("height", height);
    this->KeyholeProgram->Program->SetUniformf("gamma",  this->gamma);
    this->KeyholeProgram->Program->SetUniformi("use_mask_texture", 0);
    this->KeyholeProgram->Program->SetUniformi("use_hard_edges", static_cast<int>(this->allow_hard_edges));

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    this->FrameBufferObject->RenderQuad(0, width-1, 0,  height-1,
                                        this->KeyholeProgram->Program,
                                        this->KeyholeProgram->VAO);
    this->Pass1->Deactivate();

#ifdef VTK_KEYHOLE_PASS_DEBUG
    // Save the output of the first pass to a file for debugging
    pbo = this->Pass2->Download();

    openglRawData = new unsigned char[4*width*height];

	status = pbo->Download2D(VTK_UNSIGNED_CHAR, openglRawData, this->dimensions, 4, increments);
    assert("check" && status );
    pbo->Delete();

    importer = vtkImageImport::New();
    importer->CopyImportVoidPointer(openglRawData, 4*width*height*sizeof(unsigned char));
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents(4);
    importer->SetWholeExtent(0, width-1, 0, height-1, 0, 0);
    importer->SetDataExtentToWholeExtent();
    delete[] openglRawData;

    rgbatoRgb = vtkImageExtractComponents::New();
    rgbatoRgb->SetInputConnection( importer->GetOutputPort() );
    rgbatoRgb->SetComponents(0, 1, 2);

    writer = vtkPNGWriter::New();
    writer->SetFileName("KeyholePass12.png");
    writer->SetInputConnection( rgbatoRgb->GetOutputPort() );
    writer->Write();

    importer->Delete();
    rgbatoRgb->Delete();
    writer->Delete();
#endif
    this->FrameBufferObject->UnBind();

    this->Pass2->Activate();

    this->Pass2->CopyToFrameBuffer(0, 0, width-1, height-1,
                                   0, 0, width, height,
                                   this->KeyholeProgram->Program,
                                   this->KeyholeProgram->VAO);
    this->Pass2->Deactivate();
  }
  else
  {
    vtkWarningMacro(<<"no delegate.");
  }

  vtkOpenGLCheckErrorMacro("Failed after Render");
}

//------------------------------------------------------------------------------------------------------------
// Description:
// Compute gradient texture of the foreground and save it to a texture object
void vtkKeyholePass::GetForegroudGradient(vtkRenderer *r)
{

  vtkOpenGLRenderWindow *renwin = vtkOpenGLRenderWindow::SafeDownCast(r->GetRenderWindow());
  int *size = renwin->GetSize();
  int width = size[0];
  int height  = size[1];

  /*const int extraPixels = 1; // one on each side
  int w = width+2*extraPixels;
  int h = height+2*extraPixels;*/

  // Create new TOs and set FBO color attachments
  if( this->GX == 0)
  {
    this->GX = vtkTextureObject::New();
    this->GX->SetContext( renwin );
  }
  if( this->GX->GetWidth() != static_cast<unsigned int>(width) ||
      this->GX->GetHeight() != static_cast<unsigned int>(height))
  {
    this->GX->Create2D( width, height, 4, VTK_UNSIGNED_CHAR, false);
  }

  if( this->GY == 0)
  {
    this->GY = vtkTextureObject::New();
    this->GY->SetContext( renwin );
  }
  if( this->GY->GetWidth() != static_cast<unsigned int>(width) ||
      this->GY->GetHeight() != static_cast<unsigned int>(height))
  {
    this->GY->Create2D( width, height, 4, VTK_UNSIGNED_CHAR, false);
  }

  this->FrameBufferObject->SetNumberOfRenderTargets( 2 );
  this->FrameBufferObject->SetColorBuffer(0, this->GX);
  this->FrameBufferObject->SetColorBuffer(1, this->GY);
  unsigned int indices[2] = {0, 1};
  this->FrameBufferObject->SetActiveBuffers(2,indices);
  this->FrameBufferObject->Start(width, height, false);

  // Set the shader program for the first pass of GX and GY
  if( this->gradientProgram1 == 0)
  {
    this->gradientProgram1 = new vtkOpenGLHelper;
    // build the shader source code
    LoadShaders("./gradientMagPass1.fs", "./vtkKeyhole.vs");

    std::string GSSource;
    // compile and bind it if needed
    vtkShaderProgram *newShader =
      renwin->GetShaderCache()->ReadyShaderProgram(
        this->ver_shader_src.c_str(),
        this->frag_shader_src.c_str(),
        GSSource.c_str());

    // if the shader changed reinitialize the VAO
    if (newShader != this->gradientProgram1->Program)
    {
      this->gradientProgram1->Program = newShader;
      this->gradientProgram1->VAO->ShaderProgramChanged(); // reset the VAO as the shader has changed
    }

    this->gradientProgram1->ShaderSourceTime.Modified();
  }
  else
  {
    renwin->GetShaderCache()->ReadyShaderProgram(this->gradientProgram1->Program);
  }

  if( this->gradientProgram1->Program->GetCompiled() != true)
  {
    vtkErrorMacro("Couldn't build the shader (Gradient 1) shader program. At this point it can be an error in the shader or a driver bug.");
    return;
  }

  this->foreground_to->Activate();
  int sourceID = this->foreground_to->GetTextureUnit();
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  this->gradientProgram1->Program->SetUniformi("source", sourceID);

  float fvalue = static_cast<float>(1.0/width);
  this->gradientProgram1->Program->SetUniformf("stepSize", fvalue);
  
  this->FrameBufferObject->RenderQuad(0, width-1, 0, height-1,
                                      this->gradientProgram1->Program,
                                      this->gradientProgram1->VAO);
  this->foreground_to->Deactivate();
#ifdef VTK_KEYHOLE_PASS_DEBUG
  // Save the output for debugging
  vtkPixelBufferObject *pbo = GX->Download();

  unsigned int dims[2] = {size[0], size[1]};

  unsigned char * openglRawData = new unsigned char[4*dims[0]*dims[1]];
  vtkIdType incs[2];
  incs[0] = 0;
  incs[1] = 0;
  bool status = pbo->Download2D(VTK_UNSIGNED_CHAR, openglRawData, dims, 4, incs);
  assert("check" && status );
  pbo->Delete();

  vtkImageImport *importer = vtkImageImport::New();
  importer->CopyImportVoidPointer(openglRawData, 4*dims[0]*dims[1]*sizeof(unsigned char));
  importer->SetDataScalarTypeToUnsignedChar();
  importer->SetNumberOfScalarComponents(4);
  importer->SetWholeExtent(0, dims[0]-1, 0, dims[1]-1, 0, 0);
  importer->SetDataExtentToWholeExtent();
  delete[] openglRawData;

  vtkImageExtractComponents *rgbatoRgb = vtkImageExtractComponents::New();
  rgbatoRgb->SetInputConnection( importer->GetOutputPort() );
  rgbatoRgb->SetComponents(0, 1, 2);

  vtkPNGWriter *writer = vtkPNGWriter::New();
  writer->SetFileName("Foreground_Grad_Pass1.png");
  writer->SetInputConnection( rgbatoRgb->GetOutputPort());
  writer->Write();

  importer->Delete();
  writer->Delete();
  rgbatoRgb->Delete();
#endif

  if(this->foreground_grad_to == 0)
  {
    this->foreground_grad_to = vtkTextureObject::New();
    this->foreground_grad_to->SetContext( renwin );
  }

  if( this->foreground_grad_to->GetWidth() != static_cast<unsigned int>(width) ||
      this->foreground_grad_to->GetHeight() != static_cast<unsigned int>(height))
  {
    this->foreground_grad_to->Create2D(static_cast<unsigned int>(width),
                                       static_cast<unsigned int>(height), 4,
                                       VTK_UNSIGNED_CHAR, false);
  }

  this->FrameBufferObject->UnBind();
  // Now bind foreground_grad_to to the FBO
  this->FrameBufferObject->SetNumberOfRenderTargets( 1 );
  this->FrameBufferObject->SetColorBuffer(0, foreground_grad_to);
  this->FrameBufferObject->SetActiveBuffer( 0 );
  this->FrameBufferObject->Start(width, height, false);

  // Set the shader program for the first pass of GX and GY
  if( this->gradientProgram2 == 0)
  {
    this->gradientProgram2 = new vtkOpenGLHelper;
    // build the shader source code
    LoadShaders("./gradientMagPass2.fs", "./vtkKeyhole.vs");

    std::string GSSource;
    // compile and bind it if needed
    vtkShaderProgram *newShader2 =
      renwin->GetShaderCache()->ReadyShaderProgram(
        this->ver_shader_src.c_str(),
        this->frag_shader_src.c_str(),
        GSSource.c_str());

    // if the shader changed reinitialize the VAO
    if (newShader2 != this->gradientProgram2->Program)
    {
      this->gradientProgram2->Program = newShader2;
      this->gradientProgram2->VAO->ShaderProgramChanged(); // reset the VAO as the shader has changed
    }

    this->gradientProgram2->ShaderSourceTime.Modified();
  }
  else
  {
    renwin->GetShaderCache()->ReadyShaderProgram(this->gradientProgram2->Program);
  }

  if( this->gradientProgram2->Program->GetCompiled() != true)
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

  this->gradientProgram2->Program->SetUniformi("gx1", id0);
  this->gradientProgram2->Program->SetUniformi("gy1", id1);

  fvalue = static_cast<float>(1.0/height);
  this->gradientProgram2->Program->SetUniformf("stepSize", fvalue);

  this->FrameBufferObject->RenderQuad(0, width-1, 0, height-1,
                                      this->gradientProgram2->Program,
                                      this->gradientProgram2->VAO);
  this->GX->Deactivate();
  this->GY->Deactivate();

#ifdef VTK_KEYHOLE_PASS_DEBUG
  // Save the output for debugging
  pbo = this->foreground_grad_to->Download();

  openglRawData = new unsigned char[4*dims[0]*dims[1]];
  status = pbo->Download2D(VTK_UNSIGNED_CHAR, openglRawData, dims, 4, incs);
  assert("check" && status );
  pbo->Delete();

  importer = vtkImageImport::New();
  importer->CopyImportVoidPointer(openglRawData, 4*dims[0]*dims[1]*sizeof(unsigned char));
  importer->SetDataScalarTypeToUnsignedChar();
  importer->SetNumberOfScalarComponents(4);
  importer->SetWholeExtent(0, dims[0]-1, 0, dims[1]-1, 0, 0);
  importer->SetDataExtentToWholeExtent();
  delete[] openglRawData;

  rgbatoRgb = vtkImageExtractComponents::New();
  rgbatoRgb->SetInputConnection( importer->GetOutputPort() );
  rgbatoRgb->SetComponents(0, 1, 2);

  writer = vtkPNGWriter::New();
  writer->SetFileName("Foreground_Grad_Pass2.png");
  writer->SetInputConnection( rgbatoRgb->GetOutputPort());
  writer->Write();

  importer->Delete();
  writer->Delete();
  rgbatoRgb->Delete();
#endif

}

//-----------------------------------------------------------------------------------------------------
// Description:
// Release graphics resources and ask components to release their own
// resource.
// \pre w_exist: w!=0
void vtkKeyholePass::ReleaseGraphicsResources(vtkWindow *w)
{

  assert("pre: w_exists" && w!=0);

  vtkImageProcessingPass::ReleaseGraphicsResources( w );

  if(this->KeyholeProgram != 0)
  {
    this->KeyholeProgram->Program->ReleaseGraphicsResources( w );
  }

  if(this->FrameBufferObject != 0 )
  {
    this->FrameBufferObject->Delete();
    this->FrameBufferObject = 0;
  }

  if(this->Pass1 != 0)
  {
    this->Pass1->Delete();
    this->Pass1  = 0;
  }

  if(this->Pass2 != 0)
  {
    this->Pass2->Delete();
    this->Pass2 = 0;
  }

  if( this->foreground_pbo != 0)
  {
    this->foreground_pbo->Delete();
    this->foreground_pbo = 0;
  }

  if( this->foreground_to != 0)
  {
    this->foreground_to->Delete();
    this->foreground_to = 0;
  }

  if( this->GX != 0 )
  {
    this->GX->Delete();
    this->GX = 0;
  }

  if( this->GY != 0 )
  {
    this->GY->Delete();
    this->GY = 0;
  }

  if( this->foreground_grad_to != 0)
  {
    this->foreground_grad_to->Delete();
    this->foreground_grad_to = 0;
  }

  if( this->mask_pbo != 0)
  {
    this->mask_pbo->Delete();
    this->mask_pbo = 0;
  }

  if( this->mask_to != 0)
  {
    this->mask_to->Delete();
    this->mask_to = 0;
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
    fShaderFile.open( fs_path );
    std::stringstream fShaderStream;
    fShaderStream << fShaderFile.rdbuf();

    fShaderFile.close();

    this->frag_shader_src = fShaderStream.str();

  }
  catch(std::ifstream::failure e)
  {
    std::cerr << "ERROR:: FS FILE READING ERROR. " << std::endl;
  }

  // Read Vertex Shader File
  try
  {
    // Open file
    vShaderFile.open( vs_path );
    std::stringstream vShaderStream;
    vShaderStream << vShaderFile.rdbuf();

    vShaderFile.close();

    this->ver_shader_src = vShaderStream.str();

  }
  catch(std::ifstream::failure e)
  {
    std::cerr << "ERROR:: VS FILE READING ERROR. " << std::endl;
  }

  return;
}