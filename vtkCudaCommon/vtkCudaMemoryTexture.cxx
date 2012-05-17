#include "vtkCudaMemoryTexture.h"
#include "vtkObjectFactory.h"

// OpenGL
#include "vtkgl.h"
#include "vtkOpenGLExtensionManager.h"
// CUDA
#include "cuda_runtime_api.h"
#include "cuda_gl_interop.h"

#include "vtkImageData.h"

vtkStandardNewMacro(vtkCudaMemoryTexture);

vtkCudaMemoryTexture::vtkCudaMemoryTexture(){
	this->Initialize();
}

vtkCudaMemoryTexture::~vtkCudaMemoryTexture(){
	this->Deinitialize();
}

void vtkCudaMemoryTexture::Deinitialize(int withData){
	
	this->ReserveGPU();
	if(this->CudaOutputData) cudaFree( (void*) this->CudaOutputData );
	this->CudaOutputData = 0;

	if (this->TextureID == 0 || !glIsTexture(this->TextureID))
		glGenTextures(1, &this->TextureID);

	if (vtkCudaMemoryTexture::GLBufferObjectsAvailiable == true)
		if (this->BufferObjectID != 0 && vtkgl::IsBufferARB(this->BufferObjectID))
			vtkgl::DeleteBuffersARB(1, &this->BufferObjectID);
}

void vtkCudaMemoryTexture::Reinitialize(int withData){
	this->Initialize();
	if(!this->CudaOutputData){
		this->ReserveGPU();
		cudaMalloc( (void**) &this->CudaOutputData, sizeof(uchar4) * this->Width * this->Height );
		this->RebuildBuffer();
	}
}

bool  vtkCudaMemoryTexture::GLBufferObjectsAvailiable = false;

void vtkCudaMemoryTexture::Initialize()
{
	this->TextureID = 0;
	this->BufferObjectID = 0;
	this->Height = this->Width = 0;
	this->RenderDestination = NULL;
	this->CurrentRenderMode = RenderToMemory;

	this->CudaOutputData = 0;
	this->LocalOutputData = 0;

	if (vtkCudaMemoryTexture::GLBufferObjectsAvailiable == false)
	{
		// check for the RenderMode
		vtkOpenGLExtensionManager *extensions = vtkOpenGLExtensionManager::New();
		extensions->SetRenderWindow(NULL);
		if (extensions->ExtensionSupported("GL_ARB_vertex_buffer_object"))
		{
			extensions->LoadExtension("GL_ARB_vertex_buffer_object");
			vtkCudaMemoryTexture::GLBufferObjectsAvailiable = true;
			this->CurrentRenderMode = RenderToTexture;
		}
		extensions->Delete();
	}
}

void vtkCudaMemoryTexture::SetSize(unsigned int width, unsigned int height)
{
	if (width == this->Width && this->Height == height)
		return;
	else
	{

		this->Width = width;
		this->Height = height;
		
		this->ReserveGPU();
		if(this->CudaOutputData) cudaFree( (void*) this->CudaOutputData );
		if(this->LocalOutputData) delete this->LocalOutputData;

		// Allocate Memory
		cudaMalloc( (void**) &this->CudaOutputData, sizeof(uchar4) * this->Width * this->Height );
		this->LocalOutputData = new uchar4[this->Width * this->Height];

		this->RebuildBuffer();
	}
}
void vtkCudaMemoryTexture::RebuildBuffer()
{
	// TEXTURE CODE
	this->ReserveGPU();
	glEnable(GL_TEXTURE_2D);
	if (this->TextureID != 0 && glIsTexture(this->TextureID))
		glDeleteTextures(1, &this->TextureID);
	glGenTextures(1, &this->TextureID);
	glBindTexture(GL_TEXTURE_2D, this->TextureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, this->Width, this->Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*) this->LocalOutputData );
	glBindTexture(GL_TEXTURE_2D, 0);

	if (this->CurrentRenderMode == RenderToTexture)
	{
		// OpenGL Buffer Code
		this->ReserveGPU();
		if (this->BufferObjectID != 0 && vtkgl::IsBufferARB(this->BufferObjectID))
			vtkgl::DeleteBuffersARB(1, &this->BufferObjectID);
		vtkgl::GenBuffersARB(1, &this->BufferObjectID);
		vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, this->BufferObjectID);
		vtkgl::BufferDataARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, this->Width * Height * sizeof(uchar4), (void*) this->LocalOutputData, vtkgl::STREAM_COPY);
		vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, 0);
	}
}
void vtkCudaMemoryTexture::SetRenderMode(int mode)
{
	if (mode == RenderToTexture && vtkCudaMemoryTexture::GLBufferObjectsAvailiable){
		this->CurrentRenderMode = mode;
	}else{
		this->CurrentRenderMode = RenderToMemory;
	}
	this->RebuildBuffer();
}

void vtkCudaMemoryTexture::BindTexture()
{
	this->ReserveGPU();
	glPushAttrib(GL_ENABLE_BIT);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, this->TextureID);
}

void vtkCudaMemoryTexture::BindBuffer()
{
	if (this->CurrentRenderMode == RenderToTexture)
	{
		this->ReserveGPU();
		cudaGLRegisterBufferObject(this->BufferObjectID) ;
		cudaStreamSynchronize(*(this->GetStream()));

		vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, this->BufferObjectID);
		cudaGLMapBufferObject((void**)&this->RenderDestination, this->BufferObjectID);
		cudaStreamSynchronize(*(this->GetStream()));
	}
	else
	{
		this->RenderDestination = (unsigned char*) ( (void*) this->CudaOutputData );
	}
}

void vtkCudaMemoryTexture::UnbindBuffer()
{
	if (this->CurrentRenderMode == RenderToTexture)
	{
		this->ReserveGPU();
		cudaGLUnmapBufferObject(this->BufferObjectID);
		cudaStreamSynchronize(*(this->GetStream()));
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->Width, this->Height, GL_RGBA, GL_UNSIGNED_BYTE, (0));
		cudaGLUnregisterBufferObject(this->BufferObjectID) ;
		cudaStreamSynchronize(*(this->GetStream()));
		vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, 0);
	}
	else // (this->CurrentRenderMode == RenderToMemory)
	{
		this->ReserveGPU();
		cudaMemcpyAsync( this->LocalOutputData, this->CudaOutputData, sizeof(uchar4) * this->Width * this->Height, cudaMemcpyDeviceToHost, *(this->GetStream()) );
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->Width, this->Height, GL_RGBA, GL_UNSIGNED_BYTE, (void*) LocalOutputData );
	}
	this->RenderDestination = NULL;
}
void vtkCudaMemoryTexture::UnbindTexture()
{
	this->ReserveGPU();
	glPopAttrib();
}

bool vtkCudaMemoryTexture::CopyToVtkImageData(vtkImageData* data)
{
	// setting up the data type and size.
	data->SetScalarTypeToUnsignedChar();
	data->SetNumberOfScalarComponents(4);
	data->SetDimensions(this->Width, Height, 1);
	data->SetExtent(0, this->Width - 1, 
		0, this->Height - 1, 
		0, 1 - 1);
	data->SetNumberOfScalarComponents(4);
	data->AllocateScalars();
	
	this->ReserveGPU();
	cudaMemcpyAsync( (void*) data->GetScalarPointer(), (void*) this->CudaOutputData, sizeof(uchar4) * this->Width * this->Height, cudaMemcpyDeviceToHost, *(this->GetStream()));
	cudaStreamSynchronize( *(this->GetStream()) );

	return true;
}
