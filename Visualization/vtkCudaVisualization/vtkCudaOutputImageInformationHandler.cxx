/** @file vtkCudaOutputImageInformationHandler.cxx
 *
 *  @brief An internal class for vtkCudaVolumeMapper which manages information regarding the image being outputted
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 28, 2011
 *
 */

// FIXED ORDER
#include "vtkgl.h"
#include "vtkOpenGLExtensionManager.h"
#include "cuda_runtime_api.h"
#include "cuda_gl_interop.h"
// END FIXED ORDER

#include "CUDA_containerOutputImageInformation.h"
#include "vector_functions.h"
#include "vtkCudaMemoryTexture.h"
#include "vtkCudaOutputImageInformationHandler.h"
#include "vtkImageData.h"
#include "vtkObjectFactory.h"
#include "vtkRayCastImageDisplayHelper.h"
#include "vtkRenderer.h"
#include "vtkSetGet.h"
#include "vtkVolume.h"


vtkStandardNewMacro(vtkCudaOutputImageInformationHandler);

vtkCudaOutputImageInformationHandler::vtkCudaOutputImageInformationHandler(){
  this->Renderer = 0;
  this->Displayer = vtkRayCastImageDisplayHelper::New();
  this->RenderOutputScaleFactor = 1.0f;
  this->OutputImageInfo.resolution.x = this->OutputImageInfo.resolution.y = 0;
  this->oldResolution.x = this->oldResolution.y = 0;
  this->OutputImageInfo.depthBuffer = this->OutputImageInfo.numSteps = 0;
  this->OutputImageInfo.maxDepthBuffer = this->OutputImageInfo.minDepthBuffer = 0;
  this->OutputImageInfo.excludeStart = this->OutputImageInfo.excludeEnd = 0;
  this->OutputImageInfo.rayIncX = this->OutputImageInfo.rayStartX = 0;
  this->OutputImageInfo.rayIncY = this->OutputImageInfo.rayStartY = 0;
  this->OutputImageInfo.rayIncZ = this->OutputImageInfo.rayStartZ = 0;
  this->OutputImageInfo.flipped = this->ImageFlipped = false;
  this->hostOutputImage = 0;
  this->deviceOutputImage = 0;
  this->OutputImageInfo.renderType = 1;
  this->oldRenderType = 1;
  this->Reinitialize();
  this->ImageTint.w = this->OutputImageInfo.tint.w = 0;
  this->ImageTint.x = this->OutputImageInfo.tint.x = 0;
  this->ImageTint.y = this->OutputImageInfo.tint.y = 0;
  this->ImageTint.z = this->OutputImageInfo.tint.z = 0;
}


vtkCudaOutputImageInformationHandler::~vtkCudaOutputImageInformationHandler(){
  this->Deinitialize();
  this->Displayer->Delete();
}

void vtkCudaOutputImageInformationHandler::Deinitialize(int withData){
  this->MemoryTexture->Delete();
  if(this->OutputImageInfo.numSteps) cudaFree(this->OutputImageInfo.numSteps);
  if(this->OutputImageInfo.excludeStart) cudaFree(this->OutputImageInfo.excludeStart);
  if(this->OutputImageInfo.excludeEnd) cudaFree(this->OutputImageInfo.excludeEnd);
  if(this->OutputImageInfo.depthBuffer) cudaFree(this->OutputImageInfo.depthBuffer);
  if(this->OutputImageInfo.maxDepthBuffer) cudaFree(this->OutputImageInfo.maxDepthBuffer);
  if(this->OutputImageInfo.minDepthBuffer) cudaFree(this->OutputImageInfo.minDepthBuffer);
  if(this->OutputImageInfo.rayIncX) cudaFree(this->OutputImageInfo.rayIncX);
  if(this->OutputImageInfo.rayIncY) cudaFree(this->OutputImageInfo.rayIncY);
  if(this->OutputImageInfo.rayIncZ) cudaFree(this->OutputImageInfo.rayIncZ);
  if(this->OutputImageInfo.rayStartX) cudaFree(this->OutputImageInfo.rayStartX);
  if(this->OutputImageInfo.rayStartY) cudaFree(this->OutputImageInfo.rayStartY);
  if(this->OutputImageInfo.rayStartZ) cudaFree(this->OutputImageInfo.rayStartZ);
  if(this->hostOutputImage) delete this->hostOutputImage;
  if(this->deviceOutputImage) cudaFree(this->deviceOutputImage);
  this->OutputImageInfo.resolution.x = this->OutputImageInfo.resolution.y = 0;
  this->oldResolution.x = this->oldResolution.y = 0;
  this->OutputImageInfo.depthBuffer = this->OutputImageInfo.numSteps = 0;
  this->OutputImageInfo.maxDepthBuffer = this->OutputImageInfo.minDepthBuffer = 0;
  this->OutputImageInfo.excludeStart = this->OutputImageInfo.excludeEnd = 0;
  this->OutputImageInfo.rayIncX = this->OutputImageInfo.rayStartX = 0;
  this->OutputImageInfo.rayIncY = this->OutputImageInfo.rayStartY = 0;
  this->OutputImageInfo.rayIncZ = this->OutputImageInfo.rayStartZ = 0;
  this->OutputImageInfo.flipped = this->ImageFlipped;
  this->hostOutputImage = 0;
  this->deviceOutputImage = 0;
}

void vtkCudaOutputImageInformationHandler::Reinitialize(int withData){
  this->MemoryTexture = vtkCudaMemoryTexture::New();
  this->MemoryTexture->ReplicateObject(this, withData);
}

void vtkCudaOutputImageInformationHandler::SetRenderOutputScaleFactor(float scaleFactor) {
  this->RenderOutputScaleFactor = (scaleFactor > 1.0) ? scaleFactor : 1.0;
  this->Update();
}

void vtkCudaOutputImageInformationHandler::SetRenderType(int t){
  if(t >= 0 && t < 2){
    this->oldRenderType = this->OutputImageInfo.renderType;
    this->OutputImageInfo.renderType = t;
    this->Update();
  }
}

const cudaOutputImageInformation& vtkCudaOutputImageInformationHandler::GetOutputImageInfo()
{
  return (this->OutputImageInfo);
}

vtkRenderer* vtkCudaOutputImageInformationHandler::GetRenderer(){
  return this->Renderer;
}

void vtkCudaOutputImageInformationHandler::SetRenderer(vtkRenderer* renderer){
  this->Renderer = renderer;
  this->Update();
}

vtkImageData* vtkCudaOutputImageInformationHandler::GetCurrentImageData(){
  return 0;
}

void vtkCudaOutputImageInformationHandler::Prepare(){
  this->OutputImageInfo.deviceOutputImage = this->deviceOutputImage;
  if(this->OutputImageInfo.renderType == 0){
    this->MemoryTexture->BindTexture();
    this->MemoryTexture->BindBuffer();
    this->OutputImageInfo.deviceOutputImage = (uchar4*) this->MemoryTexture->GetRenderDestination();
  }
}

void vtkCudaOutputImageInformationHandler::Display(vtkVolume* volume, vtkRenderer* renderer){

  this->ReserveGPU();
  cudaStreamSynchronize(*(this->GetStream()));

  //do the actual rendering
  if(this->OutputImageInfo.renderType == 0){

    // Enter 2D Mode
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glEnable( GL_TEXTURE_2D );
    glEnable( GL_SCISSOR_TEST );
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, 1.0, 1.0, 0.0,0.0,1.0);
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glPushMatrix();
    glLoadIdentity();

    // Actual rendering on the screen
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2i(0,0);  glVertex2d(0, 1);
    glTexCoord2i(1,0);  glVertex2d(1, 1);
    glTexCoord2i(1,1);  glVertex2d(1, 0);
    glTexCoord2i(0,1);  glVertex2d(0, 0);
    glEnd();
    this->MemoryTexture->UnbindBuffer();
    this->MemoryTexture->UnbindTexture();

    // Leave the 2D Mode again.
    this->ReserveGPU();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glPopAttrib();

  //if desired, render using the fulling compatible displayer tool
  }else if(this->OutputImageInfo.renderType == 1){
    cudaMemcpyAsync( this->hostOutputImage, this->deviceOutputImage, 4*sizeof(unsigned char)*this->OutputImageInfo.resolution.x*this->OutputImageInfo.resolution.y, cudaMemcpyDeviceToHost, *(this->GetStream()));
    int imageMemorySize[2];
    imageMemorySize[0] = this->OutputImageInfo.resolution.x;
    imageMemorySize[1] = this->OutputImageInfo.resolution.y;
    int imageOrigin[2] = {0,0};
    this->Displayer->RenderTexture(volume,renderer,imageMemorySize,imageMemorySize,imageMemorySize,imageOrigin,0.001,(unsigned char*) this->hostOutputImage);

  }else if(this->OutputImageInfo.renderType == 2){
    cudaMemcpyAsync( this->hostOutputImage, this->deviceOutputImage, 4*sizeof(unsigned char)*this->OutputImageInfo.resolution.x*this->OutputImageInfo.resolution.y, cudaMemcpyDeviceToHost, *(this->GetStream()));

  }else{
    //error
  }

  this->ReserveGPU();
  cudaStreamSynchronize(*(this->GetStream()));

}

void vtkCudaOutputImageInformationHandler::Update(){

  if (this->Renderer == 0) return;

  // Image size update.
  int *size = this->Renderer->GetSize();
  this->OutputImageInfo.resolution.x = size[0] / this->RenderOutputScaleFactor;
  this->OutputImageInfo.resolution.y = size[1] / this->RenderOutputScaleFactor;

  //make it such that every thread fits within the solid for optimal access coalescing
  this->OutputImageInfo.resolution.x += (this->OutputImageInfo.resolution.x % 16) ? 16-(this->OutputImageInfo.resolution.x % 16) : 0;
  this->OutputImageInfo.resolution.y += (this->OutputImageInfo.resolution.y % 16) ?16-(this->OutputImageInfo.resolution.y % 16): 0;
  if(this->OutputImageInfo.resolution.y < 256) this->OutputImageInfo.resolution.y = 256;
  if(this->OutputImageInfo.resolution.x < 256) this->OutputImageInfo.resolution.x = 256;

  //if our image size hasn't changed, we don't have to reallocate any buffers, so we can just leave
  if(this->OutputImageInfo.resolution.x == this->oldResolution.x && this->OutputImageInfo.resolution.y == this->oldResolution.y)
    return;

  //reset the values for the old resolution to the current (for the next update)
  this->oldResolution = this->OutputImageInfo.resolution;

  //update image flip value
  this->OutputImageInfo.flipped = this->ImageFlipped;

  //allocate the buffers used for intermediate output results in rendering
  this->ReserveGPU();
  if(this->OutputImageInfo.numSteps) cudaFree(this->OutputImageInfo.numSteps);
  cudaMalloc( (void**) &this->OutputImageInfo.numSteps, sizeof(float)*this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y);
  if(this->OutputImageInfo.excludeStart) cudaFree(this->OutputImageInfo.excludeStart);
  cudaMalloc( (void**) &this->OutputImageInfo.excludeStart, sizeof(float)*this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y);
  if(this->OutputImageInfo.excludeEnd) cudaFree(this->OutputImageInfo.excludeEnd);
  cudaMalloc( (void**) &this->OutputImageInfo.excludeEnd, sizeof(float)*this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y);

  if(this->OutputImageInfo.depthBuffer) cudaFree(this->OutputImageInfo.depthBuffer);
  cudaMalloc( (void**) &this->OutputImageInfo.depthBuffer, sizeof(float)*this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y);
  if(this->OutputImageInfo.maxDepthBuffer) cudaFree(this->OutputImageInfo.maxDepthBuffer);
  cudaMalloc( (void**) &this->OutputImageInfo.maxDepthBuffer, sizeof(float)*this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y);
  if(this->OutputImageInfo.minDepthBuffer) cudaFree(this->OutputImageInfo.minDepthBuffer);
  cudaMalloc( (void**) &this->OutputImageInfo.minDepthBuffer, sizeof(float)*this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y);
  if(this->OutputImageInfo.rayIncX) cudaFree(this->OutputImageInfo.rayIncX);
  cudaMalloc( (void**) &this->OutputImageInfo.rayIncX, sizeof(float)*this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y);
  if(this->OutputImageInfo.rayIncY) cudaFree(this->OutputImageInfo.rayIncY);
  cudaMalloc( (void**) &this->OutputImageInfo.rayIncY, sizeof(float)*this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y);
  if(this->OutputImageInfo.rayIncZ) cudaFree(this->OutputImageInfo.rayIncZ);
  cudaMalloc( (void**) &this->OutputImageInfo.rayIncZ, sizeof(float)*this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y);
  if(this->OutputImageInfo.rayStartX) cudaFree(this->OutputImageInfo.rayStartX);
  cudaMalloc( (void**) &this->OutputImageInfo.rayStartX, sizeof(float)*this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y);
  if(this->OutputImageInfo.rayStartY) cudaFree(this->OutputImageInfo.rayStartY);
  cudaMalloc( (void**) &this->OutputImageInfo.rayStartY, sizeof(float)*this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y);
  if(this->OutputImageInfo.rayStartZ) cudaFree(this->OutputImageInfo.rayStartZ);
  cudaMalloc( (void**) &this->OutputImageInfo.rayStartZ, sizeof(float)*this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y);

  //reallocate the render type specific buffers
  if(this->OutputImageInfo.renderType == 0){
    //recreate the device image buffer
    this->MemoryTexture->SetSize(this->OutputImageInfo.resolution.x, this->OutputImageInfo.resolution.y);
  }else{
    //allocate the buffers
    this->ReserveGPU();
    if(this->deviceOutputImage) cudaFree(this->deviceOutputImage);
    cudaMalloc( (void**) &this->deviceOutputImage, 4*sizeof(unsigned char)*this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y);
    if(this->hostOutputImage) delete this->hostOutputImage;
    this->hostOutputImage = new uchar4[this->OutputImageInfo.resolution.x * this->OutputImageInfo.resolution.y];
  }


}

void vtkCudaOutputImageInformationHandler::SetTint(unsigned char RGBA[4]){
  this->ImageTint.x = this->OutputImageInfo.tint.x = RGBA[0];
  this->ImageTint.y = this->OutputImageInfo.tint.y = RGBA[1];
  this->ImageTint.z = this->OutputImageInfo.tint.z = RGBA[2];
  this->ImageTint.w = this->OutputImageInfo.tint.w = RGBA[3];
}

void vtkCudaOutputImageInformationHandler::GetTint(unsigned char RGBA[4]){
  RGBA[0] = this->ImageTint.x;
  RGBA[1] = this->ImageTint.y;
  RGBA[2] = this->ImageTint.z;
  RGBA[3] = this->ImageTint.w;
}