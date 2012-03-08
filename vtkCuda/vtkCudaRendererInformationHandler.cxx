#include "vtkCudaRendererInformationHandler.h"

// std
#include <vector>
// cuda functions
#include "vector_functions.h"

// vtk base
#include "vtkObjectFactory.h"

// Renderer Information
#include "vtkCamera.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkPlane.h"
#include "vtkMatrix4x4.h"
#include "CUDA_renderAlgo.h"

// vtkCuda
#include "vtkCudaMemoryTexture.h"
vtkStandardNewMacro(vtkCudaRendererInformationHandler);

vtkCudaRendererInformationHandler::vtkCudaRendererInformationHandler()
{
    this->Renderer = NULL;
    this->RendererInfo.ActualResolution.x = this->RendererInfo.ActualResolution.y = 0;
	this->RendererInfo.NumberOfClippingPlanes = 0;

	this->RendererInfo.a = 0.0f;
	this->RendererInfo.b = 0.0f;
	this->RendererInfo.darkness = 0.0f;
	this->RendererInfo.computedShift = 1.0f;

	this->RendererInfo.gradShadeScale = 0.0f;
	this->RendererInfo.gradShadeShift = 1.0f;

	this->RendererInfo.depthShadeShift = 1.0f;
	this->RendererInfo.depthShadeScale = 0.0f;
	this->twoOldDepthShift = 1.0f;
	this->twoOldDepthScale = 0.0f;
	this->depthShadeDarkness = 0.0f;

	this->clipModified = 0;
	this->MemoryTexture = vtkCudaMemoryTexture::New();

    this->SetRenderOutputScaleFactor(1.0f);
}

vtkCudaRendererInformationHandler::~vtkCudaRendererInformationHandler()
{
    this->Renderer = NULL;
    this->MemoryTexture->Delete();
}


void vtkCudaRendererInformationHandler::SetRenderer(vtkRenderer* renderer)
{
    this->Renderer = renderer;
    this->Update();
}

void vtkCudaRendererInformationHandler::SetRenderOutputScaleFactor(float scaleFactor) 
{
    this->RenderOutputScaleFactor = (scaleFactor > 1.0) ? scaleFactor : 1.0;
    this->Update();
}

void vtkCudaRendererInformationHandler::SetGoochShadingConstants(float darkness, float a, float b){
	if(darkness >= 0.0f && darkness <= 1.0f ){
		a = a * b;
		this->RendererInfo.a = a;
		this->RendererInfo.b = b;
		double r = 1.0 / ( 1.0 - 1.0 / (1.0 + exp(a) ) );
		this->RendererInfo.darkness = -1.0f * darkness * r;
		this->RendererInfo.computedShift = r + (1.0 - darkness) * ( 1.0 - r );
	}
}

void vtkCudaRendererInformationHandler::SetGradientShadingConstants(float darkness){
	if(darkness >= 0.0f && darkness <= 1.0f ){
		this->RendererInfo.gradShadeScale = darkness;
		this->RendererInfo.gradShadeShift = 1.0 - darkness;
	}
}

void vtkCudaRendererInformationHandler::SetDepthShadingConstants(float darkness){
	if(darkness >= 0.0f && darkness <= 1.0f ){
		this->depthShadeDarkness = darkness;
	}
}

void vtkCudaRendererInformationHandler::Bind()
{
    this->MemoryTexture->BindTexture();
    this->MemoryTexture->BindBuffer();
    this->RendererInfo.OutputImage = (uchar4*)this->MemoryTexture->GetRenderDestination();
}

void vtkCudaRendererInformationHandler::Unbind()
{
    this->MemoryTexture->UnbindBuffer();
    this->MemoryTexture->UnbindTexture();
}

void vtkCudaRendererInformationHandler::Update()
{
    if (this->Renderer != NULL)
    {
        // Renderplane Update.
        vtkRenderWindow *renWin= this->Renderer->GetRenderWindow();
        int *size=renWin->GetSize();
        this->RendererInfo.ActualResolution.x = size[0];
        this->RendererInfo.ActualResolution.y = size[1];

        this->RendererInfo.Resolution.x = this->RendererInfo.ActualResolution.x / this->RenderOutputScaleFactor;
        this->RendererInfo.Resolution.y = this->RendererInfo.ActualResolution.y / this->RenderOutputScaleFactor;

		//make it such that every thread fits within the solid for optimal access coalescing
		this->RendererInfo.Resolution.x -= this->RendererInfo.Resolution.x % 16;
		this->RendererInfo.Resolution.y -= this->RendererInfo.Resolution.y % 16;
		if(this->RendererInfo.Resolution.y < 256) {
			this->RendererInfo.Resolution.y = 256;
		}

        this->MemoryTexture->SetSize(this->RendererInfo.Resolution.x, this->RendererInfo.Resolution.y);
        this->RendererInfo.OutputImage = (uchar4*)this->MemoryTexture->GetRenderDestination();

	}
}

void vtkCudaRendererInformationHandler::SetViewToVoxelsMatrix(vtkMatrix4x4* matrix){
	//load the original table
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			this->RendererInfo.ViewToVoxelsMatrix[i*4+j] = matrix->GetElement(i,j);
		}
	}
	
	//compute the obtimizations to measure the view via the x,y position of the pixel divided by the resolution
	this->RendererInfo.ViewToVoxelsMatrix[3] += this->RendererInfo.ViewToVoxelsMatrix[0] - this->RendererInfo.ViewToVoxelsMatrix[1];
	this->RendererInfo.ViewToVoxelsMatrix[7] += this->RendererInfo.ViewToVoxelsMatrix[4] - this->RendererInfo.ViewToVoxelsMatrix[5];
	this->RendererInfo.ViewToVoxelsMatrix[11] += this->RendererInfo.ViewToVoxelsMatrix[8] - this->RendererInfo.ViewToVoxelsMatrix[9];
	this->RendererInfo.ViewToVoxelsMatrix[15] += this->RendererInfo.ViewToVoxelsMatrix[12] - this->RendererInfo.ViewToVoxelsMatrix[13];

	this->RendererInfo.ViewToVoxelsMatrix[0] *= -2.0f;
	this->RendererInfo.ViewToVoxelsMatrix[4] *= -2.0f;
	this->RendererInfo.ViewToVoxelsMatrix[8] *= -2.0f;
	this->RendererInfo.ViewToVoxelsMatrix[12] *= -2.0f;

	this->RendererInfo.ViewToVoxelsMatrix[1] *= 2.0f;
	this->RendererInfo.ViewToVoxelsMatrix[5] *= 2.0f;
	this->RendererInfo.ViewToVoxelsMatrix[9] *= 2.0f;
	this->RendererInfo.ViewToVoxelsMatrix[13] *= 2.0f;

}

void vtkCudaRendererInformationHandler::SetWorldToVoxelsMatrix(vtkMatrix4x4* matrix){
	this->clipModified = 0;
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			this->WorldToVoxelsMatrix[i*4+j] = matrix->GetElement(i,j);
		}
	}
}

void vtkCudaRendererInformationHandler::SetVoxelsToWorldMatrix(vtkMatrix4x4* matrix){
	this->clipModified = 0;
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			this->VoxelsToWorldMatrix[i*4+j] = matrix->GetElement(i,j);
		}
	}
}

void vtkCudaRendererInformationHandler::HoneDepthShadingConstants(vtkMatrix4x4* viewToVoxelsMatrix, vtkMatrix4x4* voxelsToViewMatrix, const float* extent){

	float inPoint[4] = {0.0f, 0.0f, 0.0f, 1.0f};
	int minIndex = -1;
	float minValue = 0.0f;
	float minViewX;
	float minViewY;
	bool minSet = false;

	int maxIndex = -1;
	float maxValue = 0.0f;
	float maxViewX;
	float maxViewY;
	bool maxSet = false;

	//find the minimum and maximum distance to a corner
	for(int i = 0; i < 8; i++){
		inPoint[0] = i & 1 ? extent[0] : extent[1];
		inPoint[1] = i & 2 ? extent[2] : extent[3];
		inPoint[2] = i & 4 ? extent[4] : extent[5];
		inPoint[3] = 1.0f;
		voxelsToViewMatrix->MultiplyPoint(inPoint, inPoint);

		//grab Z value and save if it is the min or max
		inPoint[2] /= inPoint[3];
		if( minIndex == -1 || inPoint[2] <= minValue ){
			minValue = inPoint[2];
			minIndex = i;
			minViewX = inPoint[0] /= inPoint[3];
			minViewY = inPoint[1] /= inPoint[3];if(inPoint[2] >= 0.0f && inPoint[2] <= 1.0f){
				minSet = true;
			}else{
				minSet = false;
			}
		}
		if( maxIndex == -1 || inPoint[2] >= maxValue ){
			maxValue = inPoint[2];
			maxIndex = i;
			maxViewX = inPoint[0] /= inPoint[3];
			maxViewY = inPoint[1] /= inPoint[3];
			if(inPoint[2] >= 0.0f && inPoint[2] <= 1.0f){
				maxSet = true;
			}else{
				maxSet = false;
			}
		}
	}

	//calculate that distance in voxel space
	float shortDistance = 0.0f;
	float longDistance = 1.0f;
	float refPoint[3];
	if( minSet ){
		refPoint[0] = minIndex & 1 ? extent[0] : extent[1];
		refPoint[1] = minIndex & 2 ? extent[2] : extent[3];
		refPoint[2] = minIndex & 4 ? extent[4] : extent[5];

		inPoint[0] = minViewX;
		inPoint[1] = minViewY;
		inPoint[2] = 0;
		inPoint[3] = 1.0f;
		viewToVoxelsMatrix->MultiplyPoint(inPoint, inPoint);
		refPoint[0] -= inPoint[0] / inPoint[3];
		refPoint[1] -= inPoint[1] / inPoint[3];
		refPoint[2] -= inPoint[2] / inPoint[3];

		shortDistance = sqrt( refPoint[0]*refPoint[0] + refPoint[1]*refPoint[1] + refPoint[2]*refPoint[2] );
	}

	if( maxSet ){
		refPoint[0] = maxIndex & 1 ? extent[0] : extent[1];
		refPoint[1] = maxIndex & 2 ? extent[2] : extent[3];
		refPoint[2] = maxIndex & 4 ? extent[4] : extent[5];

		inPoint[0] = maxViewX;
		inPoint[1] = maxViewY;
		inPoint[2] = 0;
		inPoint[3] = 1.0f;
		viewToVoxelsMatrix->MultiplyPoint(inPoint, inPoint);
		refPoint[0] -= inPoint[0] / inPoint[3];
		refPoint[1] -= inPoint[1] / inPoint[3];
		refPoint[2] -= inPoint[2] / inPoint[3];

		longDistance = sqrt( refPoint[0]*refPoint[0] + refPoint[1]*refPoint[1] + refPoint[2]*refPoint[2] );
	}

	//calculate modified shading constants
	if(maxSet){
		float temp = this->RendererInfo.depthShadeShift;
		this->RendererInfo.depthShadeShift = (this->twoOldDepthShift + this->RendererInfo.depthShadeShift + 1.0f + this->depthShadeDarkness * shortDistance / (longDistance - shortDistance)) * (1.0f/3.0f);
		this->twoOldDepthShift = temp;
		temp = this->RendererInfo.depthShadeScale;
		this->RendererInfo.depthShadeScale = (this->twoOldDepthScale + this->RendererInfo.depthShadeScale - this->depthShadeDarkness / (longDistance - shortDistance)) * (1.0f/3.0f);	
		this->twoOldDepthScale = temp;
	}else{
		float temp = this->RendererInfo.depthShadeShift;
		this->RendererInfo.depthShadeShift = (1.0f + this->twoOldDepthScale + this->RendererInfo.depthShadeScale) * (1.0f/3.0f);
		this->twoOldDepthShift = temp;
		temp = this->RendererInfo.depthShadeScale;
		this->RendererInfo.depthShadeScale = (this->twoOldDepthScale + this->RendererInfo.depthShadeScale) * (1.0f/3.0f);
		this->twoOldDepthScale = temp;
	}
}

void vtkCudaRendererInformationHandler::SetClippingPlanes(vtkPlaneCollection* planes){

	int numberOfPlanes = 0;
	if(planes){
		numberOfPlanes = planes->GetNumberOfItems();
		if(planes->GetMTime() == this->clipModified){
			return;
		}else{
			this->clipModified = planes->GetMTime();
		}
	}
	double worldNormal[3];
	double worldOrigin[3];
	double volumeOrigin[4];

	//set the number of planes in the information sent to CUDA
	this->RendererInfo.NumberOfClippingPlanes = numberOfPlanes;

	//load the planes into the local buffer and then into the CUDA buffer, providing the required pointer at the end
	float* clippingPlane = this->RendererInfo.ClippingPlanes;
	for(int i = 0; i < numberOfPlanes; i++){
		vtkPlane* onePlane = planes->GetItem(i);
		
		onePlane->GetNormal(worldNormal);
		onePlane->GetOrigin(worldOrigin);

		clippingPlane[4*i] = worldNormal[0]*VoxelsToWorldMatrix[0]  + worldNormal[1]*VoxelsToWorldMatrix[4]  + worldNormal[2]*VoxelsToWorldMatrix[8];
		clippingPlane[4*i+1] = worldNormal[0]*VoxelsToWorldMatrix[1]  + worldNormal[1]*VoxelsToWorldMatrix[5]  + worldNormal[2]*VoxelsToWorldMatrix[9];
		clippingPlane[4*i+2] = worldNormal[0]*VoxelsToWorldMatrix[2]  + worldNormal[1]*VoxelsToWorldMatrix[6]  + worldNormal[2]*VoxelsToWorldMatrix[10];

		volumeOrigin[0] = worldOrigin[0]*WorldToVoxelsMatrix[0]  + worldOrigin[1]*WorldToVoxelsMatrix[1]  + worldOrigin[2]*WorldToVoxelsMatrix[2]  + WorldToVoxelsMatrix[3];
		volumeOrigin[1] = worldOrigin[0]*WorldToVoxelsMatrix[4]  + worldOrigin[1]*WorldToVoxelsMatrix[5]  + worldOrigin[2]*WorldToVoxelsMatrix[6]  + WorldToVoxelsMatrix[7];
		volumeOrigin[2] = worldOrigin[0]*WorldToVoxelsMatrix[8]  + worldOrigin[1]*WorldToVoxelsMatrix[9]  + worldOrigin[2]*WorldToVoxelsMatrix[10] + WorldToVoxelsMatrix[11];
		volumeOrigin[3] = worldOrigin[0]*WorldToVoxelsMatrix[12] + worldOrigin[1]*WorldToVoxelsMatrix[13] + worldOrigin[2]*WorldToVoxelsMatrix[14] + WorldToVoxelsMatrix[15];
		if ( volumeOrigin[3] != 1.0 ) { volumeOrigin[0] /= volumeOrigin[3]; volumeOrigin[1] /= volumeOrigin[3]; volumeOrigin[2] /= volumeOrigin[3]; }

		clippingPlane[4*i+3] =	-(clippingPlane[4*i]*volumeOrigin[0] + clippingPlane[4*i+1]*volumeOrigin[1] + clippingPlane[4*i+2]*volumeOrigin[2]);
	}

}