#include "vtkCudaRendererInformationHandler.h"

#include <vector>
#include "vector_functions.h"

#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"

// vtk base
#include "vtkObjectFactory.h"

// Renderer Information
#include "vtkCamera.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkPlane.h"
#include "vtkMatrix4x4.h"

vtkStandardNewMacro(vtkCudaRendererInformationHandler);

vtkCudaRendererInformationHandler::vtkCudaRendererInformationHandler()
{
	this->Renderer = 0;
	this->RendererInfo.actualResolution.x = this->RendererInfo.actualResolution.y = 0;
	this->RendererInfo.NumberOfClippingPlanes = 0;
	this->RendererInfo.NumberOfKeyholePlanes = 0;

	SetCelShadingConstants( 0.70f, 17.3f, 0.2629f );
	SetGradientShadingConstants(0.605f);
	
	this->ZBuffer = 0;

	this->clipModified = 0;

	
}

vtkRenderer* vtkCudaRendererInformationHandler::GetRenderer(){
	return this->Renderer;
}

void vtkCudaRendererInformationHandler::SetRenderer(vtkRenderer* renderer){
	this->Renderer = renderer;
	this->Update();
}

void vtkCudaRendererInformationHandler::SetCelShadingConstants(float darkness, float a, float b){
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

void vtkCudaRendererInformationHandler::Update(){
	if (this->Renderer != 0)
	{
		// Renderplane Update.
		int *size = this->Renderer->GetSize();
		this->RendererInfo.actualResolution.x = size[0];
		this->RendererInfo.actualResolution.y = size[1];
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

	//if we don't have a good number of planes, act as if we have none
	if( numberOfPlanes != 6 ){
		this->RendererInfo.NumberOfClippingPlanes = 0;
		return;
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

		clippingPlane[4*i+3] = -(clippingPlane[4*i]*volumeOrigin[0] + clippingPlane[4*i+1]*volumeOrigin[1] + clippingPlane[4*i+2]*volumeOrigin[2]);
	}

}

void vtkCudaRendererInformationHandler::SetKeyholePlanes(vtkPlaneCollection* planes){

	int numberOfPlanes = 0;
	if(planes){
		numberOfPlanes = planes->GetNumberOfItems();
		if(planes->GetMTime() == this->clipModified){
			return;
		}else{
			this->clipModified = planes->GetMTime();
		}
	}
	
	//if we don't have a good number of planes, act as if we have none
	if( numberOfPlanes != 6 ){
		this->RendererInfo.NumberOfKeyholePlanes = 0;
		return;
	}

	double worldNormal[3];
	double worldOrigin[3];
	double volumeOrigin[4];

	//set the number of planes in the information sent to CUDA
	this->RendererInfo.NumberOfKeyholePlanes = numberOfPlanes;

	//load the planes into the local buffer and then into the CUDA buffer, providing the required pointer at the end
	float* keyholePlane = this->RendererInfo.KeyholePlanes;
	for(int i = 0; i < numberOfPlanes; i++){
		vtkPlane* onePlane = planes->GetItem(i);
		
		onePlane->GetNormal(worldNormal);
		onePlane->GetOrigin(worldOrigin);

		keyholePlane[4*i] = worldNormal[0]*VoxelsToWorldMatrix[0]  + worldNormal[1]*VoxelsToWorldMatrix[4]  + worldNormal[2]*VoxelsToWorldMatrix[8];
		keyholePlane[4*i+1] = worldNormal[0]*VoxelsToWorldMatrix[1]  + worldNormal[1]*VoxelsToWorldMatrix[5]  + worldNormal[2]*VoxelsToWorldMatrix[9];
		keyholePlane[4*i+2] = worldNormal[0]*VoxelsToWorldMatrix[2]  + worldNormal[1]*VoxelsToWorldMatrix[6]  + worldNormal[2]*VoxelsToWorldMatrix[10];

		volumeOrigin[0] = worldOrigin[0]*WorldToVoxelsMatrix[0]  + worldOrigin[1]*WorldToVoxelsMatrix[1]  + worldOrigin[2]*WorldToVoxelsMatrix[2]  + WorldToVoxelsMatrix[3];
		volumeOrigin[1] = worldOrigin[0]*WorldToVoxelsMatrix[4]  + worldOrigin[1]*WorldToVoxelsMatrix[5]  + worldOrigin[2]*WorldToVoxelsMatrix[6]  + WorldToVoxelsMatrix[7];
		volumeOrigin[2] = worldOrigin[0]*WorldToVoxelsMatrix[8]  + worldOrigin[1]*WorldToVoxelsMatrix[9]  + worldOrigin[2]*WorldToVoxelsMatrix[10] + WorldToVoxelsMatrix[11];
		volumeOrigin[3] = worldOrigin[0]*WorldToVoxelsMatrix[12] + worldOrigin[1]*WorldToVoxelsMatrix[13] + worldOrigin[2]*WorldToVoxelsMatrix[14] + WorldToVoxelsMatrix[15];
		if ( volumeOrigin[3] != 1.0 ) { volumeOrigin[0] /= volumeOrigin[3]; volumeOrigin[1] /= volumeOrigin[3]; volumeOrigin[2] /= volumeOrigin[3]; }

		keyholePlane[4*i+3] = -(keyholePlane[4*i]*volumeOrigin[0] + keyholePlane[4*i+1]*volumeOrigin[1] + keyholePlane[4*i+2]*volumeOrigin[2]);
	}

}


void vtkCudaRendererInformationHandler::LoadZBuffer(){

	//image origin in pixels
	int x1 = this->Renderer->GetOrigin()[0];
	int y1 = this->Renderer->GetOrigin()[1];
	int x2 = x1 + this->RendererInfo.actualResolution.x - 1;
	int y2 = y1 + this->RendererInfo.actualResolution.y - 1;
	
	//remove old zBuffer and get a new one
	if(this->ZBuffer) delete this->ZBuffer;
	this->ZBuffer = this->Renderer->GetRenderWindow()->GetZbufferData(x1,y1,x2,y2);
	CUDA_vtkCudaVolumeMapper_renderAlgo_loadZBuffer(this->ZBuffer, this->RendererInfo.actualResolution.x, this->RendererInfo.actualResolution.y );

}