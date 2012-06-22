#include "vtkCudaCT2USSimulation.h"
#include "vtkObjectFactory.h"

vtkStandardNewMacro(vtkCudaCT2USSimulation);

void vtkCudaCT2USSimulation::Reinitialize(int withData){
	if( withData && this->caster->GetInput() )
		this->SetInput((vtkImageData*) this->caster->GetInput());
}

void vtkCudaCT2USSimulation::Deinitialize(int withData){
	if(this->caster->GetInput()){
		this->ReserveGPU();
		CUDAsetup_unloadCTImage( this->GetStream() );
	}
}

void vtkCudaCT2USSimulation::SetInput( vtkImageData * i ){
	//load the input to a texture
	this->caster->SetInput(i);
	this->caster->Update();
	
	//get the volume size from the input
	double inputOrigin[3];
	double spacing[3];
	int inputExtent[6];
	int dims[3];
	i->GetOrigin(inputOrigin);
	i->GetExtent(inputExtent);
	i->GetSpacing(spacing);
	i->GetDimensions(dims);

		//set the volume dimensions
	this->information.VolumeSize.x = dims[0];
	this->information.VolumeSize.y = dims[1];
	this->information.VolumeSize.z = dims[2];

	//get the spacing information from the input
	this->information.spacing.x = 0.5 / spacing[0];
	this->information.spacing.y = 0.5 / spacing[1];
	this->information.spacing.z = 0.5 / spacing[2];

	// Compute the origin of the extent the volume origin is at voxel (0,0,0)
	// but we want to consider (0,0,0) in voxels to be at
	// (inputExtent[0], inputExtent[2], inputExtent[4]).
	double extentOrigin[3];
	extentOrigin[0] = inputOrigin[0] + inputExtent[0]*spacing[0];
	extentOrigin[1] = inputOrigin[1] + inputExtent[2]*spacing[1];
	extentOrigin[2] = inputOrigin[2] + inputExtent[4]*spacing[2];

	// Create a transform that will account for the scaling and translation of
	// the scalar data. The is the volume to voxels matrix.
	vtkTransform* VoxelsTransform = vtkTransform::New();
	VoxelsTransform->Identity();
	VoxelsTransform->Translate( extentOrigin[0], extentOrigin[1], extentOrigin[2] );
	VoxelsTransform->Scale( spacing[0], spacing[1], spacing[2] );
	
	// Now we actually have the world to voxels matrix - copy it out
	vtkMatrix4x4* WorldToVoxelsMatrix = vtkMatrix4x4::New();
	WorldToVoxelsMatrix->DeepCopy( VoxelsTransform->GetMatrix() );
	WorldToVoxelsMatrix->Invert();

	//output the CT location information to the information holder
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			this->information.WorldToVolume[i*4+j] = WorldToVoxelsMatrix->GetElement(i,j);
		}
	}

	//load the input volume into the CUDA kernel
	this->ReserveGPU();
	CUDAsetup_loadCTImage((float*) this->caster->GetOutput()->GetScalarPointer(),this->information,
		this->GetStream() );

}

void vtkCudaCT2USSimulation::SetTransform( vtkTransform * t ){
	this->usTransform = t;
	this->Modified();
}

#include "vtkTimerLog.h"

void vtkCudaCT2USSimulation::Update(){

	//if we are missing either input or transform, do not update
	if( !this->caster->GetInput() || !this->usTransform ) return;

	vtkTimerLog* timer = vtkTimerLog::New();
	timer->StartTimer();

	//output the ultrasound location information to the information holder
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			this->information.UltraSoundToWorld[i*4+j] = this->usTransform->GetMatrix()->GetElement(i,j);
		}
	}

	//run the algorithm
	this->ReserveGPU();
	CUDAalgo_simulateUltraSound((float*) this->densOutput->GetScalarPointer(),
		(float*) this->transOutput->GetScalarPointer(),
		(float*) this->reflOutput->GetScalarPointer(),
		(unsigned char*) this->usOutput->GetScalarPointer(),
		this->information, this->GetStream() );
	this->CallSyncThreads();

	timer->StopTimer();
	//std::cout << timer->GetElapsedTime() << std::endl;
	timer->Delete();

}

vtkImageData* vtkCudaCT2USSimulation::GetOutput(){
	return this->usOutput;
}

vtkImageData* vtkCudaCT2USSimulation::GetOutput(int i){
	switch(i){
		case 0: return this->usOutput;
		case 1: return this->transOutput;
		case 2: return this->reflOutput;
		case 3: return this->densOutput;
		default: return 0;
	}
}

void vtkCudaCT2USSimulation::SetOutputResolution(int x, int y, int z){

	//if we are 2D, treat us as such (make sure z is still depth)
	if( z == 1){
		this->information.Resolution.x = x;
		this->information.Resolution.y = z;
		this->information.Resolution.z = y;
	}else{
		this->information.Resolution.x = x;
		this->information.Resolution.y = y;
		this->information.Resolution.z = z;
	}
	
	//create new output image buffers
	if( !this->densOutput){
		this->densOutput = vtkImageData::New();
		this->densOutput->Register( this );
	}
	this->densOutput->SetNumberOfScalarComponents(1);
	this->densOutput->SetScalarTypeToFloat();
	this->densOutput->SetExtent(0,x-1,
								0,y-1,
								0,z-1);
	this->densOutput->SetOrigin(0,0,0);
	this->densOutput->SetSpacing(1.0,1.0,1.0);
	this->densOutput->Update();
	this->densOutput->AllocateScalars();
	
	if( !this->transOutput){
		this->transOutput = vtkImageData::New();
		this->transOutput->Register( this );
	}
	this->transOutput->SetNumberOfScalarComponents(1);
	this->transOutput->SetScalarTypeToFloat();
	this->transOutput->SetExtent(0,x-1,
								 0,y-1,
								 0,z-1);
	this->transOutput->SetOrigin(0,0,0);
	this->transOutput->SetSpacing(1.0,1.0,1.0);
	this->transOutput->Update();
	this->transOutput->AllocateScalars();
	
	if( !this->reflOutput){
		this->reflOutput = vtkImageData::New();
		this->reflOutput->Register( this );
	}
	this->reflOutput->SetNumberOfScalarComponents(1);
	this->reflOutput->SetScalarTypeToFloat();
	this->reflOutput->SetExtent(0,x-1,
								0,y-1,
								0,z-1);
	this->reflOutput->SetOrigin(0,0,0);
	this->reflOutput->SetSpacing(1.0,1.0,1.0);
	this->reflOutput->Update();
	this->reflOutput->AllocateScalars();
	
	//create a new simulated image
	if( !this->usOutput){
		this->usOutput = vtkImageData::New();
		this->usOutput->Register( this );
	}
	this->usOutput->SetNumberOfScalarComponents(3);
	this->usOutput->SetScalarTypeToUnsignedChar();
	this->usOutput->SetExtent(0,x-1,
							  0,y-1,
							  0,z-1);
	this->usOutput->SetOrigin(0,0,0);
	this->usOutput->SetSpacing(1.0,1.0,1.0);
	this->usOutput->Update();
	this->usOutput->AllocateScalars();
	
}

void vtkCudaCT2USSimulation::SetLogarithmicScaleFactor(float factor){
	this->information.a = factor;
}

void vtkCudaCT2USSimulation::SetTotalReflectionThreshold(float threshold){
	this->information.reflectionThreshold = threshold;
}

void vtkCudaCT2USSimulation::SetLinearCombinationAlpha(float a){
	this->information.alpha = a;
}

void vtkCudaCT2USSimulation::SetLinearCombinationBeta(float b){
	this->information.beta = b;
}

void vtkCudaCT2USSimulation::SetLinearCombinationBias(float bias){
	this->information.bias = bias;
}

void vtkCudaCT2USSimulation::SetProbeWidth(float xWidth, float yWidth){
	this->information.probeWidth.x = xWidth;
	this->information.probeWidth.y = yWidth;
}

void vtkCudaCT2USSimulation::SetFanAngle(float xAngle, float yAngle){
	this->information.fanAngle.x = xAngle * 3.1415926 / 180.0;
	this->information.fanAngle.y = yAngle * 3.1415926 / 180.0;
}

void vtkCudaCT2USSimulation::SetNearClippingDepth(float depth){
	this->information.StartDepth = depth;
}

void vtkCudaCT2USSimulation::SetFarClippingDepth(float depth){
	this->information.EndDepth = depth;
}

void vtkCudaCT2USSimulation::SetDensityScaleModel(float scale, float offset){
	this->information.hounsfieldScale = scale;
	this->information.hounsfieldOffset = offset;
}

vtkCudaCT2USSimulation::vtkCudaCT2USSimulation(){
	this->usOutput = 0;
	this->densOutput = 0;
	this->transOutput = 0;
	this->reflOutput = 0;
	this->usTransform = 0;
	this->information.a = 1.0f;
	this->alpha = 0.5f;
	this->beta = 0.5f;
	this->bias = 0.0f;
	this->information.StartDepth = 0.0f;
	this->information.EndDepth = 0.0f;
	this->information.VolumeSize.x = this->information.VolumeSize.y = this->information.VolumeSize.z = 0;
	this->information.Resolution.x = this->information.Resolution.y = this->information.Resolution.z = 0;
	this->information.probeWidth.x = 0.0f;
	this->information.probeWidth.y = 0.0f;
	this->information.fanAngle.x = 0.0f;
	this->information.fanAngle.y = 0.0f;
	this->information.reflectionThreshold = 1000000.0;
	this->information.hounsfieldScale = 1.0;
	this->information.hounsfieldOffset = -1024.0;

	this->caster = vtkImageCast::New();
	this->caster->SetOutputScalarTypeToFloat();
	this->caster->ClampOverflowOn();
}

vtkCudaCT2USSimulation::~vtkCudaCT2USSimulation(){
	if(this->usOutput){
		this->usOutput->Delete();
		this->transOutput->Delete();
		this->reflOutput->Delete();
		this->densOutput->Delete();
	}
	this->caster->Delete();
}
