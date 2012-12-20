/** @file vtkCudaVolumeMapper.cxx
 *
 *  @brief Header file defining a volume mapper (ray caster) using CUDA kernels for parallel ray calculation
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
 *  @note First documented on March 29, 2011
 *
 */

#include "vtkCudaVolumeMapper.h"
#include "vtkObjectFactory.h"

#include "vtkVolume.h"
#include "vtkImageData.h"

#include "vtkCamera.h"
#include "vtkRenderer.h"

#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CUDA_containerOutputImageInformation.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"

vtkCudaVolumeMapper::vtkCudaVolumeMapper()
{
	this->VolumeInfoHandler = vtkCudaVolumeInformationHandler::New();
	this->VolumeInfoHandler->ReplicateObject(this);
	this->RendererInfoHandler = vtkCudaRendererInformationHandler::New();
	this->RendererInfoHandler->ReplicateObject(this);
	this->OutputInfoHandler = vtkCudaOutputImageInformationHandler::New();
	this->OutputInfoHandler->ReplicateObject(this);
	
	this->KeyholePlanes = NULL;
	erroredOut = false;

	this->ViewToVoxelsMatrix = vtkMatrix4x4::New();
	this->WorldToVoxelsMatrix = vtkMatrix4x4::New();
	this->PerspectiveTransform = vtkTransform::New();
	this->VoxelsTransform = vtkTransform::New();
	this->VoxelsToViewTransform = vtkTransform::New();
	this->NextVoxelsToViewTransform = vtkTransform::New();

	this->renModified = 0;
	this->volModified = 0;
	this->currFrame = 0;
	this->numFrames = 1;
	
	this->Reinitialize();
}

void vtkCudaVolumeMapper::Deinitialize(int withData){
	CUDA_vtkCudaVolumeMapper_renderAlgo_unloadrandomRayOffsets(this->GetStream());
}

void vtkCudaVolumeMapper::Reinitialize(int withData){
	this->VolumeInfoHandler->ReplicateObject(this, withData);
	this->RendererInfoHandler->ReplicateObject(this, withData);
	this->OutputInfoHandler->ReplicateObject(this, withData);

	//initialize the random ray denoising buffer
	float randomRayOffsets[256];
	randomRayOffsets[0] = 0.70554; 	randomRayOffsets[1] = 0.53342;
	randomRayOffsets[2] = 0.57951;	randomRayOffsets[3] = 0.28956;
	randomRayOffsets[4] = 0.30194;	randomRayOffsets[5] = 0.77474;
	randomRayOffsets[6] = 0.01401;	randomRayOffsets[7] = 0.76072;
	randomRayOffsets[8] = 0.81449;	randomRayOffsets[9] = 0.70903;
	randomRayOffsets[10] = 0.04535;	randomRayOffsets[11] = 0.41403;
	randomRayOffsets[12] = 0.86261;	randomRayOffsets[13] = 0.79048;
	randomRayOffsets[14] = 0.37353;	randomRayOffsets[15] = 0.96195;
	randomRayOffsets[16] = 0.87144;	randomRayOffsets[17] = 0.05623;
	randomRayOffsets[18] = 0.94955;	randomRayOffsets[19] = 0.36401;
	randomRayOffsets[20] = 0.52486;	randomRayOffsets[21] = 0.76711;
	randomRayOffsets[22] = 0.0535;	randomRayOffsets[23] = 0.59245;
	randomRayOffsets[24] = 0.4687;	randomRayOffsets[25] = 0.29816;
	randomRayOffsets[26] = 0.62269;	randomRayOffsets[27] = 0.64782;
	randomRayOffsets[28] = 0.26379;	randomRayOffsets[29] = 0.27934;
	randomRayOffsets[30] = 0.8298;	randomRayOffsets[31] = 0.8246;
	randomRayOffsets[32] = 0.58916;	randomRayOffsets[33] = 0.98609;
	randomRayOffsets[34] = 0.91096;	randomRayOffsets[35] = 0.22686;
	randomRayOffsets[36] = 0.69511;	randomRayOffsets[37] = 0.98;
	randomRayOffsets[38] = 0.24393;	randomRayOffsets[39] = 0.53387;
	randomRayOffsets[40] = 0.10636;	randomRayOffsets[41] = 0.99941;
	randomRayOffsets[42] = 0.67617;	randomRayOffsets[43] = 0.0157;
	randomRayOffsets[44] = 0.57518;	randomRayOffsets[45] = 0.10005;
	randomRayOffsets[46] = 0.10302;	randomRayOffsets[47] = 0.79888;
	randomRayOffsets[48] = 0.28448;	randomRayOffsets[49] = 0.04564;
	randomRayOffsets[50] = 0.29577;	randomRayOffsets[51] = 0.38201;
	randomRayOffsets[52] = 0.30097;	randomRayOffsets[53] = 0.94857;
	randomRayOffsets[54] = 0.97982;	randomRayOffsets[55] = 0.40137;
	randomRayOffsets[56] = 0.27827;	randomRayOffsets[57] = 0.16044;
	randomRayOffsets[58] = 0.16282;	randomRayOffsets[59] = 0.64658;
	randomRayOffsets[60] = 0.41007;	randomRayOffsets[61] = 0.41276;
	randomRayOffsets[62] = 0.71273;	randomRayOffsets[63] = 0.3262;
	randomRayOffsets[64] = 0.63317;	randomRayOffsets[65] = 0.20756;
	randomRayOffsets[66] = 0.18601;	randomRayOffsets[67] = 0.58335;
	randomRayOffsets[68] = 0.08071;	randomRayOffsets[69] = 0.45797;
	randomRayOffsets[70] = 0.90572;	randomRayOffsets[71] = 0.26136;
	randomRayOffsets[72] = 0.78521;	randomRayOffsets[73] = 0.3789;
	randomRayOffsets[74] = 0.28966;	randomRayOffsets[75] = 0.91937;
	randomRayOffsets[76] = 0.63174;	randomRayOffsets[77] = 0.62764;
	randomRayOffsets[78] = 0.42845;	randomRayOffsets[79] = 0.09797;
	randomRayOffsets[80] = 0.56104;	randomRayOffsets[81] = 0.69448;
	randomRayOffsets[82] = 0.91371;	randomRayOffsets[83] = 0.83481;
	randomRayOffsets[84] = 0.02262;	randomRayOffsets[85] = 0.54336;
	randomRayOffsets[86] = 0.91616;	randomRayOffsets[87] = 0.43026;
	randomRayOffsets[88] = 0.67794;	randomRayOffsets[89] = 0.50245;
	randomRayOffsets[90] = 0.51373;	randomRayOffsets[91] = 0.46298;
	randomRayOffsets[92] = 0.35347;	randomRayOffsets[93] = 0.40483;
	randomRayOffsets[94] = 0.26973;	randomRayOffsets[95] = 0.05559;
	randomRayOffsets[96] = 0.24384;	randomRayOffsets[97] = 0.97907;
	randomRayOffsets[98] = 0.06091;	randomRayOffsets[99] = 0.39029;
	randomRayOffsets[100] = 0.36499;	randomRayOffsets[101] = 0.48989;
	randomRayOffsets[102] = 0.15566;	randomRayOffsets[103] = 0.47445;
	randomRayOffsets[104] = 0.25726;	randomRayOffsets[105] = 0.62875;
	randomRayOffsets[106] = 0.54207;	randomRayOffsets[107] = 0.1563;
	randomRayOffsets[108] = 0.93854;	randomRayOffsets[109] = 0.65449;
	randomRayOffsets[110] = 0.50608;	randomRayOffsets[111] = 0.39047;
	randomRayOffsets[112] = 0.10737;	randomRayOffsets[113] = 0.78399;
	randomRayOffsets[114] = 0.45964;	randomRayOffsets[115] = 0.75368;
	randomRayOffsets[116] = 0.59609;	randomRayOffsets[117] = 0.83273;
	randomRayOffsets[118] = 0.01875;	randomRayOffsets[119] = 0.21036;
	randomRayOffsets[120] = 0.07395;	randomRayOffsets[121] = 0.10545;
	randomRayOffsets[122] = 0.33169;	randomRayOffsets[123] = 0.12824;
	randomRayOffsets[124] = 0.00024;	randomRayOffsets[125] = 0.53679;
	randomRayOffsets[126] = 0.65705;	randomRayOffsets[127] = 0.54401;
	randomRayOffsets[128] = 0.82741;	randomRayOffsets[129] = 0.08189;
	randomRayOffsets[130] = 0.19192;	randomRayOffsets[131] = 0.67891;
	randomRayOffsets[132] = 0.4542;	randomRayOffsets[133] = 0.35702;
	randomRayOffsets[134] = 0.14998;	randomRayOffsets[135] = 0.70439;
	randomRayOffsets[136] = 0.92878;	randomRayOffsets[137] = 0.53021;
	randomRayOffsets[138] = 0.08964;	randomRayOffsets[139] = 0.75772;
	randomRayOffsets[140] = 0.40184;	randomRayOffsets[141] = 0.46187;
	randomRayOffsets[142] = 0.49216;	randomRayOffsets[143] = 0.20762;
	randomRayOffsets[144] = 0.32973;	randomRayOffsets[145] = 0.09542;
	randomRayOffsets[146] = 0.58979;	randomRayOffsets[147] = 0.16987;
	randomRayOffsets[148] = 0.92761;	randomRayOffsets[149] = 0.09792;
	randomRayOffsets[150] = 0.44386;	randomRayOffsets[151] = 0.27294;
	randomRayOffsets[152] = 0.87254;	randomRayOffsets[153] = 0.75068;
	randomRayOffsets[154] = 0.27294;	randomRayOffsets[155] = 0.67364;
	randomRayOffsets[156] = 0.25662;	randomRayOffsets[157] = 0.08989;
	randomRayOffsets[158] = 0.03095;	randomRayOffsets[159] = 0.32271;
	randomRayOffsets[160] = 0.79012;	randomRayOffsets[161] = 0.29725;
	randomRayOffsets[162] = 0.23528;	randomRayOffsets[163] = 0.48047;
	randomRayOffsets[164] = 0.2546;	randomRayOffsets[165] = 0.3406;
	randomRayOffsets[166] = 0.04493;	randomRayOffsets[167] = 0.48242;
	randomRayOffsets[168] = 0.20601;	randomRayOffsets[169] = 0.86453;
	randomRayOffsets[170] = 0.58862;	randomRayOffsets[171] = 0.7549;
	randomRayOffsets[172] = 0.92788;	randomRayOffsets[173] = 0.33101;
	randomRayOffsets[174] = 0.54294;	randomRayOffsets[175] = 0.08069;
	randomRayOffsets[176] = 0.63437;	randomRayOffsets[177] = 0.41003;
	randomRayOffsets[178] = 0.96042;	randomRayOffsets[179] = 0.11462;
	randomRayOffsets[180] = 0.92344;	randomRayOffsets[181] = 0.6202;
	randomRayOffsets[182] = 0.34772;	randomRayOffsets[183] = 0.14924;
	randomRayOffsets[184] = 0.47997;	randomRayOffsets[185] = 0.2194;
	randomRayOffsets[186] = 0.99373;	randomRayOffsets[187] = 0.13042;
	randomRayOffsets[188] = 0.02888;	randomRayOffsets[189] = 0.34539;
	randomRayOffsets[190] = 0.54766;	randomRayOffsets[191] = 0.92295;
	randomRayOffsets[192] = 0.53824;	randomRayOffsets[193] = 0.40642;
	randomRayOffsets[194] = 0.84724;	randomRayOffsets[195] = 0.82622;
	randomRayOffsets[196] = 0.67242;	randomRayOffsets[197] = 0.72189;
	randomRayOffsets[198] = 0.99677;	randomRayOffsets[199] = 0.3398;
	randomRayOffsets[200] = 0.49521;	randomRayOffsets[201] = 0.41296;
	randomRayOffsets[202] = 0.69528;	randomRayOffsets[203] = 0.17908;
	randomRayOffsets[204] = 0.42291;	randomRayOffsets[205] = 0.54317;
	randomRayOffsets[206] = 0.81466;	randomRayOffsets[207] = 0.54091;
	randomRayOffsets[208] = 0.42753;	randomRayOffsets[209] = 0.50906;
	randomRayOffsets[210] = 0.22778;	randomRayOffsets[211] = 0.61918;
	randomRayOffsets[212] = 0.48983;	randomRayOffsets[213] = 0.68081;
	randomRayOffsets[214] = 0.8866;	randomRayOffsets[215] = 0.37051;
	randomRayOffsets[216] = 0.30249;	randomRayOffsets[217] = 0.29286;
	randomRayOffsets[218] = 0.15031;	randomRayOffsets[219] = 0.52982;
	randomRayOffsets[220] = 0.22326;	randomRayOffsets[221] = 0.58452;
	randomRayOffsets[222] = 0.36345;	randomRayOffsets[223] = 0.87597;
	randomRayOffsets[224] = 0.47801;	randomRayOffsets[225] = 0.19063;
	randomRayOffsets[226] = 0.68406;	randomRayOffsets[227] = 0.74741;
	randomRayOffsets[228] = 0.61393;	randomRayOffsets[229] = 0.78213;
	randomRayOffsets[230] = 0.16174;	randomRayOffsets[231] = 0.80777;
	randomRayOffsets[232] = 0.20261;	randomRayOffsets[233] = 0.95676;
	randomRayOffsets[234] = 0.06585;	randomRayOffsets[235] = 0.06152;
	randomRayOffsets[236] = 0.79319;	randomRayOffsets[237] = 0.3796;
	randomRayOffsets[238] = 0.46358;	randomRayOffsets[239] = 0.11954;
	randomRayOffsets[240] = 0.11547;	randomRayOffsets[241] = 0.17377;
	randomRayOffsets[242] = 0.04811;	randomRayOffsets[243] = 0.71481;
	randomRayOffsets[244] = 0.53302;	randomRayOffsets[245] = 0.561;
	randomRayOffsets[246] = 0.21673;	randomRayOffsets[247] = 0.468;
	randomRayOffsets[248] = 0.74635;	randomRayOffsets[249] = 0.75231;
	randomRayOffsets[250] = 0.39893;	randomRayOffsets[251] = 0.90309;
	randomRayOffsets[252] = 0.746;	randomRayOffsets[253] = 0.08855;
	randomRayOffsets[254] = 0.63457;	randomRayOffsets[255] = 0.71302;
	CUDA_vtkCudaVolumeMapper_renderAlgo_loadrandomRayOffsets(randomRayOffsets,this->GetStream());

	//re-copy the image data if any
	if( withData )
		for( std::map<int,vtkImageData*>::iterator it = this->inputImages.begin();
			 it != this->inputImages.end(); it++ )
			this->SetInputInternal(it->second, it->first);

}

void vtkCudaVolumeMapper::SetNumberOfFrames(int n) {
	if( n > 0 && n <= VTKCUDAVOLUMEMAPPER_UPPER_BOUND )
		this->numFrames = n;
}

vtkCudaVolumeMapper::~vtkCudaVolumeMapper()
{
	this->Deinitialize();
	this->VolumeInfoHandler->UnRegister(this);
	this->RendererInfoHandler->UnRegister(this);
	this->OutputInfoHandler->UnRegister(this);
	this->ViewToVoxelsMatrix->UnRegister(this);
	this->WorldToVoxelsMatrix->UnRegister(this);
	this->PerspectiveTransform->UnRegister(this);
	this->VoxelsTransform->UnRegister(this);
	this->VoxelsToViewTransform->UnRegister(this);
	this->NextVoxelsToViewTransform->UnRegister(this);
	if (this->KeyholePlanes)
		this->KeyholePlanes->UnRegister(this);
}

void vtkCudaVolumeMapper::SetInput(vtkImageData * input){

	//set information at this level
	this->vtkVolumeMapper::SetInput(input);
	this->VolumeInfoHandler->SetInputData(input, 0);
	this->inputImages.insert( std::pair<int,vtkImageData*>(0,input) );
	
	//pass down to subclass
	this->SetInputInternal( input, 0 );
	if( this->currFrame == 0 ) this->ChangeFrame(0);
}

void vtkCudaVolumeMapper::SetInput(vtkImageData * input, int index){
	//check for consistency
	if( index < 0 || !(index < this->numFrames) ) return;

	//set information at this level
	this->vtkVolumeMapper::SetInput(input);
	this->VolumeInfoHandler->SetInputData(input, index);
	this->inputImages.insert( std::pair<int,vtkImageData*>(index,input) );

	//pass down to subclass
	this->SetInputInternal(input, index);
	if( this->currFrame == 0 ) this->ChangeFrame(0);
}

vtkImageData * vtkCudaVolumeMapper::GetInput(){
	return GetInput(0);
}

vtkImageData * vtkCudaVolumeMapper::GetInput( int frame){
	if( this->inputImages.find(frame) != this->inputImages.end() )
		return this->inputImages[frame];
	return 0;
}

void vtkCudaVolumeMapper::ClearInput(){
	//clear information at this class level
	this->VolumeInfoHandler->ClearInput();
	this->inputImages.clear();

	//pass down to subclass
	this->ClearInputInternal();
}

void vtkCudaVolumeMapper::SetCelShadingConstants(float darkness, float a, float b){
	this->RendererInfoHandler->SetCelShadingConstants(darkness, a, b);
}

void vtkCudaVolumeMapper::SetDistanceShadingConstants(float darkness, float a, float b){
	this->RendererInfoHandler->SetDistanceShadingConstants(darkness, a, b);
}

void vtkCudaVolumeMapper::SetRenderOutputScaleFactor(float scaleFactor){
	 this->OutputInfoHandler->SetRenderOutputScaleFactor(scaleFactor);
}

void vtkCudaVolumeMapper::ChangeFrame(unsigned int frame){
	if(frame >= 0 && frame < this->numFrames ){
		this->ChangeFrameInternal(frame);
		this->currFrame = frame;
	}
}

void vtkCudaVolumeMapper::AdvanceFrame(){
	this->ChangeFrame( (this->currFrame + 1) % this->numFrames );
}

void vtkCudaVolumeMapper::UseCUDAOpenGLInteroperability(){
	this->OutputInfoHandler->SetRenderType(0);
}

void vtkCudaVolumeMapper::UseFullVTKCompatibility(){
	this->OutputInfoHandler->SetRenderType(1);
}

void vtkCudaVolumeMapper::UseImageDataRenderering(){
	this->OutputInfoHandler->SetRenderType(2);
}

void vtkCudaVolumeMapper::Render(vtkRenderer *renderer, vtkVolume *volume)
{
	//prepare the 3 main information handlers
	if (volume != this->VolumeInfoHandler->GetVolume()) this->VolumeInfoHandler->SetVolume(volume);
	this->VolumeInfoHandler->Update();
	this->RendererInfoHandler->SetRenderer(renderer);
	this->OutputInfoHandler->SetRenderer(renderer);
	this->ComputeMatrices();
	this->RendererInfoHandler->LoadZBuffer();
	this->RendererInfoHandler->SetClippingPlanes( this->ClippingPlanes );
	this->RendererInfoHandler->SetKeyholePlanes( this->KeyholePlanes );
	this->OutputInfoHandler->Prepare();

	//pass the actual rendering process to the subclass
	if( erroredOut ){
		vtkErrorMacro(<< "Error propogation in rendering - cause error flag previously set - MARKER 3");
	}else{
		try{
			this->InternalRender(renderer, volume, 
								 this->RendererInfoHandler->GetRendererInfo(),
								 this->VolumeInfoHandler->GetVolumeInfo(),
								 this->OutputInfoHandler->GetOutputImageInfo() );
		}catch(...){
			erroredOut = true;
			vtkErrorMacro(<< "Internal rendering error - cause unknown - MARKER 2");
		}
	}

	//display the rendered results
	this->OutputInfoHandler->Display(volume,renderer);

	return;
}

vtkImageData* vtkCudaVolumeMapper::GetOutput(){
	return this->OutputInfoHandler->GetCurrentImageData();
}

void vtkCudaVolumeMapper::ComputeMatrices()
{
	// Get the renderer and the volume from the information handlers
	vtkRenderer *ren = this->RendererInfoHandler->GetRenderer();
	vtkVolume *vol = this->VolumeInfoHandler->GetVolume();
	bool flag = false;

	if( ren->GetMTime() > this->renModified ){
		this->renModified = ren->GetMTime();
		flag = true;

		// Get the camera from the renderer
		vtkCamera *cam = ren->GetActiveCamera();
		
		// Get the aspect ratio from the renderer. This is needed for the
		// computation of the perspective matrix
		ren->ComputeAspect();
		double *aspect = ren->GetAspect();

		// Keep track of the projection matrix - we'll need it in a couple of places
		// Get the projection matrix. The method is called perspective, but
		// the matrix is valid for perspective and parallel viewing transforms.
		// Don't replace this with the GetCompositePerspectiveTransformMatrix 
		// because that turns off stereo rendering!!!
		this->PerspectiveTransform->Identity();
		this->PerspectiveTransform->Concatenate(cam->GetProjectionTransformMatrix(aspect[0]/aspect[1], 0.0, 1.0 ));
		this->PerspectiveTransform->Concatenate(cam->GetViewTransformMatrix());
	}
	
	if( vol->GetMTime() > this->volModified ){
		this->volModified = vol->GetMTime();
		flag = true;

		//get the input origin, spacing and extents
		vtkImageData* img = this->VolumeInfoHandler->GetInputData();
		double inputOrigin[3];
		double inputSpacing[3];
		int inputExtent[6];
		img->GetOrigin(inputOrigin);
		img->GetSpacing(inputSpacing);
		img->GetExtent(inputExtent);

		// Compute the origin of the extent the volume origin is at voxel (0,0,0)
		// but we want to consider (0,0,0) in voxels to be at
		// (inputExtent[0], inputExtent[2], inputExtent[4]).
		double extentOrigin[3];
		extentOrigin[0] = inputOrigin[0] + inputExtent[0]*inputSpacing[0];
		extentOrigin[1] = inputOrigin[1] + inputExtent[2]*inputSpacing[1];
		extentOrigin[2] = inputOrigin[2] + inputExtent[4]*inputSpacing[2];
		
		// Create a transform that will account for the scaling and translation of
		// the scalar data. The is the volume to voxels matrix.
		this->VoxelsTransform->Identity();
		this->VoxelsTransform->Translate( extentOrigin[0], extentOrigin[1], extentOrigin[2] );
		this->VoxelsTransform->Scale( inputSpacing[0], inputSpacing[1], inputSpacing[2] );

		// Get the volume matrix. This is a volume to world matrix right now. 
		// We'll need to invert it, translate by the origin and scale by the 
		// spacing to change it to a world to voxels matrix.
		if(vol->GetUserMatrix() != NULL){
			this->VoxelsToViewTransform->SetMatrix( vol->GetUserMatrix() );
		}else{
			this->VoxelsToViewTransform->Identity();
		}

		// Now concatenate the volume's matrix with this scalar data matrix (sending the result off as the voxels to world matrix)
		this->VoxelsToViewTransform->PreMultiply();
		this->VoxelsToViewTransform->Concatenate( this->VoxelsTransform->GetMatrix() );
		this->RendererInfoHandler->SetVoxelsToWorldMatrix( VoxelsToViewTransform->GetMatrix() );

		// Invert the transform (sending the result off as the world to voxels matrix)
		this->WorldToVoxelsMatrix->DeepCopy( this->VoxelsToViewTransform->GetMatrix() );
		this->WorldToVoxelsMatrix->Invert();
		this->RendererInfoHandler->SetWorldToVoxelsMatrix(this->WorldToVoxelsMatrix);

	}

	if(flag){

		this->Modified();
		
		// Compute the voxels to view transform by concatenating the
		// voxels to world matrix with the projection matrix (world to view)
		this->NextVoxelsToViewTransform->DeepCopy( this->VoxelsToViewTransform );
		this->NextVoxelsToViewTransform->PostMultiply();
		this->NextVoxelsToViewTransform->Concatenate( this->PerspectiveTransform->GetMatrix() );

		this->ViewToVoxelsMatrix->DeepCopy(this->NextVoxelsToViewTransform->GetMatrix());
		this->ViewToVoxelsMatrix->Invert();

		//load into the renderer information via the handler
		this->RendererInfoHandler->SetViewToVoxelsMatrix(this->ViewToVoxelsMatrix);
	}

}

vtkCxxSetObjectMacro(vtkCudaVolumeMapper,KeyholePlanes,vtkPlaneCollection);

void vtkCudaVolumeMapper::AddKeyholePlane(vtkPlane *plane){
  if (this->KeyholePlanes == NULL){
    this->KeyholePlanes = vtkPlaneCollection::New();
    this->KeyholePlanes->Register(this);
    this->KeyholePlanes->Delete();
  }

  this->KeyholePlanes->AddItem(plane);
  this->Modified();
}

void vtkCudaVolumeMapper::RemoveKeyholePlane(vtkPlane *plane){
  if (this->KeyholePlanes == NULL) vtkErrorMacro(<< "Cannot remove Keyhole plane: mapper has none");
  this->KeyholePlanes->RemoveItem(plane);
  this->Modified();
}

void vtkCudaVolumeMapper::RemoveAllKeyholePlanes(){
  if ( this->KeyholePlanes ) this->KeyholePlanes->RemoveAllItems();
}

void vtkCudaVolumeMapper::SetKeyholePlanes(vtkPlanes *planes){
  vtkPlane *plane;
  if (!planes) return;

  int numPlanes = planes->GetNumberOfPlanes();

  this->RemoveAllKeyholePlanes();
  for (int i=0; i<numPlanes && i<6; i++){
    plane = vtkPlane::New();
    planes->GetPlane(i, plane);
    this->AddKeyholePlane(plane);
    plane->Delete();
  }
}