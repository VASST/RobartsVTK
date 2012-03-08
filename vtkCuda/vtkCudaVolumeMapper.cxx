// Type
#include "vtkCudaVolumeMapper.h"
#include "vtkObjectFactory.h"

// Volume
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkImageData.h"

// Rendering
#include "vtkCamera.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"

// VTKCUDA
#include "vtkCudaVolumeInformationHandler.h"
#include "vtkCudaRendererInformationHandler.h"
#include "vtkCudaMemoryTexture.h"

#include "vtkgl.h"
extern "C" {
#include "CUDA_renderAlgo.h"
#include "cuda_runtime_api.h"
}


vtkStandardNewMacro(vtkCudaVolumeMapper);

vtkCudaVolumeMapper::vtkCudaVolumeMapper()
{
    this->VolumeInfoHandler = vtkCudaVolumeInformationHandler::New();
    this->RendererInfoHandler = vtkCudaRendererInformationHandler::New();

	this->PerspectiveMatrix = vtkMatrix4x4::New();
	this->ViewToVoxelsMatrix = vtkMatrix4x4::New();
	this->VoxelsToViewMatrix = vtkMatrix4x4::New();
	this->WorldToVoxelsMatrix = vtkMatrix4x4::New();
	this->VoxelsToWorldMatrix = vtkMatrix4x4::New();
	this->PerspectiveTransform = vtkTransform::New();
	this->VoxelsTransform = vtkTransform::New();
	this->VoxelsToViewTransform = vtkTransform::New();
	this->NextVoxelsToViewTransform = vtkTransform::New();

	this->renModified = 0;
	this->volModified = 0;
	this->currFrame = 0;
	this->numFrames = 1;
	this->SetFrameRate(20.0);

	this->timer = vtkTimerLog::New();
	this->lastFrameTime = timer->GetUniversalTime();

	//define a random sequence to smooth away staircase artefacts
	float randoms[256];
   randoms[0] = 0.70554;
   randoms[1] = 0.53342;
   randoms[2] = 0.57951;
   randoms[3] = 0.28956;
   randoms[4] = 0.30194;
   randoms[5] = 0.77474;
   randoms[6] = 0.01401;
   randoms[7] = 0.76072;
   randoms[8] = 0.81449;
   randoms[9] = 0.70903;
   randoms[10] = 0.04535;
   randoms[11] = 0.41403;
   randoms[12] = 0.86261;
   randoms[13] = 0.79048;
   randoms[14] = 0.37353;
   randoms[15] = 0.96195;
   randoms[16] = 0.87144;
   randoms[17] = 0.05623;
   randoms[18] = 0.94955;
   randoms[19] = 0.36401;
   randoms[20] = 0.52486;
   randoms[21] = 0.76711;
   randoms[22] = 0.0535;
   randoms[23] = 0.59245;
   randoms[24] = 0.4687;
   randoms[25] = 0.29816;
   randoms[26] = 0.62269;
   randoms[27] = 0.64782;
   randoms[28] = 0.26379;
   randoms[29] = 0.27934;
   randoms[30] = 0.8298;
   randoms[31] = 0.8246;
   randoms[32] = 0.58916;
   randoms[33] = 0.98609;
   randoms[34] = 0.91096;
   randoms[35] = 0.22686;
   randoms[36] = 0.69511;
   randoms[37] = 0.98;
   randoms[38] = 0.24393;
   randoms[39] = 0.53387;
   randoms[40] = 0.10636;
   randoms[41] = 0.99941;
   randoms[42] = 0.67617;
   randoms[43] = 0.0157;
   randoms[44] = 0.57518;
   randoms[45] = 0.10005;
   randoms[46] = 0.10302;
   randoms[47] = 0.79888;
   randoms[48] = 0.28448;
   randoms[49] = 0.04564;
   randoms[50] = 0.29577;
   randoms[51] = 0.38201;
   randoms[52] = 0.30097;
   randoms[53] = 0.94857;
   randoms[54] = 0.97982;
   randoms[55] = 0.40137;
   randoms[56] = 0.27827;
   randoms[57] = 0.16044;
   randoms[58] = 0.16282;
   randoms[59] = 0.64658;
   randoms[60] = 0.41007;
   randoms[61] = 0.41276;
   randoms[62] = 0.71273;
   randoms[63] = 0.3262;
   randoms[64] = 0.63317;
   randoms[65] = 0.20756;
   randoms[66] = 0.18601;
   randoms[67] = 0.58335;
   randoms[68] = 0.08071;
   randoms[69] = 0.45797;
   randoms[70] = 0.90572;
   randoms[71] = 0.26136;
   randoms[72] = 0.78521;
   randoms[73] = 0.3789;
   randoms[74] = 0.28966;
   randoms[75] = 0.91937;
   randoms[76] = 0.63174;
   randoms[77] = 0.62764;
   randoms[78] = 0.42845;
   randoms[79] = 0.09797;
   randoms[80] = 0.56104;
   randoms[81] = 0.69448;
   randoms[82] = 0.91371;
   randoms[83] = 0.83481;
   randoms[84] = 0.02262;
   randoms[85] = 0.54336;
   randoms[86] = 0.91616;
   randoms[87] = 0.43026;
   randoms[88] = 0.67794;
   randoms[89] = 0.50245;
   randoms[90] = 0.51373;
   randoms[91] = 0.46298;
   randoms[92] = 0.35347;
   randoms[93] = 0.40483;
   randoms[94] = 0.26973;
   randoms[95] = 0.05559;
   randoms[96] = 0.24384;
   randoms[97] = 0.97907;
   randoms[98] = 0.06091;
   randoms[99] = 0.39029;
   randoms[100] = 0.36499;
   randoms[101] = 0.48989;
   randoms[102] = 0.15566;
   randoms[103] = 0.47445;
   randoms[104] = 0.25726;
   randoms[105] = 0.62875;
   randoms[106] = 0.54207;
   randoms[107] = 0.1563;
   randoms[108] = 0.93854;
   randoms[109] = 0.65449;
   randoms[110] = 0.50608;
   randoms[111] = 0.39047;
   randoms[112] = 0.10737;
   randoms[113] = 0.78399;
   randoms[114] = 0.45964;
   randoms[115] = 0.75368;
   randoms[116] = 0.59609;
   randoms[117] = 0.83273;
   randoms[118] = 0.01875;
   randoms[119] = 0.21036;
   randoms[120] = 0.07395;
   randoms[121] = 0.10545;
   randoms[122] = 0.33169;
   randoms[123] = 0.12824;
   randoms[124] = 0.00024;
   randoms[125] = 0.53679;
   randoms[126] = 0.65705;
   randoms[127] = 0.54401;
   randoms[128] = 0.82741;
   randoms[129] = 0.08189;
   randoms[130] = 0.19192;
   randoms[131] = 0.67891;
   randoms[132] = 0.4542;
   randoms[133] = 0.35702;
   randoms[134] = 0.14998;
   randoms[135] = 0.70439;
   randoms[136] = 0.92878;
   randoms[137] = 0.53021;
   randoms[138] = 0.08964;
   randoms[139] = 0.75772;
   randoms[140] = 0.40184;
   randoms[141] = 0.46187;
   randoms[142] = 0.49216;
   randoms[143] = 0.20762;
   randoms[144] = 0.32973;
   randoms[145] = 0.09542;
   randoms[146] = 0.58979;
   randoms[147] = 0.16987;
   randoms[148] = 0.92761;
   randoms[149] = 0.09792;
   randoms[150] = 0.44386;
   randoms[151] = 0.27294;
   randoms[152] = 0.87254;
   randoms[153] = 0.75068;
   randoms[154] = 0.27294;
   randoms[155] = 0.67364;
   randoms[156] = 0.25662;
   randoms[157] = 0.08989;
   randoms[158] = 0.03095;
   randoms[159] = 0.32271;
   randoms[160] = 0.79012;
   randoms[161] = 0.29725;
   randoms[162] = 0.23528;
   randoms[163] = 0.48047;
   randoms[164] = 0.2546;
   randoms[165] = 0.3406;
   randoms[166] = 0.04493;
   randoms[167] = 0.48242;
   randoms[168] = 0.20601;
   randoms[169] = 0.86453;
   randoms[170] = 0.58862;
   randoms[171] = 0.7549;
   randoms[172] = 0.92788;
   randoms[173] = 0.33101;
   randoms[174] = 0.54294;
   randoms[175] = 0.08069;
   randoms[176] = 0.63437;
   randoms[177] = 0.41003;
   randoms[178] = 0.96042;
   randoms[179] = 0.11462;
   randoms[180] = 0.92344;
   randoms[181] = 0.6202;
   randoms[182] = 0.34772;
   randoms[183] = 0.14924;
   randoms[184] = 0.47997;
   randoms[185] = 0.2194;
   randoms[186] = 0.99373;
   randoms[187] = 0.13042;
   randoms[188] = 0.02888;
   randoms[189] = 0.34539;
   randoms[190] = 0.54766;
   randoms[191] = 0.92295;
   randoms[192] = 0.53824;
   randoms[193] = 0.40642;
   randoms[194] = 0.84724;
   randoms[195] = 0.82622;
   randoms[196] = 0.67242;
   randoms[197] = 0.72189;
   randoms[198] = 0.99677;
   randoms[199] = 0.3398;
   randoms[200] = 0.49521;
   randoms[201] = 0.41296;
   randoms[202] = 0.69528;
   randoms[203] = 0.17908;
   randoms[204] = 0.42291;
   randoms[205] = 0.54317;
   randoms[206] = 0.81466;
   randoms[207] = 0.54091;
   randoms[208] = 0.42753;
   randoms[209] = 0.50906;
   randoms[210] = 0.22778;
   randoms[211] = 0.61918;
   randoms[212] = 0.48983;
   randoms[213] = 0.68081;
   randoms[214] = 0.8866;
   randoms[215] = 0.37051;
   randoms[216] = 0.30249;
   randoms[217] = 0.29286;
   randoms[218] = 0.15031;
   randoms[219] = 0.52982;
   randoms[220] = 0.22326;
   randoms[221] = 0.58452;
   randoms[222] = 0.36345;
   randoms[223] = 0.87597;
   randoms[224] = 0.47801;
   randoms[225] = 0.19063;
   randoms[226] = 0.68406;
   randoms[227] = 0.74741;
   randoms[228] = 0.61393;
   randoms[229] = 0.78213;
   randoms[230] = 0.16174;
   randoms[231] = 0.80777;
   randoms[232] = 0.20261;
   randoms[233] = 0.95676;
   randoms[234] = 0.06585;
   randoms[235] = 0.06152;
   randoms[236] = 0.79319;
   randoms[237] = 0.3796;
   randoms[238] = 0.46358;
   randoms[239] = 0.11954;
   randoms[240] = 0.11547;
   randoms[241] = 0.17377;
   randoms[242] = 0.04811;
   randoms[243] = 0.71481;
   randoms[244] = 0.53302;
   randoms[245] = 0.561;
   randoms[246] = 0.21673;
   randoms[247] = 0.468;
   randoms[248] = 0.74635;
   randoms[249] = 0.75231;
   randoms[250] = 0.39893;
   randoms[251] = 0.90309;
   randoms[252] = 0.746;
   randoms[253] = 0.08855;
   randoms[254] = 0.63457;
   randoms[255] = 0.71302;

	CUDAkernelsetup_loadRandoms(randoms);
	CUDAkernelsetup_initImageArray();
}  

vtkCudaVolumeMapper::~vtkCudaVolumeMapper()
{
    this->VolumeInfoHandler->Delete();
    this->RendererInfoHandler->Delete();
	this->PerspectiveMatrix->Delete();
	this->ViewToVoxelsMatrix->Delete();
	this->VoxelsToViewMatrix->Delete();
	this->WorldToVoxelsMatrix->Delete();
	this->VoxelsToWorldMatrix->Delete();
	this->PerspectiveTransform->Delete();
	this->VoxelsTransform->Delete();
	this->VoxelsToViewTransform->Delete();
	CUDAkernelsetup_clearImageArray();
}

void vtkCudaVolumeMapper::SetInput(vtkImageData * input)
{
    this->vtkVolumeMapper::SetInput(input);
    this->VolumeInfoHandler->SetInputData(input, 0);
}

void vtkCudaVolumeMapper::SetInput(vtkImageData * input, int index){
    this->vtkVolumeMapper::SetInput(input);
    this->VolumeInfoHandler->SetInputData(input, index);
}

void vtkCudaVolumeMapper::SetRenderMode(int mode)
{
    //HACK
    //this->MemoryTexture->SetRenderMode(mode);
}

int vtkCudaVolumeMapper::GetCurrentRenderMode() const
{
    //HACK
    return 0; //this->MemoryTexture->GetCurrentRenderMode();
    //TODO
}

void vtkCudaVolumeMapper::SetSampleDistance(float sampleDistance)
{
    this->VolumeInfoHandler->SetSampleDistance(sampleDistance);
}

void vtkCudaVolumeMapper::SetGoochShadingConstants(float darkness, float a, float b){
	this->RendererInfoHandler->SetGoochShadingConstants(darkness, a, b);
}

void vtkCudaVolumeMapper::SetGradientShadingConstants(float darkness){
	this->RendererInfoHandler->SetGradientShadingConstants(darkness);
}

void vtkCudaVolumeMapper::SetDepthShadingConstants(float darkness){
	this->RendererInfoHandler->SetDepthShadingConstants(darkness);
}

void vtkCudaVolumeMapper::SetRenderOutputScaleFactor(float scaleFactor)
{
    this->RendererInfoHandler->SetRenderOutputScaleFactor(scaleFactor);
}

void vtkCudaVolumeMapper::SetFunction(vtkCuda2DTransferClassificationFunction* funct){
	this->VolumeInfoHandler->SetTransferFunction(funct);
}

void vtkCudaVolumeMapper::Render(vtkRenderer *renderer, vtkVolume *volume)
{

	if (volume != this->VolumeInfoHandler->GetVolume()){
        this->VolumeInfoHandler->SetVolume(volume);
	}
	this->VolumeInfoHandler->Update();
    
	this->RendererInfoHandler->SetRenderer(renderer);
	this->ComputeMatrices();
	this->RendererInfoHandler->SetClippingPlanes( this->ClippingPlanes );
    this->RendererInfoHandler->Bind();

	double temp = this->timer->GetUniversalTime();
	if( this->lastFrameTime + this->frameDiff < temp ){
		int frame = this->currFrame + (0.5 + (temp - this->lastFrameTime) / this->frameDiff);
		this->lastFrameTime += this->frameDiff * (double) (frame - this->currFrame);
		this->currFrame = frame % this->numFrames;
		CUDArenderAlgo_changeFrame(this->currFrame);
	}

    CUDArenderAlgo_doRender(this->RendererInfoHandler->GetRendererInfo(),
							this->VolumeInfoHandler->GetVolumeInfo());

	//approximate the visual depth of the rendered image in 3D space (not taking into accoutn clipping planes
	renderer->SetWorldPoint(volume->GetCenter()[0], volume->GetCenter()[1],
							volume->GetCenter()[2], 1.0 );
	renderer->WorldToView();
	float depth = 1.0 - 2.0 * renderer->GetViewPoint()[2];

	// Enter 2D Mode
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, 1.0, 1.0, 0.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
    glPushMatrix();
    glLoadIdentity();

    // Actual Rendering
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2i(1,0);   glVertex3f(0, 1, depth);
    glTexCoord2i(0,0);   glVertex3f(1, 1, depth);
    glTexCoord2i(0,1);   glVertex3f(1, 0, depth);
    glTexCoord2i(1,1);   glVertex3f(0, 0, depth);
    glEnd();
    this->RendererInfoHandler->Unbind();

    // Leave the 2D Mode again.
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();

    cudaThreadSynchronize();

    return;
}

void vtkCudaVolumeMapper::PrintSelf(ostream& os, vtkIndent indent)
{
    vtkVolumeMapper::PrintSelf(os, indent);
}

void vtkCudaVolumeMapper::SetFrameRate(double n)
{
	if(n > 0.0) this->frameDiff = 1.0 / n;
}

void vtkCudaVolumeMapper::ComputeMatrices()
{
	// Get the renderer and the volume from the information handlers
	vtkRenderer *ren = this->RendererInfoHandler->GetRenderer();
	vtkVolume *vol = this->VolumeInfoHandler->GetVolume();
	bool flag = false;

	if(ren->GetMTime() > this->renModified){
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
		this->PerspectiveMatrix->DeepCopy(this->PerspectiveTransform->GetMatrix());
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
		
		// Get the volume matrix. This is a volume to world matrix right now. 
		// We'll need to invert it, translate by the origin and scale by the 
		// spacing to change it to a world to voxels matrix.
		if(vol->GetUserMatrix() != NULL){
			this->VoxelsToViewTransform->SetMatrix( vol->GetUserMatrix() );
		}else{
			this->VoxelsToViewTransform->Identity();
		}

		// Create a transform that will account for the scaling and translation of
		// the scalar data. The is the volume to voxels matrix.
		this->VoxelsTransform->Identity();
		this->VoxelsTransform->Translate( extentOrigin[0], extentOrigin[1], extentOrigin[2] );
		this->VoxelsTransform->Scale( inputSpacing[0], inputSpacing[1], inputSpacing[2] );

		// Now concatenate the volume's matrix with this scalar data matrix
		this->VoxelsToViewTransform->PreMultiply();
		this->VoxelsToViewTransform->Concatenate( this->VoxelsTransform->GetMatrix() );

		// Now we actually have the world to voxels matrix - copy it out
		this->WorldToVoxelsMatrix->DeepCopy( VoxelsToViewTransform->GetMatrix() );
		this->WorldToVoxelsMatrix->Invert();
	  
		// We also want to invert this to get voxels to world
		this->VoxelsToWorldMatrix->DeepCopy( VoxelsToViewTransform->GetMatrix() );
		
		//change the renderer information variables
		this->RendererInfoHandler->SetWorldToVoxelsMatrix(this->WorldToVoxelsMatrix);
		this->RendererInfoHandler->SetVoxelsToWorldMatrix(this->VoxelsToWorldMatrix);

	}

	if(flag){

		this->Modified();
		
		// Compute the voxels to view transform by concatenating the
		// voxels to world matrix with the projection matrix (world to view)
		this->NextVoxelsToViewTransform->DeepCopy( this->VoxelsToViewTransform );
		this->NextVoxelsToViewTransform->PostMultiply();
		this->NextVoxelsToViewTransform->Concatenate( this->PerspectiveMatrix );

		this->VoxelsToViewMatrix->DeepCopy(  this->NextVoxelsToViewTransform->GetMatrix() );

		this->ViewToVoxelsMatrix->DeepCopy( this->VoxelsToViewMatrix );
		this->ViewToVoxelsMatrix->Invert();

		//load into the renderer information via the handler
		this->RendererInfoHandler->SetViewToVoxelsMatrix(this->ViewToVoxelsMatrix);
		this->RendererInfoHandler->HoneDepthShadingConstants(this->ViewToVoxelsMatrix, this->VoxelsToViewMatrix, this->VolumeInfoHandler->GetVolumeInfo().Bounds );
	}

}
