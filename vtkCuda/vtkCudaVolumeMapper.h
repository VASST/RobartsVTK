#ifndef __vtkCudaVolumeMapper_h
#define __vtkCudaVolumeMapper_h

#include "vtkVolumeMapper.h"

class vtkVolumeProperty;

#include "vtkTransform.h"
#include "vtkMatrix4x4.h"
#include "vtkTimerLog.h"
#include "vtkAlgorithmOutput.h"

#include "vtkCudaRendererInformationHandler.h"
#include "vtkCudaVolumeInformationHandler.h"

class vtkCudaMemoryTexture;

class vtkCudaVolumeMapper : public vtkVolumeMapper
{
public:
    static vtkCudaVolumeMapper *New();

    virtual void SetInput( vtkImageData * );
    virtual void SetInput( vtkImageData * , int);
    virtual void Render(vtkRenderer *, vtkVolume *);

	void SetNumberOfFrames(int n) {this->numFrames = n;}
	void SetFrameRate(double n);

    void SetSampleDistance(float sampleDistance);
    void SetRenderOutputScaleFactor(float scaleFactor);
	void SetGoochShadingConstants(float darkness, float a, float b);
	void SetGradientShadingConstants(float darkness);
	void SetDepthShadingConstants(float darkness);

	void SetFunction(vtkCuda2DTransferClassificationFunction*);

   //BTX
   void SetRenderMode(int mode);
   int GetCurrentRenderMode() const;

   vtkImageData* GetOutput() { return NULL; }

   void PrintSelf(ostream& os, vtkIndent indent);

protected:
    vtkCudaVolumeMapper();
    virtual ~vtkCudaVolumeMapper();

    void UpdateOutputResolution(unsigned int width, unsigned int height, bool TypeChanged = false);

    vtkCudaRendererInformationHandler* RendererInfoHandler;
    vtkCudaVolumeInformationHandler* VolumeInfoHandler;

	//modified time variables used to minimize setup
	unsigned int	renModified;
	unsigned int	volModified;
	unsigned int	currFrame;
	unsigned int	numFrames;
	double			frameDiff;
	double			lastFrameTime;
	vtkTimerLog*	timer;

	// Some variables used for ray computation
	void ComputeMatrices();
	vtkMatrix4x4   *PerspectiveMatrix;
	vtkMatrix4x4   *ViewToVoxelsMatrix;
	vtkMatrix4x4   *VoxelsToViewMatrix;
	vtkMatrix4x4   *WorldToVoxelsMatrix;
	vtkMatrix4x4   *VoxelsToWorldMatrix;

	vtkTransform   *PerspectiveTransform;
	vtkTransform   *VoxelsTransform;
	vtkTransform   *VoxelsToViewTransform;
	vtkTransform   *NextVoxelsToViewTransform;

private:
    vtkCudaVolumeMapper operator=(const vtkCudaVolumeMapper&);
    vtkCudaVolumeMapper(const vtkCudaVolumeMapper&);

	void RenderTextureInternal( vtkVolume *vol, vtkRenderer *ren,
                            int imageMemorySize[2], int imageViewportSize[2],
                            int imageInUseSize[2], int imageOrigin[2],
                            float requestedDepth, int imageScalarType,
                            void *image );

};

#endif /* __vtkCudaVolumeMapper_h */
