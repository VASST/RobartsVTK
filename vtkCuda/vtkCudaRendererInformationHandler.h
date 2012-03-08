#ifndef VTKCUDARENDERERINFORMATIONHANDLER_H_
#define VTKCUDARENDERERINFORMATIONHANDLER_H_

#include "vtkObject.h"
#include "vtkMatrix4x4.h"
#include "vtkRenderer.h"
#include "vtkPlaneCollection.h"
#include "vtkCudaMemoryTexture.h"

#include "cudaRendererInformation.h"

class vtkCudaRendererInformationHandler : public vtkObject
{
public:
    static vtkCudaRendererInformationHandler* New();

    //BTX
    void SetRenderer(vtkRenderer* renderer);
    vtkGetMacro(Renderer, vtkRenderer*);
    const cudaRendererInformation& GetRendererInfo() { return (this->RendererInfo); }
    //ETX

    void Bind();
    void Unbind();

    void SetRenderOutputScaleFactor(float scaleFactor);
	void SetGoochShadingConstants(float darkness, float a, float b);
	void SetGradientShadingConstants(float darkness);
	void SetDepthShadingConstants(float darkness);

	void SetViewToVoxelsMatrix(vtkMatrix4x4*);
	void SetVoxelsToWorldMatrix(vtkMatrix4x4*);
	void SetWorldToVoxelsMatrix(vtkMatrix4x4*);

	void HoneDepthShadingConstants(vtkMatrix4x4* viewToVoxels, vtkMatrix4x4* voxelsToView, const float* extent);

	void SetClippingPlanes(vtkPlaneCollection*);

    virtual void Update();

protected:
    vtkCudaRendererInformationHandler();
    ~vtkCudaRendererInformationHandler();

    void UpdateResolution(unsigned int width, unsigned int height);
private:
    vtkCudaRendererInformationHandler& operator=(const vtkCudaRendererInformationHandler&); // not implemented
    vtkCudaRendererInformationHandler(const vtkCudaRendererInformationHandler&); // not implemented


private:
    vtkRenderer*             Renderer;

    cudaRendererInformation  RendererInfo;
    float                    RenderOutputScaleFactor;

    vtkCudaMemoryTexture*	MemoryTexture;

	float					WorldToVoxelsMatrix[16];
	float					VoxelsToWorldMatrix[16];

	float					depthShadeDarkness;
	float					twoOldDepthShift;
	float					twoOldDepthScale;

	unsigned int			clipModified;
};

#endif /* VTKCUDARENDERERINFORMATIONHANDLER_H_ */
