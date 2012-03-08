#ifndef vtkCudaVolumeInformationHandler_H_
#define vtkCudaVolumeInformationHandler_H_

#include "vtkObject.h"
#include "vtkVolume.h"
#include "vtkImageData.h"
#include "cudaVolumeInformation.h"
#include "vtkCuda2DTransferClassificationFunction.h"

class vtkCudaVolumeInformationHandler : public vtkObject
{
public:
    static vtkCudaVolumeInformationHandler* New();

    //BTX
    vtkGetMacro(Volume, vtkVolume*);
    void SetVolume(vtkVolume* Volume);
    void SetInputData(vtkImageData* inputData, int index);
	vtkImageData* GetInputData() const { return InputData; }
    const cudaVolumeInformation& GetVolumeInfo() const { return (this->VolumeInfo); }
    //ETX

    void SetSampleDistance(float sampleDistance);
	void SetTransferFunction(vtkCuda2DTransferClassificationFunction*);
    virtual void Update();

protected:
    vtkCudaVolumeInformationHandler();
    ~vtkCudaVolumeInformationHandler();

    void UpdateTransferFunction();
    void UpdateVolume();
    void UpdateImageData(int index);

    virtual void PrintSelf(ostream& os, vtkIndent indent);


private:
    vtkCudaVolumeInformationHandler& operator=(const vtkCudaVolumeInformationHandler&); // not implemented
    vtkCudaVolumeInformationHandler(const vtkCudaVolumeInformationHandler&); // not implemented


private:
	
	vtkImageData*           InputData;
    vtkVolume*              Volume;

	vtkCuda2DTransferClassificationFunction* function;

    cudaVolumeInformation   VolumeInfo;

	float					FunctionRange[2];
	int						FunctionSize;

};

#endif /* vtkCudaVolumeInformationHandler_H_ */
