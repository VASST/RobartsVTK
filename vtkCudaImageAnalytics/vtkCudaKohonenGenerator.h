#ifndef __VTKKOHONENGENERATOR_H__
#define __VTKKOHONENGENERATOR_H__

#include "CUDA_kohonengenerator.h"
#include "vtkAlgorithm.h"
#include "vtkImageData.h"
#include "vtkImageCast.h"
#include "vtkTransform.h"
#include "vtkCudaObject.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkAlgorithmOutput.h"

class vtkCudaKohonenGenerator : public vtkThreadedImageAlgorithm, public vtkCudaObject
{
public:
	static vtkCudaKohonenGenerator *New();



	void SetAlphaInitial(double alphaInitial);
	double GetAlphaInitial(){ return (double) alphaInit; }
	void SetAlphaDecay(double alphaDecay);
	double GetAlphaDecay(){ return (double) alphaDecay; }
	void SetWidthInitial(double widthInitial);
	double GetWidthInitial(){ return (double) widthInit; }
	void SetWidthDecay(double widthDecay);
	double GetWidthDecay(){ return (double) widthDecay; }
	
	void UsePositionDataOn();
	void UsePositionDataOff();

	Kohonen_Generator_Information& GetCudaInformation(){ return this->info; }

	// Description:
	// If the subclass does not define an Execute method, then the task
	// will be broken up, multiple threads will be spawned, and each thread
	// will call this method. It is public so that the thread functions
	// can call this method.
	virtual void ThreadedRequestData(vtkInformation *request, 
									vtkInformationVector **inputVector, 
									vtkInformationVector *outputVector,
									vtkImageData ***inData, 
									vtkImageData **outData,
									int extent[6], int threadId);
  
	// also support the old signature
	virtual void ThreadedExecute(vtkImageData *inData, 
								vtkImageData *outData,
								int extent[6], int threadId);

protected:
	vtkCudaKohonenGenerator();
	virtual ~vtkCudaKohonenGenerator();
	
	void Reinitialize(int withData);
	void Deinitialize(int withData);

private:
	vtkCudaKohonenGenerator operator=(const vtkCudaKohonenGenerator&){}
	vtkCudaKohonenGenerator(const vtkCudaKohonenGenerator&){}
	
	int outputWidth;
	int outputHeight;
	
	float alphaInit;
	float alphaDecay;
	float widthInit;
	float widthDecay;
	int numIterations;

	bool usePosition;
	float diagonalLength;
	int outExt[6];

	Kohonen_Generator_Information info;

};

#endif