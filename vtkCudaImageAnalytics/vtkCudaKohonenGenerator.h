#ifndef __VTKCUDAKOHONENGENERATOR_H__
#define __VTKCUDAKOHONENGENERATOR_H__

#include "CUDA_kohonengenerator.h"
#include "vtkAlgorithm.h"
#include "vtkImageData.h"
#include "vtkImageCast.h"
#include "vtkTransform.h"
#include "vtkCudaObject.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkAlgorithmOutput.h"

class vtkCudaKohonenGenerator : public vtkImageAlgorithm, public vtkCudaObject
{
public:
	vtkTypeMacro( vtkCudaKohonenGenerator, vtkImageAlgorithm );

	static vtkCudaKohonenGenerator *New();

	void SetAlphaInitial(double alphaInitial);
	double GetAlphaInitial(){ return (double) alphaInit; }
	void SetAlphaProlong(double alphaProlong);
	double GetAlphaProlong(){ return (double) alphaProlong; }
	void SetAlphaDecay(double alphaDecay);
	double GetAlphaDecay(){ return (double) alphaDecay; }
	void SetAlphaBaseline(double alphaBaseline);
	double GetAlphaBaseline(){ return (double) alphaBaseline; }

	void SetWidthInitial(double widthInitial);
	double GetWidthInitial(){ return (double) widthInit; }
	void SetWidthProlong(double widthProlong);
	double GetWidthProlong(){ return (double) widthProlong; }
	void SetWidthDecay(double widthDecay);
	double GetWidthDecay(){ return (double) widthDecay; }
	void SetWidthBaseline(double widthBaseline);
	double GetWidthBaseline(){ return (double) widthBaseline; }
	
	void SetNumberOfIterations(int number);
	int GetNumberOfIterations();

	void SetWeight(int index, double weight);
	void SetWeights(const double* weights);
	double GetWeight(int index);
	double* GetWeights();
	void SetWeightNormalization(bool set);
	bool GetWeightNormalization();

	void SetBatchSize(double fraction);
	double GetBatchSize();

	void SetKohonenMapSize(int SizeX, int SizeY);
	
	vtkDataObject* GetInput(int idx);
	void SetInput(int idx, vtkDataObject *input);

	Kohonen_Generator_Information& GetCudaInformation(){ return this->info; }

	bool GetUseMaskFlag();
	void SetUseMaskFlag(bool t);

	bool GetUseAllVoxelsFlag();
	void SetUseAllVoxelsFlag(bool t);

	// Description:
	// If the subclass does not define an Execute method, then the task
	// will be broken up, multiple threads will be spawned, and each thread
	// will call this method. It is public so that the thread functions
	// can call this method.
	virtual int RequestData(vtkInformation *request, 
							 vtkInformationVector **inputVector, 
							 vtkInformationVector *outputVector);
	virtual int RequestInformation( vtkInformation* request,
							 vtkInformationVector** inputVector,
							 vtkInformationVector* outputVector);
	virtual int RequestUpdateExtent( vtkInformation* request,
							 vtkInformationVector** inputVector,
							 vtkInformationVector* outputVector);
	virtual int FillInputPortInformation(int i, vtkInformation* info);

protected:
	vtkCudaKohonenGenerator();
	virtual ~vtkCudaKohonenGenerator();
	
	void Reinitialize(int withData);
	void Deinitialize(int withData);

private:
	vtkCudaKohonenGenerator operator=(const vtkCudaKohonenGenerator&){}
	vtkCudaKohonenGenerator(const vtkCudaKohonenGenerator&){}
	
	float alphaInit;
	float alphaProlong;
	float alphaDecay;
	float alphaBaseline;

	float widthInit;
	float widthProlong;
	float widthDecay;
	float widthBaseline;

	int outExt[6];

	Kohonen_Generator_Information info;

	double	UnnormalizedWeights[MAX_DIMENSIONALITY];
	bool	WeightNormalization;

	int		MaxEpochs;
	double	BatchPercent;
	bool	UseAllVoxels;

	bool	UseMask;

};

#endif