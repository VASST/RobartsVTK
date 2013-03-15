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

	void SetMeansAlphaInitial(double alphaInitial);
	double GetMeansAlphaInitial(){ return (double) MeansAlphaInit; }
	void SetMeansAlphaProlong(double alphaProlong);
	double GetMeansAlphaProlong(){ return (double) MeansAlphaProlong; }
	void SetMeansAlphaDecay(double alphaDecay);
	double GetMeansAlphaDecay(){ return (double) MeansAlphaDecay; }
	void SetMeansAlphaBaseline(double alphaBaseline);
	double GetMeansAlphaBaseline(){ return (double) MeansAlphaBaseline; }

	void SetMeansWidthInitial(double widthInitial);
	double GetMeansWidthInitial(){ return (double) MeansWidthInit; }
	void SetMeansWidthProlong(double widthProlong);
	double GetMeansWidthProlong(){ return (double) MeansWidthProlong; }
	void SetMeansWidthDecay(double widthDecay);
	double GetMeansWidthDecay(){ return (double) MeansWidthDecay; }
	void SetMeansWidthBaseline(double widthBaseline);
	double GetMeansWidthBaseline(){ return (double) MeansWidthBaseline; }
	
	void SetVarsAlphaInitial(double alphaInitial);
	double GetVarsAlphaInitial(){ return (double) VarsAlphaInit; }
	void SetVarsAlphaProlong(double alphaProlong);
	double GetVarsAlphaProlong(){ return (double) VarsAlphaProlong; }
	void SetVarsAlphaDecay(double alphaDecay);
	double GetVarsAlphaDecay(){ return (double) VarsAlphaDecay; }
	void SetVarsAlphaBaseline(double alphaBaseline);
	double GetVarsAlphaBaseline(){ return (double) VarsAlphaBaseline; }

	void SetVarsWidthInitial(double widthInitial);
	double GetVarsWidthInitial(){ return (double) VarsWidthInit; }
	void SetVarsWidthProlong(double widthProlong);
	double GetVarsWidthProlong(){ return (double) VarsWidthProlong; }
	void SetVarsWidthDecay(double widthDecay);
	double GetVarsWidthDecay(){ return (double) VarsWidthDecay; }
	void SetVarsWidthBaseline(double widthBaseline);
	double GetVarsWidthBaseline(){ return (double) VarsWidthBaseline; }
	
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

	Kohonen_Generator_Information& GetCudaInformation(){ return this->info; }

	vtkCudaKohonenGenerator operator=(const vtkCudaKohonenGenerator&){}
	vtkCudaKohonenGenerator(const vtkCudaKohonenGenerator&){}
	
	float MeansAlphaInit;
	float MeansAlphaProlong;
	float MeansAlphaDecay;
	float MeansAlphaBaseline;

	float MeansWidthInit;
	float MeansWidthProlong;
	float MeansWidthDecay;
	float MeansWidthBaseline;

	float VarsAlphaInit;
	float VarsAlphaProlong;
	float VarsAlphaDecay;
	float VarsAlphaBaseline;

	float VarsWidthInit;
	float VarsWidthProlong;
	float VarsWidthDecay;
	float VarsWidthBaseline;

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