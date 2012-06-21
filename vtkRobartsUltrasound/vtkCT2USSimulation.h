#ifndef __vtkCT2USSimulation_H__
#define __vtkCT2USSimulation_H__

#include "vtkThreadedImageAlgorithm.h"
#include "vtkImageData.h"
#include "vtkTransform.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkMultiThreader.h"
#include "vtkImplicitVolume.h"

struct vtkCT2USSimulationInformation;

class vtkCT2USSimulation : public vtkThreadedImageAlgorithm
{
public:

	vtkTypeMacro( vtkCT2USSimulation, vtkThreadedImageAlgorithm )

	static vtkCT2USSimulation *New();

	void SetTransform( vtkTransform * );

	//output parameters
	void SetOutputResolution(int x, int y, int z);
	void SetLogarithmicScaleFactor(double factor);
	void SetTotalReflectionThreshold(double threshold);
	void SetLinearCombinationAlpha(double a); //weighting for the reflection
	void SetLinearCombinationBeta(double b); //weighting for the density
	void SetLinearCombinationBias(double bias); //bias amount
	void SetDensityScaleModel(double scale, double offset);

	//probe geometry
	void SetProbeWidth(double x, double y);
	void SetFanAngle(double xAngle, double yAngle);
	void SetNearClippingDepth(double depth);
	void SetFarClippingDepth(double depth);
	
	// Description:
	// Get/Set the number of threads to create when rendering
	vtkSetClampMacro( NumberOfThreads, int, 1, VTK_MAX_THREADS );
	vtkGetMacro( NumberOfThreads, int );
	
	// The method that starts the multithreading
	void ThreadedExecute(vtkImageData *inData, vtkImageData *outData, int threadId, int numThreads);
protected:
	
	int RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector);

	vtkCT2USSimulation();
	virtual ~vtkCT2USSimulation();

private:
	vtkCT2USSimulation operator=(const vtkCT2USSimulation&){}
	vtkCT2USSimulation(const vtkCT2USSimulation&){}
	
	vtkCT2USSimulationInformation* CT2USInformation;
	vtkTransform* usTransform;
	vtkTransform* VoxelsTransform;
	vtkMatrix4x4* WorldToVoxelsMatrix;

	vtkMultiThreader* Threader;
	int NumberOfThreads;

	void FindVectors(const double normIndex[2], double rayStart[2], double rayInc[2]);
	void SampleAlongRay(const int index[2], double rayStart[2], const double rayInc[2], const int numStepsToTake,
										const int xResolution, const int yResolution, char* const outputUltrasound);
	void GetCTValue(double i[3], double& f, double g[3]);
	vtkImplicitVolume* Interpolator;

};

#endif