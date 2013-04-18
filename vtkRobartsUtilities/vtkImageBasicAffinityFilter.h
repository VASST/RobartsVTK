/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkImageBasicAffinityFilter.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __vtkImageBasicAffinityFilter_H__
#define __vtkImageBasicAffinityFilter_H__

#include "vtkThreadedImageAlgorithm.h"
#include "vtkImageData.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkMultiThreader.h"

class vtkImageBasicAffinityFilter : public vtkThreadedImageAlgorithm
{
public:

	vtkTypeMacro( vtkImageBasicAffinityFilter, vtkThreadedImageAlgorithm )

	static vtkImageBasicAffinityFilter *New();
	
	// Description:
	// Get/Set the number of threads to create when rendering
	vtkSetClampMacro( NumberOfThreads, int, 1, VTK_MAX_THREADS );
	vtkGetMacro( NumberOfThreads, int );
	
	// Description:
	// Get/Set the weights for the basic affinity function
	vtkSetClampMacro( DistanceWeight, double, 0.0, 1000000.0 );
	vtkGetMacro( DistanceWeight, double );
	vtkSetClampMacro( IntensityWeight, double, 0.0, 1000000.0 );
	vtkGetMacro( IntensityWeight, double );

	// The method that starts the multithreading
	template< class T >
	void ThreadedExecuteCasted(vtkImageData *inData, vtkImageData *outData, int threadId, int numThreads);
	void ThreadedExecute(vtkImageData *inData, vtkImageData *outData, int threadId, int numThreads);
protected:
	
	int RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector);

	vtkImageBasicAffinityFilter();
	virtual ~vtkImageBasicAffinityFilter();

private:
	vtkImageBasicAffinityFilter operator=(const vtkImageBasicAffinityFilter&){} //not implemented
	vtkImageBasicAffinityFilter(const vtkImageBasicAffinityFilter&){} //not implemented
	
	vtkMultiThreader* Threader;
	int NumberOfThreads;

	double DistanceWeight;
	double IntensityWeight;

};

#endif