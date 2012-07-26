#ifndef __vtkImage2DHistogram_H__
#define __vtkImage2DHistogram_H__

#include "vtkThreadedImageAlgorithm.h"
#include "vtkImageData.h"
#include "vtkMultiThreader.h"

class vtkImage2DHistogram : public vtkThreadedImageAlgorithm
{
public:

	vtkTypeMacro( vtkImage2DHistogram, vtkThreadedImageAlgorithm )

	static vtkImage2DHistogram *New();
	
	// Description:
	// Get/Set the number of threads to create when rendering
	vtkSetClampMacro( NumberOfThreads, int, 1, VTK_MAX_THREADS );
	vtkGetMacro( NumberOfThreads, int );
	
	// Description:
	// Get/Set the resolution of the returned histogram
	void SetResolution( int res[2] );
	vtkGetMacro( Resolution, int* );

	// The method that starts the multithreading
	template< class T >
	void ThreadedExecuteCasted(vtkImageData *inData, vtkImageData *outData, int threadId, int numThreads);
	void ThreadedExecute(vtkImageData *inData, vtkImageData *outData, int threadId, int numThreads);

protected:

	int Resolution[2];

	int RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector);

	vtkImage2DHistogram();
	virtual ~vtkImage2DHistogram();
	
	vtkMultiThreader* Threader;
	int NumberOfThreads;

private:
	vtkImage2DHistogram operator=(const vtkImage2DHistogram&){} //not implemented
	vtkImage2DHistogram(const vtkImage2DHistogram&){} //not implemented

};

#endif