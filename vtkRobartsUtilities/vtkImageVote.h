#ifndef __VTKIMAGEVOTE_H__
#define __VTKIMAGEVOTE_H__

#include "vtkImageAlgorithm.h"
#include "vtkImageData.h"
#include "vtkImageCast.h"
#include "vtkTransform.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkSetGet.h"
#include <map>

#include <limits.h>

//INPUT PORT DESCRIPTION

//OUTPUT PORT DESCRIPTION

class vtkImageVote : public vtkImageAlgorithm 
{
public:
	vtkTypeMacro( vtkImageVote, vtkImageAlgorithm );

	static vtkImageVote *New();

	vtkDataObject* GetInput(int idx);
	void SetInput(int idx, vtkDataObject *input);

	vtkSetClampMacro(OutputDataType,int,1,11);
	vtkGetMacro(OutputDataType,int);

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

	template<class T>
	T GetMappedTerm(int i){ return (T)(BackwardsInputPortMapping.find(i) == BackwardsInputPortMapping.end() ? 0: BackwardsInputPortMapping[i]); }

protected:
	vtkImageVote();
	virtual ~vtkImageVote();

private:
	vtkImageVote operator=(const vtkImageVote&){}
	vtkImageVote(const vtkImageVote&){}
	
	int CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int* NumLabels, int* DataType);

	std::map<vtkIdType,int> InputPortMapping;
	std::map<int,vtkIdType> BackwardsInputPortMapping;
	int FirstUnusedPort;

	int OutputDataType;
};

#endif
