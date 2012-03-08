#ifndef VTKCUDA2DTRANSFERCLASSIFICATIONFUNCTION_H
#define VTKCUDA2DTRANSFERCLASSIFICATIONFUNCTION_H

#include "vtkCudaFunctionObject.h"
#include <vector>

class vtkCuda2DTransferClassificationFunction : public vtkObject {
public:
	static vtkCuda2DTransferClassificationFunction*	New();

	void GetClassifyTable(	short* outputTable, int sizeI, int sizeG,
							float lowI, float highI, float lowG, float highG);
	void GetTransferTable(	float* outputRTable, float* outputGTable, float* outputBTable, float* outputATable,
							int sizeI, int sizeG, float lowI, float highI, float lowG, float highG);

	short GetNumberOfClassifications();

	void AddFunctionObject(vtkCudaFunctionObject* object);
	void RemoveFunctionObject(vtkCudaFunctionObject* object);

	bool NeedsUpdate();
	void SignalUpdate();
	void SatisfyUpdate();

protected:
	vtkCuda2DTransferClassificationFunction();
	~vtkCuda2DTransferClassificationFunction();

private:
	std::vector<vtkCudaFunctionObject*>* components;
	bool updateNeeded;

};

#endif