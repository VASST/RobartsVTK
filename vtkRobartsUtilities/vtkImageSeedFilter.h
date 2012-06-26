#ifndef __vtkImageSeedFilter_H__
#define __vtkImageSeedFilter_H__

#include "vtkImageAlgorithm.h"
#include "vtkImageData.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"

#include <vector>

class vtkImageSeedFilter : public vtkImageAlgorithm {
public:

	vtkTypeMacro( vtkImageSeedFilter, vtkImageAlgorithm )

	static vtkImageSeedFilter *New();

	void Clear();

	void AddPointInVoxelCoordinates(double point[3], int component);
	void AddPointInVolumeCoordinates(double point[3], int component);
	
	//get/set the number of components (separate objects represented)
	vtkSetClampMacro( NumberOfComponents, int, 1, VTK_MAX_INT );
	vtkGetMacro( NumberOfComponents, int );

protected:
	
	int RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector);

	vtkImageSeedFilter();
	~vtkImageSeedFilter();

	int	NumberOfComponents;

private:
	vtkImageSeedFilter operator=(const vtkImageSeedFilter&){}
	vtkImageSeedFilter(const vtkImageSeedFilter&){}
	
	std::vector<double> pointsInVoxelX;
	std::vector<double> pointsInVoxelY;
	std::vector<double> pointsInVoxelZ;
	std::vector<int>	pointsInVoxelW;
	std::vector<double> pointsInVolumeX;
	std::vector<double> pointsInVolumeY;
	std::vector<double> pointsInVolumeZ;
	std::vector<int>	pointsInVolumeW;

};

#endif