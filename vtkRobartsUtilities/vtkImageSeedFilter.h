/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkImageSeedFilter.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

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
	vtkSetClampMacro( NumberOfComponents, int, 1, 1000 );
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