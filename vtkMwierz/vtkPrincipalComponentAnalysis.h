/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkPrincipalComponentAnalysis.h,v $
  Language:  C++
  Date:      $Date: 2007/05/04 14:34:35 $
  Version:   $Revision: 1.1 $

  Copyright (c) 1993-2002 Ken Martin, Will Schroeder, Bill Lorensen 
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkPrincipalComponentAnalysis - performs common math operations
// .SECTION Description
// vtkPrincipalComponentAnalysis is provides methods to perform common math operations. These 
// include providing constants such as Pi; conversion from degrees to 
// radians; vector operations such as dot and cross products and vector 
// norm; matrix determinant for 2x2 and 3x3 matrices; and random 
// number generation.

#ifndef __vtkPrincipalComponentAnalysis_h
#define __vtkPrincipalComponentAnalysis_h

#include "vtkObject.h"
#include "vtkImageData.h"
#include "vtkDoubleArray.h"
#include "vtkPoints.h"

// This is the maximum number of images fitable.
#define MAX_M 40

class VTK_EXPORT vtkPrincipalComponentAnalysis : public vtkObject
{
public:
  static vtkPrincipalComponentAnalysis *New();
  vtkTypeRevisionMacro(vtkPrincipalComponentAnalysis,vtkObject);
  
  // Add an image to fit
  virtual void AddImage(vtkImageData *image);

  // Mask image to calculate PCA over
  virtual void SetMask(vtkImageData *mask);

  // Set/Get the mask to consider when calculating a new image
  // from PCA based on weights
  virtual void SetMaskPoints(vtkPoints *maskPoints);
  vtkPoints *GetMaskPoints();

  // Calculate the mean image and the eigenvectors
  virtual void Fit();

  // Set the number of modes to keep
  vtkSetMacro(NumberOfModes, int);
  vtkGetMacro(NumberOfModes, int);
  int NumberOfModes;

  // Get an output image for particular weights, or get the
  // weights given a particular image.
  vtkImageData *GetOutput(vtkDoubleArray *weights);
  vtkDoubleArray *GetWeightsForImage(vtkImageData *image);

  // Set/Get the eigenvectors and mean image results
  vtkImageData *GetEigenVectorsImage();
  vtkImageData *GetMeanIntensitiesImage();
  virtual void SetEigenVectorsImage(vtkImageData *EVI);
  virtual void SetMeanIntensitiesImage(vtkImageData *MII);

protected:
  vtkPrincipalComponentAnalysis();
  ~vtkPrincipalComponentAnalysis() {};

  vtkImageData *Images[MAX_M];
  vtkImageData *MaskImage;
  vtkImageData *OutputImage;
  vtkPoints *MaskPoints;
  int M;
  int N;
  double **EigenVectors;
  double **MeanIntensities;

  int ext[6];
  double spa[3];
  double ori[3];
  
private:
  vtkPrincipalComponentAnalysis(const vtkPrincipalComponentAnalysis&);  // Not implemented.
  void operator=(const vtkPrincipalComponentAnalysis&);  // Not implemented.
};

#endif
