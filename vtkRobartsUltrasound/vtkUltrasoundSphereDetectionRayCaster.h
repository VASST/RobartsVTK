/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkUltrasoundSphereDetectionRayCaster.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkUltrasoundSphereDetectionRayCaster - Helper for automatic US
// sphere detection
// .SECTION Description
// vtkUltrasoundSphereDetectionRayCaster Casts rays out from a seed point
// to find two points for each ray in the canny edge detection image (one
// for the inside edge and one for the outside edge) that either 1) have
// the greatest intensity and are next to a zero pixel or 2) have the
// the greatest intensity and are next to a pixel whos intensity is less
// than the pixel by a certain threshold.  Input and output must have one
// scalar component

#ifndef __vtkUltrasoundSphereDetectionRayCaster_h
#define __vtkUltrasoundSphereDetectionRayCaster_h

#include "vtkImageAlgorithm.h"
#include "vtkImageData.h" // makes things a bit easier

class VTK_EXPORT vtkUltrasoundSphereDetectionRayCaster : public vtkImageAlgorithm
{
public:
  static vtkUltrasoundSphereDetectionRayCaster *New();
  vtkTypeRevisionMacro(vtkUltrasoundSphereDetectionRayCaster,vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);
  
  // Description:
  // Set the canny edge detection input
  void SetCannyInput (vtkImageData *input) {this->SetInput(0,input);};

  // Description:
  // Set the seed point (should be approximately in the center of the sphere),
  // in world coordinates
  vtkSetVector3Macro(Seed, double);
  vtkGetVectorMacro(Seed, double, 3);

  // Description:
  // Set the intensity threshold between a pixel being considered and its neighbor
  // along the ray
  vtkSetMacro(IntensityDifferenceThreshold, double);
  vtkGetMacro(IntensityDifferenceThreshold, double);

  // Description:
  // Set the angle increments (in degrees)
  vtkSetMacro(AngleIncrement, double);
  vtkGetMacro(AngleIncrement, double);

  // Description:
  // Set the minimum distance from the sphere edge to the seed (in world coordinates)
  vtkSetMacro(MinimumDistanceFromSeed, double);
  vtkGetMacro(MinimumDistanceFromSeed, double);

  // Description:
  // Set the maximum distance from the sphere edge to the seed (in world coordinates)
  vtkSetMacro(MaximumDistanceFromSeed, double);
  vtkGetMacro(MaximumDistanceFromSeed, double);

  // Description:
  // Set the replacement intensity value for the edges
  vtkSetMacro(EdgeValue, double);
  vtkGetMacro(EdgeValue, double);

  // Description:
  // Get the mean distance from the edge points to the seed
  //vtkGetMacro(MeanDistance, double);

  // Description:
  // Set the dimensionality
  vtkGetMacro(Dimensionality, int);
 
protected:
  vtkUltrasoundSphereDetectionRayCaster();
  ~vtkUltrasoundSphereDetectionRayCaster() {};

  double Seed[3];
  double IntensityDifferenceThreshold;
  double AngleIncrement;
  double MinimumDistanceFromSeed;
  double MaximumDistanceFromSeed;
  double EdgeValue;
  //double MeanDistance;

  int Dimensionality; // considering 3D images only for now
  
  virtual int RequestData(vtkInformation *,
                          vtkInformationVector **,
                          vtkInformationVector *);
  
private:
  vtkUltrasoundSphereDetectionRayCaster(const vtkUltrasoundSphereDetectionRayCaster&);  // Not implemented.
  void operator=(const vtkUltrasoundSphereDetectionRayCaster&);  // Not implemented.
};

#endif



