/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    $RCSfile: vtkTPSRegistration.h,v $
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
// .NAME vtkTPSRegistration -

#ifndef __vtkTPSRegistration_h
#define __vtkTPSRegistration_h

#include "vtkRobartsRegistrationExport.h"

#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkPolyData.h"
#include "vtkGeneralTransform.h"
#include "vtkIndent.h"

class vtkRobartsRegistrationExport vtkTPSRegistration : public vtkObject
{
public:
  static vtkTPSRegistration *New();
  vtkTypeMacro(vtkTPSRegistration,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set/get the 2 input images.
  virtual void SetInputData(vtkImageData *srcImage, vtkImageData *tgtImage,
                            vtkPolyData *srcPoly, vtkPolyData *tgtPoly,
                            vtkGeneralTransform *affTransform);
  //SourceImage(vtkImageData *input);
//   virtual void SetTargetImage(vtkImageData *input);
//   virtual void SetSourcePolyData(vtkPolyData *input);
//   virtual void SetTargetPolyData(vtkPolyData *input);
//   virtual void SetAffineTransform(vtkGeneralTransform *input);
//   vtkImageData *GetSourceImage();
//   vtkImageData *GetTargetImage();
//   vtkPolyData *GetSourcePolyData();
//   vtkPolyData *GetTargetPolyData();
//   vtkGeneralTransform *GetAffineTransform();

  // Description:
  // Select NMI (Metric = 0), MI (Metric = 1), or ECR (Metric = 2)
//   vtkSetMacro(Metric, int);
//   vtkGetMacro(Metric, int);
//   int Metric;

//   // Description:
//   // Select NMI (Metric = 0), MI (Metric = 1), or ECR (Metric = 2)
//   vtkSetMacro(Alpha, double);
//   vtkGetMacro(Alpha, double);
//   vtkSetMacro(Beta, double);
//   vtkGetMacro(Beta, double);
//   vtkSetMacro(Gamma, double);
//   vtkGetMacro(Gamma, double);
//   double Alpha;
//   double Beta;
//   double Gamma;

//   // Description:
//   // Select NMI (Metric = 0), MI (Metric = 1), or ECR (Metric = 2)
//   vtkSetMacro(DecimateFactor, double);
//   vtkGetMacro(DecimateFactor, double);
//   double DecimateFactor;

//   // Description:
//   // Numbers of bins to use, and number of intensities per bin.
//   // I assume images have intensities >= 0.
//   // will create bins of width
//   virtual void SetIterations(int iter1, int iter2, int iter3, int iter4, int iter5);
//   virtual void SetResolutions(double dist1, double dist2, double dist3, double dist4, double dist5);
//   virtual void SetImageSubVolume(int vol1, int vol2, int vol3);
//   int Iterations[5];
//   double Resolutions[5];
//   int SubVolume[3];

//   // Description:
//   // Numbers of bins to use, and number of intensities per bin.
//   // I assume images have intensities >= 0.
//   // will create bins of width
//   virtual void SetBinNumber(int numS, int numT);
//   virtual void SetMaxIntensities(int maxS, int maxT);
//   int BinNumber[2];
//   int MaxIntensities[2];

protected:
  vtkTPSRegistration();
  ~vtkTPSRegistration();

//   double origin[3];

};

#endif













