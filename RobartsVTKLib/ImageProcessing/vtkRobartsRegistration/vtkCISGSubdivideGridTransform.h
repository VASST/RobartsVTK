/*=========================================================================
 
  Library   : vtk/objects/library/CISgExtension
  Module    : $RCSfile: vtkCISGSubdivideGridTransform.h,v $
  Authors   : Thomas Hartkens <thomas@hartkens.de>
  Web       : http://www.image-registration.com
  Copyright : King's College London
              Div. of Radiological Sciences, 
              Computational Imaging Science Group, 1997 - 2000
        http://www-ipg.umds.ac.uk/cisg
  Date      : $Date: 2007/05/04 14:34:34 $
  Version   : $Revision: 1.1 $

This software is NOT free. Copyright: Thomas Hartkens
=========================================================================*/
// .NAME vtkCISGSubdivideGridTransform - create a grid for a vtkGridTransform
// .SECTION Description
// vtkCISGSubdivideGridTransform takes any transform as input and produces a grid
// for use by a vtkGridTransform.  This can be used, for example, to 
// invert a grid transform, concatenate two grid transforms, or to
// convert a thin plate spline transform into a grid transform.
// .SECTION See Also
// vtkGridTransform vtkThinPlateSplineTransform vtkAbstractTransform

#ifndef __vtkCISGSubdivideGridTransform_h
#define __vtkCISGSubdivideGridTransform_h

#include "vtkRobartsRegistrationModule.h"

#include "vtkImageAlgorithm.h"
#include "vtkImageData.h"
#include "vtkGridTransformBSpline.h"

class VTK_EXPORT vtkCISGSubdivideGridTransform : public vtkImageAlgorithm {

public:
  static vtkCISGSubdivideGridTransform *New();
  vtkTypeMacro(vtkCISGSubdivideGridTransform,vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set/Get the transform which will be converted into a grid.
  vtkSetObjectMacro(Input,vtkGridTransformBSpline);
  vtkGetObjectMacro(Input,vtkGridTransformBSpline);

protected:
  vtkCISGSubdivideGridTransform();
  ~vtkCISGSubdivideGridTransform();
  vtkCISGSubdivideGridTransform(const vtkCISGSubdivideGridTransform&) {};
  void operator=(const vtkCISGSubdivideGridTransform&) {};

  void ExecuteInformation();

  void Execute(vtkImageData *data);
  void Execute() { this->vtkImageAlgorithm::Execute(); };

  unsigned long GetMTime();

  vtkGridTransformBSpline *Input;
  vtkTimeStamp ShiftScaleTime;
};

#endif
