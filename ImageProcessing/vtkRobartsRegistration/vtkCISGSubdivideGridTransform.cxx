/*=========================================================================

  Library   : vtk/objects/library/CISgExtension
  Module    : $RCSfile: vtkCISGSubdivideGridTransform.cxx,v $
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
#include "vtkCISGSubdivideGridTransform.h"
#include "vtkObjectFactory.h"

#if( VTK_MAJOR_VERSION >= 6 )
#include "vtkExecutive.h"
#include "vtkInformation.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#endif
//----------------------------------------------------------------------------
vtkCISGSubdivideGridTransform* vtkCISGSubdivideGridTransform::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkCISGSubdivideGridTransform");
  if(ret)
  {
    return (vtkCISGSubdivideGridTransform*)ret;
  }

  return new vtkCISGSubdivideGridTransform;
}

//----------------------------------------------------------------------------
vtkCISGSubdivideGridTransform::vtkCISGSubdivideGridTransform()
{
  this->Input = NULL;
}

//----------------------------------------------------------------------------
vtkCISGSubdivideGridTransform::~vtkCISGSubdivideGridTransform()
{
  this->SetInput(NULL);
}

//----------------------------------------------------------------------------
void vtkCISGSubdivideGridTransform::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkImageAlgorithm::PrintSelf(os,indent);
  os << indent << "Input: (" << this->Input << ")\n";
}

//----------------------------------------------------------------------------
// This method returns the largest data that can be generated.
void vtkCISGSubdivideGridTransform::ExecuteInformation()
{
  int dim[3];
  double spacing[3];
  double origin[3];

  if (this->GetInput() == NULL)
  {
    vtkErrorMacro("Missing input");
    return;
  }
  this->Input->Update();

  vtkGridTransformBSpline *input = this->GetInput();

  input->GetDisplacementGrid()->GetDimensions(dim);
  input->GetDisplacementGrid()->GetSpacing(spacing);
  input->GetDisplacementGrid()->GetOrigin(origin);

  // there are two additional control points at the border.
  int _x = dim[0] * 2 - 1;
  int _y = dim[1] * 2 - 1;
  int _z = dim[2] * 2 - 1;

  // Spacing of control points is reduced by factor 2
  spacing[0] = spacing[0] / 2.0;
  spacing[1] = spacing[1] / 2.0;
  spacing[2] = spacing[2] / 2.0;

#if( VTK_MAJOR_VERSION < 6 )
  this->GetOutput()->SetWholeExtent(0, _x - 1, 0, _y - 1, 0, _z - 1);
  this->GetOutput()->SetScalarType(input->GetDisplacementGrid()->GetScalarType());
  this->GetOutput()->SetNumberOfScalarComponents(3);
#else
  vtkInformation* outInfo = this->GetExecutive()->GetOutputInformation(0);
  int wholeExtent[6] = {0, _x - 1, 0, _y - 1, 0, _z - 1};
  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), wholeExtent, 6);
  this->GetOutput()->SetScalarType(input->GetDisplacementGrid()->GetScalarType(), outInfo);
  this->GetOutput()->SetNumberOfScalarComponents(3, outInfo);
#endif

  this->GetOutput()->SetSpacing(spacing);
  this->GetOutput()->SetOrigin(origin);
}

//----------------------------------------------------------------------------
void vtkCISGSubdivideGridTransform::Execute(vtkImageData *grid)
{
  int i, j, k, i1, j1, k1, i2, j2, k2;
  vtkImageData *inputgrid=this->GetInput()->GetDisplacementGrid();
  int *ext = grid->GetExtent();

  // Weights for subdivision
  double w[2][3];
  w[1][0] = 0;
  w[1][1] = 1.0/2.0;
  w[1][2] = 1.0/2.0;
  w[0][0] = 1.0/8.0;
  w[0][1] = 6.0/8.0;
  w[0][2] = 1.0/8.0;

  // pointer to the start address of the whole image
  double *gridPtr = (double *)grid->GetScalarPointer();
  for (i=0; i< grid->GetNumberOfPoints(); i++)
  {
    for (j=0; j<grid->GetNumberOfScalarComponents(); j++)
    {
      *gridPtr++=0.0;
    }
  }

  int *indim=inputgrid->GetDimensions();
  int *outdim=grid->GetDimensions();

  // there are two addiotional control points at the border.
  int _x = indim[0];
  int _y = indim[1];
  int _z = indim[2];

  double *dispPtr, dx,dy,dz;
  for (i = 0; i < _x; i++)
  {
    for (j = 0; j < _y; j++)
    {
      for (k = 0; k < _z; k++)
      {
        for (i1 = 0; i1 < 2; i1++)
        {
          for (j1 = 0; j1 < 2; j1++)
          {
            for (k1 = 0; k1 < 2; k1++)
            {

              dx=0.0;
              dy=0.0;
              dz=0.0;
              for (i2 = 0; i2 < 3; i2++)
              {
                for (j2 = 0; j2 < 3; j2++)
                {
                  for (k2 = 0; k2 < 3; k2++)
                  {
                    if (i+i2-1 >= 0 && i+i2-1 < indim[0] &&
                        j+j2-1 >= 0 && j+j2-1 < indim[1] &&
                        k+k2-1 >= 0 && k+k2-1 < indim[2])
                    {
                      dispPtr=(double *)
                              inputgrid->GetScalarPointer(i+i2-1,j+j2-1,k+k2-1);
                      dx += w[i1][i2]*w[j1][j2]*w[k1][k2] * (*dispPtr++);
                      dy += w[i1][i2]*w[j1][j2]*w[k1][k2] * (*dispPtr++);
                      dz += w[i1][i2]*w[j1][j2]*w[k1][k2] * (*dispPtr++);
                    }
                  }
                }
              }
              if (2*i+i1 < outdim[0] && 2*j+j1 < outdim[1] && 2*k+k1 < outdim[2])
              {
                dispPtr=(double *)grid->GetScalarPointer(2*i+i1,2*j+j1,2*k+k1);
                *dispPtr++ = dx;
                *dispPtr++ = dy;
                *dispPtr++ = dz;
              }
            }
          }
        }
      }
    }
  }

}


//----------------------------------------------------------------------------
unsigned long vtkCISGSubdivideGridTransform::GetMTime()
{
  unsigned long mtime = this->vtkImageAlgorithm::GetMTime();

  if (this->Input)
  {
    unsigned long mtime2 = this->Input->GetMTime();
    if (mtime2 > mtime)
    {
      mtime = mtime2;
    }
  }

  return mtime;
}
