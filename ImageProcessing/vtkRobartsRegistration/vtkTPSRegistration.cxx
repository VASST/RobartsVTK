/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    $RCSfile: vtkTPSRegistration.cxx,v $
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
#include "vtkTPSRegistration.h"

//-----------------------`---------------------------------------------------
vtkTPSRegistration* vtkTPSRegistration::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkTPSRegistration");
  if(ret)
    {
    return (vtkTPSRegistration*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkTPSRegistration;
}

//----------------------------------------------------------------------------
vtkTPSRegistration::vtkTPSRegistration()
{
//   this->Alpha = 0.2;
//   this->Beta = 1.0;
//   this->Gamma = 0.15;
//   this->DecimateFactor = 0.0;
//   this->SubVolume[0] = 15;
//   this->SubVolume[1] = 9;
//   this->SubVolume[2] = 15;
//   this->Iterations[0] = 4;
//   this->Iterations[1] = 1;
//   this->Iterations[2] = 3;
//   this->Iterations[3] = 2;
//   this->Iterations[4] = 1;
//   this->Resolutions[0] = 50.0;
//   this->Resolutions[1] = 45.0;
//   this->Resolutions[2] = 30.0;
//   this->Resolutions[3] = 25.0;
//   this->Resolutions[4] = 20.0;
//   this->BinNumber[0] = 4096;
//   this->BinNumber[1] = 4096;
//   this->MaxIntensities[0] = 4095;
//   this->MaxIntensities[1] = 4095;
//   this->Metric = 0;
}

//----------------------------------------------------------------------------
vtkTPSRegistration::~vtkTPSRegistration()
{
  cout;
}

//----------------------------------------------------------------------------
void vtkTPSRegistration::SetInputData(vtkImageData *srcImage, vtkImageData *tgtImage,
              vtkPolyData *srcPoly, vtkPolyData *tgtPoly,
              vtkGeneralTransform *affTransform)
{
  cout;
}
// }

// //----------------------------------------------------------------------------
// vtkImageData *vtkTPSRegistration::GetSourceImage()
// {
//   //  return this->inData[0];
//   cout;
// }

// //----------------------------------------------------------------------------
// vtkImageData *vtkTPSRegistration::GetTargetImage()
// {
//   //  return this->inData[1];
//   cout;
// }

// //----------------------------------------------------------------------------
// void vtkTPSRegistration::SetBinNumber(int numS, int numT)
// {
//   cout;
// //   this->BinNumber[0] = numS;
// //   this->BinNumber[1] = numT;
// //   this->BinWidth[0] = (double)this->MaxIntensities[0] / ((double)this->BinNumber[0] - 1.0);
// //   this->BinWidth[1] = (double)this->MaxIntensities[1] / ((double)this->BinNumber[1] - 1.0);
// //   this->HistS = new long[numS];
// //   this->HistT = new long[numT];
// //   this->HistST = new long[numS*numT];
// }

// //----------------------------------------------------------------------------
// void vtkTPSRegistration::SetMaxIntensities(int maxS, int maxT)
// {
//   cout;
// //   this->MaxIntensities[0] = maxS;
// //   this->MaxIntensities[1] = maxT;
// //   this->BinWidth[0] = (double)this->MaxIntensities[0] / ((double)this->BinNumber[0] - 1.0);
// //   this->BinWidth[1] = (double)this->MaxIntensities[1] / ((double)this->BinNumber[1] - 1.0);
// }

// //----------------------------------------------------------------------------
// template <class T>
// void vtkTPSRegistrationExecute(vtkTPSRegistration *self,
//              T  *in1Ptr, T *in2Ptr,
//              int inc[3], int inc2[2], int inExt[6],
//              int loc000[3], int loc111[3], double count)
// {
//   cout;
// }

// //----------------------------------------------------------------------------
 void vtkTPSRegistration::PrintSelf(ostream& os, vtkIndent indent)
 {
   this->Superclass::PrintSelf(os,indent);

 //   os << indent << "Input 1: "    << this->inData[0]   << "\n";
 //   os << indent << "Input 2: "    << this->inData[1]   << "\n";
 //   os << indent << "BinWidth: ( " << this->BinWidth[0] << ", " << this->BinWidth[1]  << " )\n";
 //   os << indent << "BinNumber: ( "<< this->BinNumber[0]<< ", " << this->BinNumber[1] << " )\n";
 //   os << indent << "Extent: "     << this->Extent      << "\n";
 //   os << indent << "Result: "     << this->Result      << "\n";

}
