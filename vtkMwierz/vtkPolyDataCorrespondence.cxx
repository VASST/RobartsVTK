/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkPolyDataCorrespondence.cxx,v $
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

#include "vtkPolyDataCorrespondence.h"

vtkCxxRevisionMacro(vtkPolyDataCorrespondence, "$Revision: 1.1 $");
vtkStandardNewMacro(vtkPolyDataCorrespondence);

//------------------------------------------------------------------------------
#if VTK3
vtkPolyDataCorrespondence* vtkPolyDataCorrespondence::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkPolyDataCorrespondence");
  if(ret)
    {
    return (vtkPolyDataCorrespondence*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkPolyDataCorrespondence;
}
#endif

//----------------------------------------------------------------------------

vtkPolyDataCorrespondence::vtkPolyDataCorrespondence()
{
  this->ConstantN = 0.0;
  this->ConstantS = 0.0;
  this->ConstantC = 0.0;
  this->SliceAxis = 2;
  this->PrintOutput = 0;
  this->MeanDistance = 0.0;
  this->RMSDistance = 0.0;

  Distances = vtkDoubleArray::New();
  Distances->SetName("Distances"); 
  Distances->SetNumberOfComponents(1);

  Pairings = vtkLongArray::New();
  Pairings->SetName("Pairings"); 
  Pairings->SetNumberOfComponents(1);
}

//----------------------------------------------------------------------------
void vtkPolyDataCorrespondence::SetInput1(vtkPolyData *input)
{
  this->poly1 = input;
}

//----------------------------------------------------------------------------
void vtkPolyDataCorrespondence::SetInput2(vtkPolyData *input)
{
  this->poly2 = input;
}

//----------------------------------------------------------------------------
void vtkPolyDataCorrespondence::GetAxialCorrespondenceStats()
{
  double x,y,z,x1,y1,z1,minMagn,magn;
  double minMagnX,minMagnY,minMagnZ;
  double aveX,aveY,aveZ;
  double minX,minY,minZ;
  double maxX,maxY,maxZ;
  double RMSX,RMSY,RMSZ;

  vtkPoints *points1 = this->poly1->GetPoints();
  vtkPoints *points2 = this->poly2->GetPoints();
  int n1 = points1->GetNumberOfPoints();
  int n2 = points2->GetNumberOfPoints();

  minX = 1E300;
  minY = 1E300;
  minZ = 1E300;
  maxX = 0.0;
  maxY = 0.0;
  maxZ = 0.0;
  RMSX = 0.0;
  RMSY = 0.0;
  RMSZ = 0.0;
  aveX = 0.0;
  aveY = 0.0;
  aveZ = 0.0;
  for (int i = 0; i < n1; i++)
    {
      x1 = points1->GetPoint(i)[0];
      y1 = points1->GetPoint(i)[1];
      z1 = points1->GetPoint(i)[2];
      minMagn = 1E300;
      for (int j = 0; j < n2; j++)
	{
	  x = x1 - points2->GetPoint(j)[0];
	  y = y1 - points2->GetPoint(j)[1];
	  z = z1 - points2->GetPoint(j)[2];

	  magn = x*x + y*y + z*z;

	  if (magn < minMagn) 
	    {
	      minMagn = magn;
	      minMagnX = fabs(x);
	      minMagnY = fabs(y);
	      minMagnZ = fabs(z);
	    }
	}

      aveX = aveX + minMagnX;
      aveY = aveY + minMagnY;
      aveZ = aveZ + minMagnZ;

      if (minMagnX < minX) minX = minMagnX;
      if (minMagnY < minY) minY = minMagnY;
      if (minMagnZ < minZ) minZ = minMagnZ;
      if (minMagnX > maxX) maxX = minMagnX;
      if (minMagnY > maxY) maxY = minMagnY;
      if (minMagnZ > maxZ) maxZ = minMagnZ;
      RMSX = RMSX + minMagnX * minMagnX;
      RMSY = RMSY + minMagnY * minMagnY;
      RMSZ = RMSZ + minMagnZ * minMagnZ;
    }

  aveX = aveX / n1;
  aveY = aveY / n1;
  aveZ = aveZ / n1;

  RMSX = sqrt(RMSX / n1);
  RMSY = sqrt(RMSY / n1);
  RMSZ = sqrt(RMSZ / n1);

  if (this->PrintOutput)
    {
      cout << "\n-------------------------------------------------------------\n";
      cout << " Ave of Difference Magnitude = " << aveX << "," << aveY << "," << aveZ << "\n";
      cout << " Min of Difference Magnitude = " << minX << "," << minY << "," << minZ << "\n";
      cout << " Max of Difference Magnitude = " << maxX << "," << maxY << "," << maxZ << "\n";
      cout << " RMS of Difference Magnitude = " << RMSX << "," << RMSY << "," << RMSZ << "\n";
      cout << " Total Number of Points = " << n1 << "\n";
      cout << "-------------------------------------------------------------\n\n";
    }

};
//----------------------------------------------------------------------------
void vtkPolyDataCorrespondence::Update2D()
{
  double pnt1[3],pnt1a[3],pnt1b[3],pnt2[3],pnt2a[3],pnt2b[3];
  double vect1a[3],vect1b[3],vect2a[3],vect2b[3],dotProd;
  double minMagn,magn;
  double cost,minCost;
  double pair;
  double ave = 0.0,std = 0.0,min = 1E300,max = 0.0,RMS = 0.0;
  int big = 0;
  int i,j,k;

  vtkPoints *points1 = this->poly1->GetPoints();
  vtkPoints *points2 = this->poly2->GetPoints();
  int n1 = points1->GetNumberOfPoints();
  int n2 = points2->GetNumberOfPoints();

  this->Distances->SetNumberOfTuples(n1);
  this->Pairings->SetNumberOfTuples(n1);

  // CASE 1: All constants are 0, so do simple magnitude minimization calcultion.
  if ( (this->ConstantN == 0.0) && (this->ConstantS == 0) && (this->ConstantC == 0) )
    {
      for (i = 0; i < n1; i++)
	{
	  minMagn = 1E300;
	  minCost = 1E300;
	  memcpy(pnt1, points1->GetPoint(i), sizeof(double)*3);

	  for (j = 0; j < n2; j++)
	    {
	      memcpy(pnt2, points2->GetPoint(j), sizeof(double)*3);
	      magn = 0.0;
	      for (k = 0; k < 3; k++)
		{
		  if (k != this->SliceAxis)
		    {
		      magn = magn + (pnt1[k] - pnt2[k]) * (pnt1[k] - pnt2[k]);
		    }
		}
	      magn = sqrt(magn);

	      cost = (magn);      
	      if (cost < minCost)
		{
		  minCost = cost;
		  minMagn = magn;
		  pair = j;
		}
	    }

	  this->Distances->SetTuple1(i,minMagn);
	  this->Pairings->SetTuple1(i,pair);
	  ave = ave + minMagn;
	  if (minMagn < min) min = minMagn;
	  if (minMagn > max) max = minMagn;
	  RMS = RMS + minMagn * minMagn;
	  if (minMagn > 3) big = big + 1;
	}
    }

  // CASE 2: Do the complete calculation.
  else
    {
      for (i = 0; i < n1; i++)
	{
	  minMagn = 1E300;
	  minCost = 1E300;

	  memcpy(pnt1, points1->GetPoint(i), sizeof(double)*3);
	  if (i+1 < n1) 
	    {
	      memcpy(pnt1a, points1->GetPoint(i+1 ), sizeof(double)*3);
	    }
	  else
	    {
	      memcpy(pnt1a, points1->GetPoint( 0  ), sizeof(double)*3);
	    }
	  if (i-1 >  0)
	    { 
	      memcpy(pnt1b, points1->GetPoint(i-1 ), sizeof(double)*3);
	    }
	  else
	    {
	      memcpy(pnt1b, points1->GetPoint(n1-1), sizeof(double)*3);
	    }
	  vect1a[0] = pnt1a[0] - pnt1[0];
	  vect1a[1] = pnt1a[1] - pnt1[1];
	  vect1a[2] = pnt1a[2] - pnt1[2];
	  vect1b[0] = pnt1[0] - pnt1b[0];
	  vect1b[1] = pnt1[1] - pnt1b[1];
	  vect1b[2] = pnt1[2] - pnt1b[2];

	  for (j = 0; j < n2; j++)
	    {

	      memcpy(pnt2, points2->GetPoint(j), sizeof(double)*3);
	      if (j+1 < n2)
		{
		  memcpy(pnt2a, points2->GetPoint(j+1 ), sizeof(double)*3);
		}
	      else
		{
		  memcpy(pnt2a, points2->GetPoint( 0  ), sizeof(double)*3);
		}
	      if (j-1 >  0) 
		{
		  memcpy(pnt2b, points2->GetPoint(j-1 ), sizeof(double)*3);
		}
	      else
		{
		  memcpy(pnt2b, points2->GetPoint(n2-1), sizeof(double)*3);
		}
	      vect2a[0] = pnt2a[0] - pnt2[0];
	      vect2a[1] = pnt2a[1] - pnt2[1];
	      vect2a[2] = pnt2a[2] - pnt2[2];
	      vect2b[0] = pnt2[0] - pnt2b[0];
	      vect2b[1] = pnt2[1] - pnt2b[1];
	      vect2b[2] = pnt2[2] - pnt2b[2];
	      
	      dotProd = ( vect1a[0]*vect2a[0] + vect1a[1]*vect2a[1] + vect1a[2]*vect2a[2] +
			  vect1b[0]*vect2b[0] + vect1b[1]*vect2b[1] + vect1b[2]*vect2b[2] );

	      magn = 0.0;
	      for (k = 0; k < 3; k++)
		{
		  if (k != this->SliceAxis)
		    {
		      magn = magn + (pnt1[k] - pnt2[k]) * (pnt1[k] - pnt2[k]);
		    }
		}

	      magn = sqrt(magn);

	      cost = (magn + this->ConstantN * (1.0 - dotProd/2.0));

	      if (cost < minCost)
		{
		  minCost = cost;
		  minMagn = magn;
		  pair = j;
		}
	    }

	  this->Distances->SetTuple1(i,minMagn);
	  this->Pairings->SetTuple1(i,pair);
	  ave = ave + minMagn;
	  if (minMagn < min) min = minMagn;
	  if (minMagn > max) max = minMagn;
	  RMS = RMS + minMagn * minMagn;
	  if (minMagn > 3) big = big + 1;
	}
    }

  ave = ave / n1;
  RMS = sqrt(RMS / n1);
  
  this->MeanDistance = ave;
  this->RMSDistance = RMS;

  for (int iGN = 0; iGN < n1; iGN++)
    {
      magn = this->Distances->GetTuple1(iGN);
      std = std + (magn - ave) * (magn - ave);
    }
  std = sqrt(std / (n1 - 1));

  if (this->PrintOutput)
    {
      cout << "\n---------------------------------------\n";
      cout << " Ave of Difference Magnitude = " << ave << "\n";
      cout << " STD of Difference Magnitude = " << std << "\n";
      cout << " Min of Difference Magnitude = " << min << "\n";
      cout << " Max of Difference Magnitude = " << max << "\n";
      cout << " RMS of Difference Magnitude = " << RMS << "\n";
      cout << " Number of Difference Magnitude > 3 = " << big << "\n";
      cout << " Total Number of Points = " << n1 << "\n";
      cout << "---------------------------------------\n\n";
    }
}

//----------------------------------------------------------------------------
void vtkPolyDataCorrespondence::Update3D()
{
  double pnt1[3],pnt2[3],norm1[3],norm2[3],curv1[2],curv2[2];
  double minMagn,magn,normdot;
  double temp1,temp2,shape1,shape2,curved1,curved2;
  double cost,minCost;
  double pair;
  double ave = 0.0,std = 0.0,min = 1E300,max = 0.0,RMS = 0.0;
  int big = 0;
  int i,j;

  vtkPoints *points1 = this->poly1->GetPoints();
  vtkPoints *points2 = this->poly2->GetPoints();
  int n1 = points1->GetNumberOfPoints();
  int n2 = points2->GetNumberOfPoints();

  this->Distances->SetNumberOfTuples(n1);
  this->Pairings->SetNumberOfTuples(n1);

  // CASE 1: All constants are 0, so do simple magnitude minimization calcultion.
  if ( (this->ConstantN == 0.0) && (this->ConstantS == 0) && (this->ConstantC == 0) )
    {
      for (i = 0; i < n1; i++)
	{
	  minMagn = 1E300;
	  minCost = 1E300;
	  memcpy(pnt1, points1->GetPoint(i), sizeof(double)*3);

	  for (j = 0; j < n2; j++)
	    {
	      memcpy(pnt2, points2->GetPoint(j), sizeof(double)*3);
	      magn = sqrt( (pnt1[0] - pnt2[0]) * (pnt1[0] - pnt2[0]) +
			   (pnt1[1] - pnt2[1]) * (pnt1[1] - pnt2[1]) +
			   (pnt1[2] - pnt2[2]) * (pnt1[2] - pnt2[2]) );

	      cost = (magn);
	      
	      if (cost < minCost)
		{
		  minCost = cost;
		  minMagn = magn;
		  pair = j;
		}
	    }

	  this->Distances->SetTuple1(i,minMagn);
	  this->Pairings->SetTuple1(i,pair);
	  ave = ave + minMagn;
	  if (minMagn < min) min = minMagn;
	  if (minMagn > max) max = minMagn;
	  RMS = RMS + minMagn * minMagn;
	  if (minMagn > 3) big = big + 1;
	}
    }

  // CASE 2: As above but with the addition of the normals term.
  else if ( (this->ConstantN != 0.0) && (this->ConstantS == 0) && (this->ConstantC == 0) )
    {
      vtkDataArray *normals1 = this->poly1->GetPointData()->GetNormals();
      vtkDataArray *normals2 = this->poly2->GetPointData()->GetNormals();

      for (i = 0; i < n1; i++)
	{
	  minMagn = 20.0;
	  minCost = 1E300;
	  memcpy(pnt1, points1->GetPoint(i), sizeof(double)*3);
	  memcpy(norm1, normals1->GetTuple3(i), sizeof(double)*3);

	  for (j = 0; j < n2; j++)
	    {
	      memcpy(pnt2, points2->GetPoint(j), sizeof(double)*3);
	      magn = sqrt( (pnt1[0] - pnt2[0]) * (pnt1[0] - pnt2[0]) +
			   (pnt1[1] - pnt2[1]) * (pnt1[1] - pnt2[1]) +
			   (pnt1[2] - pnt2[2]) * (pnt1[2] - pnt2[2]) );
	      if (magn < 20.0)
		{
		  memcpy(norm2, normals2->GetTuple3(j), sizeof(double)*3);
		  normdot = norm1[0]*norm2[0] + norm1[1]*norm2[1] + norm1[2]*norm2[2];

		  cost = ( magn + this->ConstantN * (1.0 - normdot) );
	      
		  if (cost < minCost)
		    {
		      minCost = cost;
		      minMagn = magn;
		      pair = j;
		    }
		}
	    }

	  this->Distances->SetTuple1(i,minMagn);
	  this->Pairings->SetTuple1(i,pair);
	  ave = ave + minMagn;
	  if (minMagn < min) min = minMagn;
	  if (minMagn > max) max = minMagn;
	  RMS = RMS + minMagn * minMagn;
	  if (minMagn > 3) big = big + 1;
	}
    }

  // CASE 3: Do the complete calculation.
  else
    {
      vtkDataArray *normals1 = this->poly1->GetPointData()->GetNormals();
      vtkDataArray *normals2 = this->poly2->GetPointData()->GetNormals();

      vtkCurvatures *gausCurv1 = vtkCurvatures::New();
      vtkCurvatures *meanCurv1 = vtkCurvatures::New();
      vtkCurvatures *gausCurv2 = vtkCurvatures::New();
      vtkCurvatures *meanCurv2 = vtkCurvatures::New();
  
      gausCurv1->SetCurvatureTypeToGaussian();
      gausCurv1->SetInput(this->poly1);
      gausCurv1->Update();
      meanCurv1->SetCurvatureTypeToMean();
      meanCurv1->SetInput(this->poly1);
      meanCurv1->InvertMeanCurvatureOn();
      meanCurv1->Update();
      gausCurv2->SetCurvatureTypeToGaussian();
      gausCurv2->SetInput(this->poly2);
      gausCurv2->Update();
      meanCurv2->SetCurvatureTypeToMean();
      meanCurv2->SetInput(this->poly2);
      meanCurv2->InvertMeanCurvatureOn();
      meanCurv2->Update();
      
      for (i = 0; i < n1; i++)
	{
	  minMagn = 20.0;
	  minCost = 1E300;
	  memcpy(pnt1, points1->GetPoint(i), sizeof(double)*3);
	  memcpy(norm1, normals1->GetTuple3(i), sizeof(double)*3);
	  curv1[0] = gausCurv1->GetOutput()->GetPointData()->GetScalars()->GetTuple1(i);
	  curv1[1] = meanCurv1->GetOutput()->GetPointData()->GetScalars()->GetTuple1(i);

	  for (j = 0; j < n2; j++)
	    {
	      memcpy(pnt2, points2->GetPoint(j), sizeof(double)*3);
	      magn = sqrt( (pnt1[0] - pnt2[0]) * (pnt1[0] - pnt2[0]) +
			   (pnt1[1] - pnt2[1]) * (pnt1[1] - pnt2[1]) +
			   (pnt1[2] - pnt2[2]) * (pnt1[2] - pnt2[2]) );
	      if (magn < 20.0)
		{
		  memcpy(norm2, normals2->GetTuple3(j), sizeof(double)*3);
		  normdot = norm1[0]*norm2[0] + norm1[1]*norm2[1] + norm1[2]*norm2[2];
		  curv2[0]=gausCurv2->GetOutput()->GetPointData()->GetScalars()->GetTuple1(j);
		  curv2[1]=meanCurv2->GetOutput()->GetPointData()->GetScalars()->GetTuple1(j);
		  
		  temp1 = curv1[1]*curv1[1] - curv1[0];
		  if (temp1 > 0.0)
		    {
		      shape1 = 0.636619772368 * atan( -curv1[1] / sqrt(temp1) );
		    }
		  else
		    {
		      if (curv1[1] <= 0.0) shape1 = 1.0;
		      if (curv1[1] > 0.0) shape1 = -1.0;
		    }
		  
		  temp2 = curv2[1]*curv2[1] - curv2[0];
		  if (temp2 > 0.0)
		    {
		      shape2 = 0.636619772368 * atan( -curv2[1] / sqrt(temp2) );
		    }
		  else
		    {
		      if (curv2[1] <= 0.0) shape2 = 1.0;
		      if (curv2[1] > 0.0) shape2 = -1.0;
		    }
		  
		  curved1 = 0.0;
		  temp1 = 2.0 * curv1[1] * curv1[1] - curv1[0];
		  if (temp1 > 0) curved1 = sqrt(temp1);
		  curved2 = 0.0;
		  temp2 = 2.0 * curv2[1] * curv2[1] - curv2[0];
		  if (temp2 > 0) curved2 = sqrt(temp2);
		  
		  cost = ( magn + 
			   this->ConstantN * (1.0 - normdot) +
			   this->ConstantS * fabs(shape1 - shape2) +
			   this->ConstantC * fabs(curved1 - curved2) );
	      
		  if (cost < minCost)
		    {
		      minCost = cost;
		      minMagn = magn;
		      pair = j;
		    }
		}
	    }

	  this->Distances->SetTuple1(i,minMagn);
	  this->Pairings->SetTuple1(i,pair);
	  ave = ave + minMagn;
	  if (minMagn < min) min = minMagn;
	  if (minMagn > max) max = minMagn;
	  RMS = RMS + minMagn * minMagn;
	  if (minMagn > 3) big = big + 1;
	
	}
    }

  ave = ave / n1;
  RMS = sqrt(RMS / n1);
  
  this->MeanDistance = ave;
  this->RMSDistance = RMS;

  for (int iGN = 0; iGN < n1; iGN++)
    {
      magn = this->Distances->GetTuple1(i);
      std = std + (magn - ave) * (magn - ave);
    }
  std = sqrt(std / (n1 - 1));

  if (this->PrintOutput)
    {
      cout << "\n---------------------------------------\n";
      cout << " Ave of Difference Magnitude = " << ave << "\n";
      cout << " STD of Difference Magnitude = " << std << "\n";
      cout << " Min of Difference Magnitude = " << min << "\n";
      cout << " Max of Difference Magnitude = " << max << "\n";
      cout << " RMS of Difference Magnitude = " << RMS << "\n";
      cout << " Number of Difference Magnitude > 3 = " << big << "\n";
      cout << " Total Number of Points = " << n1 << "\n";
      cout << "---------------------------------------\n\n";
    }

};

//----------------------------------------------------------------------------
void vtkPolyDataCorrespondence::Update3DFast()
{
  double pnt1[3],pnt2[3];
  double minMagnSqrd,magnSqrd,RMS = 0.0;
  int i,j;

  vtkPoints *points1 = this->poly1->GetPoints();
  vtkPoints *points2 = this->poly2->GetPoints();
  int n1 = points1->GetNumberOfPoints();
  int n2 = points2->GetNumberOfPoints();

  for (i = 0; i < n1; i++)
    {
      minMagnSqrd = 1E300;
      memcpy(pnt1, points1->GetPoint(i), sizeof(double)*3);

      for (j = 0; j < n2; j++)
	{
	  memcpy(pnt2, points2->GetPoint(j), sizeof(double)*3);
	  magnSqrd = ( (pnt1[0] - pnt2[0]) * (pnt1[0] - pnt2[0]) +
		       (pnt1[1] - pnt2[1]) * (pnt1[1] - pnt2[1]) +
		       (pnt1[2] - pnt2[2]) * (pnt1[2] - pnt2[2]) );
	  
	  if (magnSqrd < minMagnSqrd) minMagnSqrd = magnSqrd;
	}

      RMS = RMS + minMagnSqrd;
    }
 
  this->RMSDistance = sqrt(RMS / n1);

};

//----------------------------------------------------------------------------
vtkDoubleArray *vtkPolyDataCorrespondence::GetDistances()
{
  return this->Distances;
}

//----------------------------------------------------------------------------
vtkLongArray *vtkPolyDataCorrespondence::GetPairings()
{
  return this->Pairings;
}

//----------------------------------------------------------------------------
double vtkPolyDataCorrespondence::GetMeanDistance()
{
  return this->MeanDistance;
}

//----------------------------------------------------------------------------
double vtkPolyDataCorrespondence::GetRMSDistance()
{
  return this->RMSDistance;
}

//----------------------------------------------------------------------------
void vtkPolyDataCorrespondence::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkPolyDataToPolyDataFilter::PrintSelf(os,indent);
}

