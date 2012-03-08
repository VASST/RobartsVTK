/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMeshSmootheness.cxx,v $
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
#include "vtkMeshSmootheness.h"

#include "vtkCellArray.h"
#include "vtkDoubleArray.h"
#include "vtkFieldData.h"
#include "vtkFloatArray.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkPolyDataNormals.h"
#include "vtkPolygon.h"
#include "vtkTensor.h"
#include "vtkTriangle.h"

vtkCxxRevisionMacro(vtkMeshSmootheness, "$Revision: 1.1 $");
vtkStandardNewMacro(vtkMeshSmootheness);

//------------------------------------------------------------------------------
#if VTK3
vtkMeshSmootheness* vtkMeshSmootheness::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMeshSmootheness");
  if(ret)
    {
    return (vtkMeshSmootheness*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMeshSmootheness;
}
#endif
//-------------------------------------------------------//
vtkMeshSmootheness::vtkMeshSmootheness()
{
  this->CurvatureType = VTK_CURVATURE_GAUSS;
  this->Smootheness = 0;
}

//-------------------------------------------------------//
vtkMeshSmootheness::~vtkMeshSmootheness()
{
  delete this->MeanCurvature;
  delete this->NumNeighb;
}

//----------------------------------------------------------------------------
void vtkMeshSmootheness::SetInput(vtkPolyData *input)
{
  this->vtkProcessObject::SetNthInput(0, input);
  this->mesh = this->GetInput();
  this->NumPts = this->mesh->GetNumberOfPoints();
  this->NumCls = this->mesh->GetNumberOfCells();

  this->MeanCurvature = new double[this->NumPts];
  this->NumNeighb =  new int[this->NumPts];
}

//-------------------------------------------------------//
void vtkMeshSmootheness::GetMeanCurvature()
{
  memset((void *)this->MeanCurvature, 0, this->NumPts*sizeof(double));
  memset((void *)this->NumNeighb, 0, this->NumPts*sizeof(int));
  
  vtkIdList* vertices, *vertices_n, *neighbours;
  vtkTriangle* facet;
  vtkTriangle* neighbour;

  vertices   = vtkIdList::New();
  vertices_n = vtkIdList::New();
  neighbours = vtkIdList::New();
  facet      = vtkTriangle::New();
  neighbour  = vtkTriangle::New();
  // Get the array so we can write to it directly
  //     data
  int v, v_l, v_r, v_o, n, nv;// n short for neighbor
  
  //     create-allocate
  double n_f[3]; // normal of facet (could be stored for later?)
  double n_n[3]; // normal of edge
  double t[3];   // to store the cross product of n_f n_n
  double ore[3]; // origin of e
  double end[3]; // end of e
  double oth[3]; //     third vertex necessary for comp of n
  double vn0[3];
  double vn1[3]; // vertices for computation of neighbour's n
  double vn2[3];
  double e[3];   // edge (oriented)
  
  double cs, sn;    // cs: cos; sn sin
  double angle, length, Af, Hf;  // temporary store
  
  this->mesh->BuildLinks();
  Hf = 0.0;
  nv = 3; // Works only for triangles

  for (int f = 0; f < this->NumCls; f++)
    {
      this->mesh->GetCellPoints(f,vertices);

      for (v = 0; v < nv; v++)
        {
	  // get neighbour
	  v_l = vertices->GetId(v);
	  v_r = vertices->GetId((v+1) % nv);
	  v_o = vertices->GetId((v+2) % nv);

	  this->mesh->GetCellEdgeNeighbors(f,v_l,v_r,neighbours);
	  
	  // compute only if there is really ONE neighbour
	  // AND meanCurvature has not been computed yet!
	  // (ensured by n > f)
	  if (neighbours->GetNumberOfIds() == 1 && (n = neighbours->GetId(0)) > f)
	    {
	      // find 3 corners of f: in order!
	      this->mesh->GetPoint(v_l,ore);
	      this->mesh->GetPoint(v_r,end);
	      this->mesh->GetPoint(v_o,oth);
	      // compute normal of f
	      facet->ComputeNormal(ore,end,oth,n_f);
	      // compute common edge
	      e[0] = end[0]; e[1] = end[1]; e[2] = end[2];
	      e[0] -= ore[0]; e[1] -= ore[1]; e[2] -= ore[2];
	      length = double(vtkMath::Normalize(e));
	      Af = double(facet->TriangleArea(ore,end,oth));
	      // find 3 corners of n: in order!
	      this->mesh->GetCellPoints(n,vertices_n);
	      this->mesh->GetPoint(vertices_n->GetId(0),vn0);
	      this->mesh->GetPoint(vertices_n->GetId(1),vn1);
	      this->mesh->GetPoint(vertices_n->GetId(2),vn2);
	      Af += double(facet->TriangleArea(vn0,vn1,vn2));
	      // compute normal of n
	      neighbour->ComputeNormal(vn0,vn1,vn2,n_n);
	      // the cosine is n_f * n_n
	      cs = double(vtkMath::Dot(n_f,n_n));
	      // the sin is (n_f x n_n) * e
	      vtkMath::Cross(n_f,n_n,t);
	      sn = double(vtkMath::Dot(t,e));
	      // signed angle in [-pi,pi]
	      if (sn!=0.0 || cs!=0.0)
		{
		  angle = atan2(sn,cs);
		  Hf    = length*angle;
		}
	      else
		{
		  Hf = 0.0;
		}
	      // add weighted Hf to scalar at v_l and v_r
	      if (Af!=0.0)
		{
		  (Hf /= Af) *=3.0;
		}
	      MeanCurvature[v_l] += Hf;
	      MeanCurvature[v_r] += Hf;
	      NumNeighb[v_l] += 1;
	      NumNeighb[v_r] += 1;
	    }
        }
    }
  
  // put curvature in vtkArray
  for (v = 0; v < this->NumPts; v++)
    {
      if (NumNeighb[v]>0)
	{
	  Hf = 0.5*MeanCurvature[v]/(double)NumNeighb[v];
	}
      this->Smootheness +=  Hf;
    }
  
  this->Smootheness = this->Smootheness/this->NumPts;
  
  // clean
  vertices  ->Delete();
  vertices_n->Delete();
  neighbours->Delete();
  facet     ->Delete();
  neighbour ->Delete();
};
//--------------------------------------------
#define CLAMP_MACRO(v)    ((v)<(-1) ? (-1) : (v) > (1) ? (1) : v)
void vtkMeshSmootheness::GetGaussCurvature()
{
    // vtk data
    vtkPolyData* output = this->GetInput();
    vtkCellArray* facets = output->GetPolys();
    vtkTriangle* facet = vtkTriangle::New();

    // other data
    vtkIdType Nv   = output->GetNumberOfPoints();

    double* K = new double[Nv];
    double* dA = new double[Nv];
    double pi2 = 2.0*vtkMath::Pi();
    for (int k = 0; k < Nv; k++)
      {
      K[k]  = pi2;
      dA[k] = 0.0;
      }

    double v0[3], v1[3], v2[3], e0[3], e1[3], e2[3];

    double A, alpha0, alpha1, alpha2;

    vtkIdType f, *vert=0;
    facets->InitTraversal();
    while (facets->GetNextCell(f,vert))
      {
      output->GetPoint(vert[0],v0);
      output->GetPoint(vert[1],v1);
      output->GetPoint(vert[2],v2);
      // edges
      e0[0] = v1[0] ; e0[1] = v1[1] ; e0[2] = v1[2] ;
      e0[0] -= v0[0]; e0[1] -= v0[1]; e0[2] -= v0[2];

      e1[0] = v2[0] ; e1[1] = v2[1] ; e1[2] = v2[2] ;
      e1[0] -= v1[0]; e1[1] -= v1[1]; e1[2] -= v1[2];

      e2[0] = v0[0] ; e2[1] = v0[1] ; e2[2] = v0[2] ;
      e2[0] -= v2[0]; e2[1] -= v2[1]; e2[2] -= v2[2];

      // normalise
      vtkMath::Normalize(e0); vtkMath::Normalize(e1); vtkMath::Normalize(e2);
      // angles
      // I get lots of acos domain errors so clamp the value to +/-1 as the
      // normalize function can return 1.000000001 etc (I think)
      double ac1 = vtkMath::Dot(e1,e2);
      double ac2 = vtkMath::Dot(e2,e0);
      double ac3 = vtkMath::Dot(e0,e1);
      alpha0 = acos(-CLAMP_MACRO(ac1));
      alpha1 = acos(-CLAMP_MACRO(ac2));
      alpha2 = acos(-CLAMP_MACRO(ac3));

      // surf. area
      A = double(facet->TriangleArea(v0,v1,v2));
      // UPDATE
      dA[vert[0]] += A;
      dA[vert[1]] += A;
      dA[vert[2]] += A;
      K[vert[0]] -= alpha1;
      K[vert[1]] -= alpha2;
      K[vert[2]] -= alpha0;
      }

    int numPts = output->GetNumberOfPoints();
    // put curvature in vtkArray
    vtkDoubleArray* gaussCurvature = vtkDoubleArray::New();
    gaussCurvature->SetName("Gauss_Curvature");
    gaussCurvature->SetNumberOfComponents(1);
    gaussCurvature->SetNumberOfTuples(numPts);
    double *gaussCurvatureData = gaussCurvature->GetPointer(0);

    for (int v = 0; v < Nv; v++)
      {
      if (dA[v]>0.0)
        {
        gaussCurvatureData[v] = 3.0*K[v]/dA[v];
        }
      else
        {
        gaussCurvatureData[v] = 0.0;
        }
      this->Smootheness +=  gaussCurvatureData[v];
      }

    this->Smootheness =	this->Smootheness/Nv;

    /*******************************************************/
    if (facet) facet->Delete();
    if (K)              delete [] K;
    if (dA)             delete [] dA;
    if (gaussCurvature) gaussCurvature->Delete();
    /*******************************************************/
};

//-------------------------------------------------------
void vtkMeshSmootheness::Execute()
{
  if ( this->CurvatureType == VTK_CURVATURE_GAUSS )
      {
      this->GetGaussCurvature();
      }
  else if ( this->CurvatureType == VTK_CURVATURE_MEAN )
      {
      this->GetMeanCurvature();
      }
  else
      {
      vtkErrorMacro("Only Gauss and Mean Curvature type available");
      }
}
/*-------------------------------------------------------*/
void vtkMeshSmootheness::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkPolyDataToPolyDataFilter::PrintSelf(os,indent);
  os << indent << "CurvatureType: " << this->CurvatureType << "\n";
}

