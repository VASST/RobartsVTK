#include "vtkImageFrangiFilter.h"
#include "vtkObjectFactory.h"
#include "vtkMath.h"

//-----------------------------------------------------------------------------------//
// Constructor, destructor, and defaults
//-----------------------------------------------------------------------------------//

vtkStandardNewMacro(vtkImageFrangiFilter);

vtkImageFrangiFilter::vtkImageFrangiFilter(){
	Sheet = 0;
	Line = 1;
	Blob = 0;
	
	AssymmetrySensitivity = 10;
	StructureSensitivity = 100;
	BlobnessSensitivity = 0.1;
	GradientSensitivity = 1;
}

vtkImageFrangiFilter::~vtkImageFrangiFilter(){
}

//-----------------------------------------------------------------------------------//
// Run Frangi filtering process
//-----------------------------------------------------------------------------------//

void vtkImageFrangiFilter::SimpleExecute(vtkImageData* input, vtkImageData* output){
	switch( input->GetScalarType() ){
		vtkTemplateMacro( SimpleExecute<VTK_TT>(input, output) );
	}
}

template<class T>
void vtkImageFrangiFilter::SimpleExecute(vtkImageData* input, vtkImageData* output){
	
	//get volume size and location info
	int Extent[6];
	input->GetExtent(Extent);
	int Dims[3] = {Extent[1]-Extent[0]+1,Extent[3]-Extent[2]+1,Extent[5]-Extent[4]+1};
	int Jump[4] = {1, Dims[0], Dims[0]*Dims[1], Dims[0]*Dims[1]*Dims[2]};
	T* inData = (T*) input->GetScalarPointer();
	T* outData = (T*) output->GetScalarPointer();
	
	//allocate temporary areas
	double* dfdx = new double[Jump[3]];
	double* dfdy = new double[Jump[3]];
	double* dfdz = new double[Jump[3]];
	double* gmag = new double[Jump[3]];

	// do central differences
	int idx = 0;
	for( int z = 0; z < Dims[2]; z++ )
	for( int y = 0; y < Dims[1]; y++ )
	for( int x = 0; x < Dims[0]; x++, idx++ ){
		if( x > 0 && x < Dims[0]-1 )	dfdx[idx] = ((double) inData[idx-Jump[0]] - (double) inData[idx+Jump[0]]);
		else if( x > 0)					dfdx[idx] = ((double) inData[idx-Jump[0]] - (double) inData[idx]);
		else							dfdx[idx] = ((double) inData[idx        ] - (double) inData[idx+Jump[0]]);

		if( y > 0 && y < Dims[1]-1 )	dfdy[idx] = ((double) inData[idx-Jump[1]] - (double) inData[idx+Jump[1]]);
		else if( y > 0)					dfdy[idx] = ((double) inData[idx-Jump[1]] - (double) inData[idx]);
		else							dfdy[idx] = ((double) inData[idx        ] - (double) inData[idx+Jump[1]]);
		
		if( z > 0 && z < Dims[2]-1 )	dfdz[idx] = ((double) inData[idx-Jump[2]] - (double) inData[idx+Jump[2]]);
		else if( z > 0)					dfdz[idx] = ((double) inData[idx-Jump[2]] - (double) inData[idx]);
		else							dfdz[idx] = ((double) inData[idx        ] - (double) inData[idx+Jump[2]]);

		gmag[idx] = sqrt( dfdx[idx]*dfdx[idx] + dfdz[idx]*dfdz[idx] + dfdz[idx]*dfdz[idx] );
	}

	//find Hessian
	idx = 0;
	for( int z = 0; z < Dims[2]; z++ )
	for( int y = 0; y < Dims[1]; y++ )
	for( int x = 0; x < Dims[0]; x++, idx++ ){
		//get hessian
		double dffdxx = ( (x>0) ? ( (x<Dims[0]-1) ? dfdx[idx-Jump[0]]-dfdx[idx+Jump[0]]: dfdx[idx-Jump[0]] - dfdx[idx] ) : dfdx[idx] - dfdx[idx+Jump[0]] );
		double dffdyx = ( (x>0) ? ( (x<Dims[0]-1) ? dfdy[idx-Jump[0]]-dfdy[idx+Jump[0]]: dfdy[idx-Jump[0]] - dfdy[idx] ) : dfdy[idx] - dfdy[idx+Jump[0]] );
		double dffdzx = ( (x>0) ? ( (x<Dims[0]-1) ? dfdz[idx-Jump[0]]-dfdz[idx+Jump[0]]: dfdz[idx-Jump[0]] - dfdz[idx] ) : dfdz[idx] - dfdz[idx+Jump[0]] );
		double dffdyy = ( (y>0) ? ( (y<Dims[1]-1) ? dfdy[idx-Jump[1]]-dfdy[idx+Jump[1]]: dfdy[idx-Jump[1]] - dfdy[idx] ) : dfdy[idx] - dfdy[idx+Jump[1]] );
		double dffdzy = ( (y>0) ? ( (y<Dims[1]-1) ? dfdz[idx-Jump[1]]-dfdz[idx+Jump[1]]: dfdz[idx-Jump[1]] - dfdz[idx] ) : dfdz[idx] - dfdz[idx+Jump[1]] );
		double dffdzz = ( (z>0) ? ( (z<Dims[2]-1) ? dfdz[idx-Jump[2]]-dfdz[idx+Jump[2]]: dfdz[idx-Jump[2]] - dfdz[idx] ) : dfdz[idx] - dfdz[idx+Jump[2]] );
		double hess[3][3] = { {dffdxx, dffdyx, dffdzx}, {dffdyx, dffdyy, dffdzy}, {dffdzx, dffdzy, dffdzz} };
		
		//get eigenvalues
		double space[3][3];
		double eig[3];
		vtkMath::Diagonalize3x3(hess,eig,space);

		//sort eigenvalues
		if( abs(eig[0]) < abs(eig[1]) ) {double tmp = eig[0]; eig[0] = eig[1]; eig[1] = tmp;}
		if( abs(eig[0]) < abs(eig[2]) ) {double tmp = eig[0]; eig[0] = eig[2]; eig[2] = tmp;}
		if( abs(eig[1]) < abs(eig[2]) ) {double tmp = eig[1]; eig[1] = eig[2]; eig[2] = tmp;}

		double RA = ((eig[0] != 0) ? eig[1] / eig[0] : 0) / AssymmetrySensitivity;
		double RB = (eig[2] / sqrt(abs(eig[1]*eig[0]))) / BlobnessSensitivity;
		double S = sqrt( eig[0]*eig[0] + eig[1]*eig[1] + eig[2]*eig[2] ) / StructureSensitivity;
		double L = gmag[idx] / ((abs(eig[0])+8*DBL_MIN)*GradientSensitivity);

		outData[idx] = (T)((	Sheet*	(1-exp(-L*L/2))*exp(-RA*RA/2)*exp(-RB*RB/2)	
						+		Line*	(1-exp(-RA*RA/2))*exp(-RB*RB/2)	
						+		Blob*	exp(-L*L/2)*(1-exp(-RB*RB/2))					) * (1-exp(-S*S/2)));

		//outData[idx] = abs(L);
	}
	
	//clean up
	delete[] dfdx;
	delete[] dfdy;
	delete[] dfdz;
	delete[] gmag;
}