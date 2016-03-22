
#include "vtkBoxRepresentation2.h"
#include "vtkObjectFactory.h"


vtkStandardNewMacro( vtkBoxRepresentation2 );

//------------------------------------------------------------------------------------------------------
vtkBoxRepresentation2::vtkBoxRepresentation2(){
	
}

vtkBoxRepresentation2::~vtkBoxRepresentation2(){

}

void vtkBoxRepresentation2::setSelectedOutlineEdge(int idx){

	this->GenerateOutline();
	vtkCellArray *cells = this->HexPolyData->GetPolys();


}