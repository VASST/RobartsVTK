#include "vtkCudaFunctionPolygonWriter.h"
#include "vtkObjectFactory.h"

#include <iostream>

vtkStandardNewMacro(vtkCudaFunctionPolygonWriter);

vtkCudaFunctionPolygonWriter::vtkCudaFunctionPolygonWriter(){
	this->fileNameSet = false;
	this->objects.clear();
}

vtkCudaFunctionPolygonWriter::~vtkCudaFunctionPolygonWriter(){
	this->objects.clear();
}

void vtkCudaFunctionPolygonWriter::SetFileName( std::string f ){
	this->filename = f;
	this->fileNameSet = true;
}

void vtkCudaFunctionPolygonWriter::Write(){
	if( !this->fileNameSet ){
		vtkErrorMacro(<<"Must set file name before writing");
		return;
	}

	//open the file stream
	this->file = new std::ofstream();
	this->file->open(this->filename.c_str(),std::ios_base::out);

	//print the objects
	*(this->file) << this->objects.size() << std::endl;
	for( std::list<vtkCudaFunctionPolygon*>::iterator it = this->objects.begin(); it != this->objects.end(); it++){
		this->printTFPolygon( *it );
	}

	//close the file
	file->close();
	delete file;
}

void vtkCudaFunctionPolygonWriter::AddInput( vtkCudaFunctionPolygon* object ){
	bool alreadyThere = false;
	for( std::list<vtkCudaFunctionPolygon*>::iterator it = this->objects.begin(); it != this->objects.end(); it++){
		alreadyThere |= (*it == object);
	}
	if( alreadyThere ) return;

	this->objects.push_back( object );

}

void vtkCudaFunctionPolygonWriter::RemoveInput( vtkCudaFunctionPolygon* object ){
	for( std::list<vtkCudaFunctionPolygon*>::iterator it = this->objects.begin(); it != this->objects.end(); it++){
		if(*it == object){
			this->objects.erase(it);
			break;
		}
	}
}

void vtkCudaFunctionPolygonWriter::printTFPolygon( vtkCudaFunctionPolygon* e ){

	//save the colour, opacity and identifier
	*(this->file) << e->GetRedColourValue() << " " << e->GetGreenColourValue() << " " << e->GetBlueColourValue() << " " << e->GetOpacity() << std::endl;
	*(this->file) << e->GetIdentifier() << std::endl;

	//save the number of vertices
	unsigned int numVertices = e->GetNumVertices();
	*file << numVertices << std::endl;

	//save each vertex (intensity first, then gradient)
	for(unsigned int i = 0; i < numVertices; i++){
		*file << e->GetVertexIntensity(i) << " " << e->GetVertexGradient(i) << std::endl;
	}

}