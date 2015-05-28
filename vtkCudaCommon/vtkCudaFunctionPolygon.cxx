#include "vtkCudaFunctionPolygon.h"
#include "vtkObjectFactory.h"

vtkStandardNewMacro(vtkCudaFunctionPolygon);

vtkCudaFunctionPolygon::vtkCudaFunctionPolygon(){

  //initialize the container
  contour.clear();
}

vtkCudaFunctionPolygon::~vtkCudaFunctionPolygon(){

  //remove the vertices from the heap
  for(std::vector<vertex*>::iterator it = contour.begin(); it != contour.end(); it++){
    delete (*it);
  }
  contour.clear();

}

const int vtkCudaFunctionPolygon::GetNumVertices(){
  return contour.size();
}

void vtkCudaFunctionPolygon::AddVertex( float intensity, float gradient ){
  
  //create a new vertex and push it onto the list
  vertex* newVertex = new vertex();
  newVertex->intensity = intensity;
  newVertex->gradient = gradient;
  contour.push_back( newVertex );
}

void vtkCudaFunctionPolygon::AddVertex( float intensity, float gradient, unsigned int index ){

  //if we are outside the contour range, just exit
  if(index > contour.size() ) return;

  //create a new vertex
  vertex* newVertex = new vertex();
  newVertex->intensity = intensity;
  newVertex->gradient = gradient;

  //if we are at the end of the polygon, just push back the new vertex
  if(index == contour.size() ){
    contour.push_back( newVertex );
    return;
  }

  //else, get an iterator to that point
  std::vector<vertex*>::iterator it = contour.begin();
  for(int i = index; i > 0; i--) it++;
  contour.insert(it, newVertex );

}

void vtkCudaFunctionPolygon::ModifyVertex( float intensity, float gradient, unsigned int index ){

  //if we choose a vertex not in the polygon, exit
  if(index >= contour.size()) return;

  //get an iterator to that point
  std::vector<vertex*>::iterator it = contour.begin();
  for(int i = index; i > 0; i--) it++;
  (*it)->intensity = intensity;
  (*it)->gradient = gradient;

}

const float vtkCudaFunctionPolygon::GetVertexIntensity( unsigned int index ){
  if(index >= contour.size()) return -1.0f;
  return contour[index]->intensity;
}

const float vtkCudaFunctionPolygon::GetVertexGradient( unsigned int index ){
  if(index >= contour.size()) return -1.0f;
  return contour[index]->gradient;
}

void vtkCudaFunctionPolygon::RemoveVertex( unsigned int index ){

  //if we choose a vertex not in the polygon, exit
  if(index >= contour.size()) return;

  //else, get an iterator to that point and erase it
  std::vector<vertex*>::iterator it = contour.begin();
  for(int i = index; i > 0; i--) it++;
  delete (*it);
  contour.erase(it);

}

double vtkCudaFunctionPolygon::getMaxGradient(){
  float max = 0.0f;
  for(std::vector<vertex*>::iterator it = contour.begin(); it != contour.end(); it++){
    if( (*it)->gradient > max ) max = (*it)->gradient;
  }
  return max;
}

double vtkCudaFunctionPolygon::getMinGradient(){
  if(contour.size() == 0) return 0.0;
  float min = contour[0]->gradient;
  for(std::vector<vertex*>::iterator it = contour.begin(); it != contour.end(); it++){
    if( (*it)->gradient < min ) min = (*it)->gradient;
  }
  return min;
}

double vtkCudaFunctionPolygon::getMaxIntensity(){
  float max = 0.0f;
  for(std::vector<vertex*>::iterator it = contour.begin(); it != contour.end(); it++){
    if( (*it)->intensity > max ) max = (*it)->intensity;
  }
  return max;
}

double vtkCudaFunctionPolygon::getMinIntensity(){
  if(contour.size() == 0) return 0.0;
  float min = contour[0]->intensity;
  for(std::vector<vertex*>::iterator it = contour.begin(); it != contour.end(); it++){
    if( (*it)->intensity < min ) min = (*it)->intensity;
  }
  return min;
}

const bool vtkCudaFunctionPolygon::pointInPolygon(const float x, const float y) {

  int i, j = this->contour.size()-1 ;
  bool oddNodes = false;

  for (i=0; i<this->contour.size(); i++) {
    if (   ( this->contour[i]->gradient < y && this->contour[j]->gradient >= y )
      || ( this->contour[j]->gradient < y && this->contour[i]->gradient >= y ) )
      if (  this->contour[i]->intensity + ( y - this->contour[i]->gradient ) /
        ( this->contour[j]->gradient - this->contour[i]->gradient )
        * ( this->contour[j]->intensity - this->contour[i]->intensity ) < x )
        oddNodes = !oddNodes;
    j=i;
  }

  return oddNodes;
}

void vtkCudaFunctionPolygon::PopulatePortionOfTransferTable(  int IntensitySize, int GradientSize,
                      float IntensityLow, float IntensityHigh, float IntensityOffset,
                      float GradientLow, float GradientHigh, float GradientOffset,
                      float* rTable, float* gTable, float* bTable, float* aTable,
                      int logUsed){
  
  //find bounding rectangle for this polygon
  float minI = this->getMinIntensity();
  float maxI = this->getMaxIntensity();
  float minG = this->getMinGradient();
  float maxG = this->getMaxGradient();

  //transform into table co-ordinates
  if( logUsed & 2 ){
    minG = (log(minG*minG + GradientOffset) - log(GradientOffset) ) / log(2.0);
    maxG = (log(maxG*maxG + GradientOffset) - log(GradientOffset) ) / log(2.0);
  }
  if( logUsed & 1 ){
    minI = (log(minI*minI + IntensityOffset) - log(IntensityOffset) ) / log(2.0);
    maxI = (log(maxI*maxI + IntensityOffset) - log(IntensityOffset) ) / log(2.0);
  }

  //find the bounding co-ordinates
  int lowIntensityIndex = (double)(IntensitySize-1) * (minI - IntensityLow) / (IntensityHigh-IntensityLow);
  if( lowIntensityIndex < 0 ) lowIntensityIndex = 0;
  if( lowIntensityIndex >= IntensitySize ) lowIntensityIndex = IntensitySize-1;
  int highIntensityIndex = (double)(IntensitySize-1) * (maxI - IntensityLow) / (IntensityHigh-IntensityLow);
  if( highIntensityIndex < 0 ) highIntensityIndex = 0;
  if( highIntensityIndex >= IntensitySize ) highIntensityIndex = IntensitySize-1;
  int lowGradientIndex = (double)(GradientSize-1) * (minG - GradientLow) / (GradientHigh-GradientLow);
  if( lowGradientIndex < 0 ) lowGradientIndex = 0;
  if( lowGradientIndex >= GradientSize ) lowGradientIndex = GradientSize-1;
  int highGradientIndex = (double)(GradientSize-1) * (maxG - GradientLow) / (GradientHigh-GradientLow);
  if( highGradientIndex < 0 ) highGradientIndex = 0;
  if( highGradientIndex >= GradientSize ) highGradientIndex = GradientSize-1;

  //populate the section of the table if it falls in the polygon
  for(int g = lowGradientIndex; g < highGradientIndex; g++){
    
    //calculate the gradient at this point
    double gradient = 0.0;
    if( logUsed & 2 ){
      gradient = (double)(g) * (double)(GradientHigh-GradientLow) / (double) (GradientSize-1) + GradientLow ;
      gradient = sqrt(exp( log(2.0) * gradient + log(GradientOffset) ) - GradientOffset);
    }else{
      gradient = (double) (g) * (GradientHigh-GradientLow) / (double)(GradientSize-1) + GradientLow;
    }

    for(int i = lowIntensityIndex; i < highIntensityIndex; i++){

      double intensity = 0.0;
      if( logUsed & 1 ){
        intensity = (double)(i) * (double)(IntensityHigh-IntensityLow) / (double) (IntensitySize-1) + IntensityLow ;
        intensity = sqrt(exp( log(2.0) * intensity + log(IntensityOffset) ) - IntensityOffset);
      }else{
        intensity = (double) (i) * (IntensityHigh-IntensityLow) / (double)(IntensitySize-1) + IntensityLow;
      }

      //tell if this point is within the triangles
      bool inside = pointInPolygon(intensity, gradient);

      if( inside ){
        int tableIndex = i + g * IntensitySize;
        if(rTable) rTable[tableIndex] = this->colourRed;
        if(gTable) gTable[tableIndex] = this->colourGreen;
        if(bTable) bTable[tableIndex] = this->colourBlue;
        if(aTable) aTable[tableIndex] = this->opacity;
      }
    }
  }

}


void vtkCudaFunctionPolygon::PopulatePortionOfShadingTable(  int IntensitySize, int GradientSize,
                      float IntensityLow, float IntensityHigh, float IntensityOffset,
                      float GradientLow, float GradientHigh, float GradientOffset,
                      float* aTable, float* dTable, float* sTable, float* pTable,
                      int logUsed){
  
  //find bounding rectangle for this polygon
  float minI = this->getMinIntensity();
  float maxI = this->getMaxIntensity();
  float minG = this->getMinGradient();
  float maxG = this->getMaxGradient();

  //transform into table co-ordinates
  if( logUsed & 2 ){
    minG = (log(minG*minG + GradientOffset) - log(GradientOffset) ) / log(2.0);
    maxG = (log(maxG*maxG + GradientOffset) - log(GradientOffset) ) / log(2.0);
  }
  if( logUsed & 1 ){
    minI = (log(minI*minI + IntensityOffset) - log(IntensityOffset) ) / log(2.0);
    maxI = (log(maxI*maxI + IntensityOffset) - log(IntensityOffset) ) / log(2.0);
  }

  //find the bounding co-ordinates
  int lowIntensityIndex = (double)(IntensitySize-1) * (minI - IntensityLow) / (IntensityHigh-IntensityLow);
  if( lowIntensityIndex < 0 ) lowIntensityIndex = 0;
  if( lowIntensityIndex >= IntensitySize ) lowIntensityIndex = IntensitySize-1;
  int highIntensityIndex = (double)(IntensitySize-1) * (maxI - IntensityLow) / (IntensityHigh-IntensityLow);
  if( highIntensityIndex < 0 ) highIntensityIndex = 0;
  if( highIntensityIndex >= IntensitySize ) highIntensityIndex = IntensitySize-1;
  int lowGradientIndex = (double)(GradientSize-1) * (minG - GradientLow) / (GradientHigh-GradientLow);
  if( lowGradientIndex < 0 ) lowGradientIndex = 0;
  if( lowGradientIndex >= GradientSize ) lowGradientIndex = GradientSize-1;
  int highGradientIndex = (double)(GradientSize-1) * (maxG - GradientLow) / (GradientHigh-GradientLow);
  if( highGradientIndex < 0 ) highGradientIndex = 0;
  if( highGradientIndex >= GradientSize ) highGradientIndex = GradientSize-1;

  //populate the section of the table if it falls in the polygon
  for(int g = lowGradientIndex; g < highGradientIndex; g++){
    
    //calculate the gradient at this point
    double gradient = 0.0;
    if( logUsed & 2 ){
      gradient = (double)(g) * (double)(GradientHigh-GradientLow) / (double) (GradientSize-1) + GradientLow ;
      gradient = sqrt(exp( log(2.0) * gradient + log(GradientOffset) ) - GradientOffset);
    }else{
      gradient = (double) (g) * (GradientHigh-GradientLow) / (double)(GradientSize-1) + GradientLow;
    }

    for(int i = lowIntensityIndex; i < highIntensityIndex; i++){

      double intensity = 0.0;
      if( logUsed & 1 ){
        intensity = (double)(i) * (double)(IntensityHigh-IntensityLow) / (double) (IntensitySize-1) + IntensityLow ;
        intensity = sqrt(exp( log(2.0) * intensity + log(IntensityOffset) ) - IntensityOffset);
      }else{
        intensity = (double) (i) * (IntensityHigh-IntensityLow) / (double)(IntensitySize-1) + IntensityLow;
      }

      //tell if this point is within the triangles
      bool inside = pointInPolygon(intensity, gradient);

      if( inside ){
        int tableIndex = i + g * IntensitySize;
        if(aTable) aTable[tableIndex] = this->Ambient;
        if(dTable) dTable[tableIndex] = this->Diffuse;
        if(sTable) sTable[tableIndex] = this->Specular;
        if(pTable) pTable[tableIndex] = this->SpecularPower;
      }
    }
  }

}

void vtkCudaFunctionPolygon::PopulatePortionOfClassifyTable(  int IntensitySize, int GradientSize,
                      float IntensityLow, float IntensityHigh, float IntensityOffset,
                      float GradientLow, float GradientHigh, float GradientOffset,
                      short* table, int logUsed){

  //find bounding rectangle for this polygon
  float minI = this->getMinIntensity();
  float maxI = this->getMaxIntensity();
  float minG = this->getMinGradient();
  float maxG = this->getMaxGradient();

  //transform into table co-ordinates
  if( logUsed & 2 ){
    minG = (log(minG*minG + GradientOffset) - log(GradientOffset) ) / log(2.0);
    maxG = (log(maxG*maxG + GradientOffset) - log(GradientOffset) ) / log(2.0);
  }
  if( logUsed & 1 ){
    minI = (log(minI*minI + IntensityOffset) - log(IntensityOffset) ) / log(2.0);
    maxI = (log(maxI*maxI + IntensityOffset) - log(IntensityOffset) ) / log(2.0);
  }

  //find the bounding co-ordinates
  int lowIntensityIndex = (double)(IntensitySize-1) * (minI - IntensityLow) / (IntensityHigh-IntensityLow);
  if( lowIntensityIndex < 0 ) lowIntensityIndex = 0;
  if( lowIntensityIndex >= IntensitySize ) lowIntensityIndex = IntensitySize-1;
  int highIntensityIndex = (double)(IntensitySize-1) * (maxI - IntensityLow) / (IntensityHigh-IntensityLow);
  if( highIntensityIndex < 0 ) highIntensityIndex = 0;
  if( highIntensityIndex >= IntensitySize ) highIntensityIndex = IntensitySize-1;
  int lowGradientIndex = (double)(GradientSize-1) * (minG - GradientLow) / (GradientHigh-GradientLow);
  if( lowGradientIndex < 0 ) lowGradientIndex = 0;
  if( lowGradientIndex >= GradientSize ) lowGradientIndex = GradientSize-1;
  int highGradientIndex = (double)(GradientSize-1) * (maxG - GradientLow) / (GradientHigh-GradientLow);
  if( highGradientIndex < 0 ) highGradientIndex = 0;
  if( highGradientIndex >= GradientSize ) highGradientIndex = GradientSize-1;

  //populate the section of the table if it falls in the polygon
  for(int g = lowGradientIndex; g < highGradientIndex; g++){
    
    //calculate the gradient at this point
    double gradient = 0.0;
    if( logUsed & 2 ){
      gradient = (double)(g) * (double)(GradientHigh-GradientLow) / (double) (GradientSize-1) + GradientLow ;
      gradient = sqrt(exp( log(2.0) * gradient + log(GradientOffset) ) - GradientOffset);
    }else{
      gradient = (double) (g) * (GradientHigh-GradientLow) / (double)(GradientSize-1) + GradientLow;
    }

    for(int i = lowIntensityIndex; i < highIntensityIndex; i++){

      double intensity = 0.0;
      if( logUsed & 1 ){
        intensity = (double)(i) * (double)(IntensityHigh-IntensityLow) / (double) (IntensitySize-1) + IntensityLow ;
        intensity = sqrt(exp( log(2.0) * intensity + log(IntensityOffset) ) - IntensityOffset);
      }else{
        intensity = (double) (i) * (IntensityHigh-IntensityLow) / (double)(IntensitySize-1) + IntensityLow;
      }

      //tell if this point is within the triangles
      bool inside = pointInPolygon(intensity, gradient);

      if( inside ){
        int tableIndex = i + g * IntensitySize;
        table[tableIndex] = this->identifier;
      }
    }
  }


  
}