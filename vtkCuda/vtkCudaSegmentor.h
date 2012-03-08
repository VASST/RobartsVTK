#ifndef __vtkCudaSegmentor_h
#define __vtkCudaSegmentor_h

#include "vtkObject.h"
#include "vtkImageData.h"
#include "vtkPolyData.h"
#include "vtkImageMirrorPad.h"
#include "vtkCuda2DTransferClassificationFunction.h"

struct vtkCudaSegmentorGraphPoint {
	//describes which classification this point belongs to
	short type;

	//boolean to describe if this node has been visited in a search
	bool visited;

	//the index of the value in the mask that this vertex was created to serve
	//later, this will be the vtkIDType of the vertex in the polydata
	vtkIdType index;

	//activation of the graph is equivalent to the location of the vertex in 3D space
	double activationXEven;
	double activationYEven;
	double activationZEven;
	double activationXOdd;
	double activationYOdd;
	double activationZOdd;

	//the activation cannot be outside [lowerBound, lowerBound+1] for any of x, y or z
	double originalPositionX;
	double originalPositionY;
	double originalPositionZ;

	//potential edges (6 maximum) in the graph ( note 0,1 for x; 2,3 for y; 4,5 for z; up and down respectively) 
	vtkCudaSegmentorGraphPoint* edges[6];

	//pointer to create a linked list with
	vtkCudaSegmentorGraphPoint* next;
};

class vtkCudaSegmentor : public vtkObject
{
public:
	static vtkCudaSegmentor *New();

	double GetConstraintThreshold(){ return this->constraintThreshold; }
	double GetSmoothingCoefficient(){ return this->smoothingCoefficient; }
	int GetConstraintType(){ return this->constraintType; }
	int GetMeshingType(){ return this->meshCase; }

	void SetConstraintThreshold(double constraintThreshold);
	void SetSmoothingCoefficient(double smoothingCoefficient);
	void SetConstraintType(int constraintType);
	void SetMeshingType(int meshType);

	void SetInput(vtkImageData*);
	void SetFunction(vtkCuda2DTransferClassificationFunction*);
	void Update();
	vtkPolyData* GetOutput(int n);

protected:
	vtkCudaSegmentor();
	~vtkCudaSegmentor();


private:

	//triangulation parameters
	double constraintThreshold;
	double smoothingCoefficient;
	int constraintType;
	int meshCase;

	//IO information
	vtkCuda2DTransferClassificationFunction* funct;
	vtkImageData* input;
	std::vector<vtkPolyData*>* output;

	//padder required for CUDA code to function optimally
	vtkImageMirrorPad* padder;

	//data required for constrained elastic surface nets
	int numberOfClassifications;
	void ComputeElasticSurfaceNet(short* mask, int* dims, float* spacing, float* origin);

	//helper methods required to transform the graph information into a surface polydata (also calculates 2 x surface area)
	void triangulate( std::vector<vtkCudaSegmentorGraphPoint*>*,  vtkCellArray*, vtkPoints*, float*, float* );
	void use64CaseMesh(vtkCudaSegmentorGraphPoint* v, vtkCellArray* triangleArray); 
	void use12CaseMesh(vtkCudaSegmentorGraphPoint* v, vtkCellArray* triangleArray);

	//helper methods for the relaxation of the net (return the number of iterations used)
	int numSmoothingIterations;
	void relaxWithNoConstraints( std::vector<vtkCudaSegmentorGraphPoint*>* );
	void relaxWithImplicitConstraints( std::vector<vtkCudaSegmentorGraphPoint*>* );
	void relaxWithExplicitConstraints( std::vector<vtkCudaSegmentorGraphPoint*>* );

};

#endif