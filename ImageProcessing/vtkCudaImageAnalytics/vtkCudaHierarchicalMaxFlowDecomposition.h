#ifndef __VTKCUDAHIERARCHICALMAXFLOWDECOMPOSITION_H__
#define __VTKCUDAHIERARCHICALMAXFLOWDECOMPOSITION_H__

#include "vtkCudaImageAnalyticsExport.h"

#include "CudaObject.h"
#include "vtkImageAlgorithm.h"
#include <map>

class vtkAlgorithmOutput;
class vtkDirectedGraph;
class vtkImageCast;
class vtkImageData;
class vtkInformation;
class vtkInformationVector;
class vtkTransform;
class vtkTree;

class vtkCudaImageAnalyticsExport vtkCudaHierarchicalMaxFlowDecomposition : public vtkImageAlgorithm, public CudaObject
{
public:
  vtkTypeMacro( vtkCudaHierarchicalMaxFlowDecomposition, vtkImageAlgorithm );

  static vtkCudaHierarchicalMaxFlowDecomposition *New();

  //Set the hierarchical model used in the segmentation, note that this has to be a
  // tree.
  void SetHierarchy(vtkTree* graph);
  vtkTree* GetHierarchy();

  vtkDataObject* GetDataInput(int idx);
  void SetDataInput(int idx, vtkDataObject *input);
  vtkDataObject* GetSmoothnessInput(int idx);
  void SetSmoothnessInput(int idx, vtkDataObject *input);
  vtkDataObject* GetLabelInput(int idx);
  void SetLabelInput(int idx, vtkDataObject *input);

  double GetF0();
  double GetF(vtkIdType n);

  // Description:
  // If the subclass does not define an Execute method, then the task
  // will be broken up, multiple threads will be spawned, and each thread
  // will call this method. It is public so that the thread functions
  // can call this method.
  virtual int RequestData(vtkInformation *request,
                          vtkInformationVector **inputVector,
                          vtkInformationVector *outputVector);
  virtual int RequestInformation( vtkInformation* request,
                                  vtkInformationVector** inputVector,
                                  vtkInformationVector* outputVector);
  virtual int RequestUpdateExtent( vtkInformation* request,
                                   vtkInformationVector** inputVector,
                                   vtkInformationVector* outputVector);
  virtual int RequestDataObject( vtkInformation* request,
                                 vtkInformationVector** inputVector,
                                 vtkInformationVector* outputVector);
  virtual int FillInputPortInformation(int i, vtkInformation* info);

protected:
  vtkCudaHierarchicalMaxFlowDecomposition();
  virtual ~vtkCudaHierarchicalMaxFlowDecomposition();

  void Reinitialize(int withData);
  void Deinitialize(int withData);

  int CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int& NumNodes, int& NumLeaves, int& NumEdges );
  void FigureOutSmoothness( vtkIdType CurrNode, vtkInformationVector **inputVector );

  vtkTree* Hierarchy;
  std::map<vtkIdType,int> LeafMap;
  std::map<vtkIdType,int> BranchMap;

  std::map<vtkIdType,float*> BranchLabelBuffer;

  double    F0;
  double*    F;

  int VolumeSize;
  int VX, VY, VZ;

  std::map<vtkIdType,int> InputDataPortMapping;
  std::map<int,vtkIdType> BackwardsInputDataPortMapping;
  int FirstUnusedDataPort;
  std::map<vtkIdType,int> InputSmoothnessPortMapping;
  std::map<int,vtkIdType> BackwardsInputSmoothnessPortMapping;
  int FirstUnusedSmoothnessPort;
  std::map<vtkIdType,int> InputLabelPortMapping;
  std::map<int,vtkIdType> BackwardsInputLabelPortMapping;
  int FirstUnusedLabelPort;

  //pointers to variable structures, easier to keep as part of the class definition
  float**  BranchSmoothnessTermBuffers;
  float**  LeafSmoothnessTermBuffers;

  float* DevGradientBuffer;

private:
  vtkCudaHierarchicalMaxFlowDecomposition operator=(const vtkCudaHierarchicalMaxFlowDecomposition&);
  vtkCudaHierarchicalMaxFlowDecomposition(const vtkCudaHierarchicalMaxFlowDecomposition&);
};

#endif
