#ifndef __VTKCUDAVOXELCLASSIFIER_H__
#define __VTKCUDAVOXELCLASSIFIER_H__

#include "CUDA_voxelclassifier.h"
#include "vtkImageAlgorithm.h"
#include "vtkImageData.h"
#include "vtkTransform.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkAlgorithmOutput.h"

#include "CudaObject.h"

#include "vtkCuda2DTransferFunction.h"

#include "vtkPlanes.h"
#include "vtkPlaneCollection.h"
#include "vtkMatrix4x4.h"

class vtkCudaVoxelClassifier : public vtkImageAlgorithm, public CudaObject
{
public:
  vtkTypeMacro( vtkCudaVoxelClassifier, vtkImageAlgorithm );

  static vtkCudaVoxelClassifier *New();

  /** @brief Set the transfer function used for determining colour and opacity in the volume rendering process which is given to the volume information handler
   *
   *  @param func The 2 dimensional transfer function
   */
  void SetFunction(vtkCuda2DTransferFunction* func);
  
  /** @brief Get the transfer function used for determining colour and opacity in the volume rendering process which is given to the volume information handler
   *
   */
  vtkCuda2DTransferFunction* GetFunction();

  /** @brief Set the transfer function used for determining colour and opacity in the volume rendering process which is given to the volume information handler within the keyhole window
   *
   *  @param func The 2 dimensional transfer function
   */
  void SetKeyholeFunction(vtkCuda2DTransferFunction* func);
  
  /** @brief Get the transfer function used for determining colour and opacity in the volume rendering process which is given to the volume information handler within the keyhole window
   *
   */
  vtkCuda2DTransferFunction* GetKeyholeFunction();

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
  
  // Description:
  // Specify Keyhole planes to be applied when the data is mapped
  // (at most 6 Keyhole planes can be specified).
  void AddKeyholePlane(vtkPlane *plane);
  void RemoveKeyholePlane(vtkPlane *plane);
  void RemoveAllKeyholePlanes();

  // Description:
  // Get/Set the vtkPlaneCollection which specifies the
  // Keyhole planes.
  virtual void SetKeyholePlanes(vtkPlaneCollection*);
  vtkGetObjectMacro(KeyholePlanes,vtkPlaneCollection);

  // Description:
  // An alternative way to set Keyhole planes: use up to six planes found
  // in the supplied instance of the implicit function vtkPlanes.
  void SetKeyholePlanes(vtkPlanes *planes);

  // Description:
  // Specify Clipping planes to be applied when the data is mapped
  // (at most 6 Clipping planes can be specified).
  void AddClippingPlane(vtkPlane *plane);
  void RemoveClippingPlane(vtkPlane *plane);
  void RemoveAllClippingPlanes();

  // Description:
  // Get/Set the vtkPlaneCollection which specifies the
  // Clipping planes.
  virtual void SetClippingPlanes(vtkPlaneCollection*);
  vtkGetObjectMacro(ClippingPlanes,vtkPlaneCollection);

  // Description:
  // An alternative way to set Clipping planes: use up to six planes found
  // in the supplied instance of the implicit function vtkPlanes.
  void SetClippingPlanes(vtkPlanes *planes);

protected:
  vtkCudaVoxelClassifier();
  virtual ~vtkCudaVoxelClassifier();
  
  void Reinitialize(int withData);
  void Deinitialize(int withData);

  Voxel_Classifier_Information ClassifierInfo;

  void ComputeMatrices( vtkImageData* inputData );
  void SetVoxelsToWorldMatrix( vtkMatrix4x4* in );
  void SetWorldToVoxelsMatrix( vtkMatrix4x4* in );
  void FigurePlanes(vtkPlaneCollection* planes, float* planesArray, int* numberOfPlanes);
  
  vtkPlaneCollection* ClippingPlanes;
  vtkPlaneCollection* KeyholePlanes;

  vtkCuda2DTransferFunction* PrimaryFunction;
  vtkCuda2DTransferFunction* KeyholeFunction;
  
  float  WorldToVoxelsMatrix[16];  /**< Array representing the world to voxels transformation as a matrix */
  float  VoxelsToWorldMatrix[16];  /**< Array representing the voxels to world transformation as a matrix */

  int    TextureSize;

private:
  vtkCudaVoxelClassifier operator=(const vtkCudaVoxelClassifier&){}
  vtkCudaVoxelClassifier(const vtkCudaVoxelClassifier&){}
  
};

#endif