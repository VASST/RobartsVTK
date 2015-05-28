/** @file vtkVolumeMapper.h
 *
 *  @brief Header file defining a volume mapper (ray caster) using CUDA kernels for parallel ray calculation
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 29, 2011
 *
 */

#ifndef __vtkCuda2DInExLogicVolumeMapper_H
#define __vtkCuda2DInExLogicVolumeMapper_H

#include "vtkCudaVolumeMapper.h"

#include "vtkCuda2DInExLogicTransferFunctionInformationHandler.h"
#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CUDA_containerOutputImageInformation.h"
#include "CUDA_container2DTransferFunctionInformation.h"

/** @brief vtkCuda2DInExLogicVolumeMapper is a volume mapper, taking a set of 3D image data objects, volume and renderer as input and creates a 2D ray casted projection of the scene which is then displayed to screen
 *
 */
class vtkCuda2DInExLogicVolumeMapper : public vtkCudaVolumeMapper {
public:

  vtkTypeMacro( vtkCuda2DInExLogicVolumeMapper, vtkCudaVolumeMapper );

  /** @brief VTK compatible constructor method
   *
   */
  static vtkCuda2DInExLogicVolumeMapper *New();

  virtual void SetInputInternal( vtkImageData * image, int frame);
  virtual void ClearInputInternal();
  virtual void ChangeFrameInternal(int frame);
  virtual void InternalRender (  vtkRenderer* ren, vtkVolume* vol,
                  const cudaRendererInformation& rendererInfo,
                  const cudaVolumeInformation& volumeInfo,
                  const cudaOutputImageInformation& outputInfo );

  /** @brief Set the transfer function used for determining colour and opacity in the volume rendering process which is given to the volume information handler
   *
   *  @param func The 2 dimensional transfer function
   */
  void SetVisualizationFunction(vtkCuda2DTransferFunction* func);
  
  /** @brief Get the transfer function used for determining colour and opacity in the volume rendering process which is given to the volume information handler
   *
   */
  vtkCuda2DTransferFunction* GetVisualizationFunction();

  /** @brief Set the transfer function used for determining inclusion and exclusion in the volume rendering process which is given to the volume information handler
   *
   *  @param func The 2 dimensional transfer function
   */
  void SetInExLogicFunction(vtkCuda2DTransferFunction* func);
  
  /** @brief Get the transfer function used for determining inclusion and exclusion in the volume rendering process which is given to the volume information handler
   *
   */
  vtkCuda2DTransferFunction* GetInExLogicFunction();
  
  void SetUseBlackKeyhole( bool t );
  vtkGetMacro( UseBlackKeyhole, bool );

protected:
  /** @brief Constructor which initializes the number of frames, rendering type and other constants to safe initial values, and creates the required information handlers
   *
   */
  vtkCuda2DInExLogicVolumeMapper();

  /** @brief Destructor which deallocates the various information handlers and matrices
   *
   */
  ~vtkCuda2DInExLogicVolumeMapper();

  virtual void Reinitialize(int withData = 0);
  virtual void Deinitialize(int withData = 0);
  
  vtkCuda2DInExLogicTransferFunctionInformationHandler* transferFunctionInfoHandler;

  static vtkMutexLock* tfLock;

  bool UseBlackKeyhole;

private:
  vtkCuda2DInExLogicVolumeMapper operator=(const vtkCuda2DInExLogicVolumeMapper&); /**< not implemented */
  vtkCuda2DInExLogicVolumeMapper(const vtkCuda2DInExLogicVolumeMapper&); /**< not implemented */

};

#endif
