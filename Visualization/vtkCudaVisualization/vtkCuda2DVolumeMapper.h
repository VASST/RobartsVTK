/** @file vtkVolumeMapper.h
 *
 *  @brief Header file defining a volume mapper (ray caster) using CUDA kernels for parallel ray calculation
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 29, 2011
 *
 */

#ifndef __VTKCUDA2DVOLUMEMAPPER_H
#define __VTKCUDA2DVOLUMEMAPPER_H

#include "vtkCudaVisualizationExport.h"

#include "vtkCudaVolumeMapper.h"

class CUDA_container2DTransferFunctionInformation;
class CUDA_containerOutputImageInformation;
class CUDA_containerRendererInformation;
class CUDA_containerVolumeInformation;
class vtkCuda2DTransferFunction;
class vtkCuda2DTransferFunctionInformationHandler;
struct cudaArray;

/** @brief vtkCuda2DVolumeMapper is a volume mapper, taking a set of 3D image data objects, volume and renderer as input and creates a 2D ray casted projection of the scene which is then displayed to screen
 *
 */
class vtkCudaVisualizationExport vtkCuda2DVolumeMapper : public vtkCudaVolumeMapper
{
public:

  vtkTypeMacro( vtkCuda2DVolumeMapper, vtkCudaVolumeMapper );

  /** @brief VTK compatible constructor method
   *
   */
  static vtkCuda2DVolumeMapper *New();

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

protected:
  /** @brief Constructor which initializes the number of frames, rendering type and other constants to safe initial values, and creates the required information handlers
   *
   */
  vtkCuda2DVolumeMapper();

  /** @brief Destructor which deallocates the various information handlers and matrices
   *
   */
  ~vtkCuda2DVolumeMapper();
  virtual void Reinitialize(int withData = 0);
  virtual void Deinitialize(int withData = 0);

  vtkCuda2DTransferFunctionInformationHandler* transferFunctionInfoHandler;

  cudaArray* SourceData[ VTKCUDAVOLUMEMAPPER_UPPER_BOUND ];

  static vtkMutexLock* tfLock;
private:
  vtkCuda2DVolumeMapper operator=(const vtkCuda2DVolumeMapper&); /**< not implemented */
  vtkCuda2DVolumeMapper(const vtkCuda2DVolumeMapper&); /**< not implemented */
};

#endif
