/** @file vtkVolumeMapper.h
 *
 *  @brief Header file defining a volume mapper (ray caster) using CUDA kernels for parallel ray calculation
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on August 13, 2014
 *
 */

#ifndef __VTKCUDADRRImageVOLUMEMAPPER_H
#define __VTKCUDADRRImageVOLUMEMAPPER_H

#include "vtkCudaVisualizationExport.h"

#include "vtkCudaVolumeMapper.h"

class CUDA_containerRendererInformation;
class CUDA_containerVolumeInformation;
class CUDA_containerOutputImageInformation;

/** @brief vtkCudaDRRImageVolumeMapper is a volume mapper, taking a set of 3D image data objects, volume and renderer as input and creates a DRRImage ray casted projection of the scene which is then displayed to screen
 *
 */
class vtkCudaVisualizationExport vtkCudaDRRImageVolumeMapper : public vtkCudaVolumeMapper
{
public:

  vtkTypeMacro(vtkCudaDRRImageVolumeMapper, vtkCudaVolumeMapper);

  /** @brief VTK compatible constructor method
   *
   */
  static vtkCudaDRRImageVolumeMapper* New();

  virtual void SetInputInternal(vtkImageData* image, int frame);
  virtual void ClearInputInternal();
  virtual void ChangeFrameInternal(int frame);
  virtual void InternalRender(vtkRenderer* ren, vtkVolume* vol,
                              const cudaRendererInformation& rendererInfo,
                              const cudaVolumeInformation& volumeInfo,
                              const cudaOutputImageInformation& outputInfo);

  vtkSetMacro(CTIntercept, float)
  vtkGetMacro(CTIntercept, float)
  vtkSetMacro(CTSlope, float)
  vtkGetMacro(CTSlope, float)
  vtkSetMacro(CTOffset, float)
  vtkGetMacro(CTOffset, float)

protected:
  /** @brief Constructor which initializes the number of frames, rendering type and other constants to safe initial values, and creates the required information handlers
   *
   */
  vtkCudaDRRImageVolumeMapper();

  /** @brief Destructor which deallocates the various information handlers and matrices
   *
   */
  ~vtkCudaDRRImageVolumeMapper();
  virtual void Reinitialize(bool withData = false);
  virtual void Deinitialize(bool withData = false);

  float CTIntercept;
  float CTSlope;
  float CTOffset;

  cudaArray* SourceData[ VTKCUDAVOLUMEMAPPER_UPPER_BOUND ];

private:
  vtkCudaDRRImageVolumeMapper operator=(const vtkCudaDRRImageVolumeMapper&); /**< not implemented */
  vtkCudaDRRImageVolumeMapper(const vtkCudaDRRImageVolumeMapper&); /**< not implemented */

};

#endif
