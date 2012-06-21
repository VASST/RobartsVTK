/** @file vtkVolumeMapper.h
 *
 *  @brief Header file defining a volume mapper (ray caster) using CUDA kernels for parallel ray calculation
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
 *  @note First documented on March 29, 2011
 *
 */

#ifndef __VTKCUDADualImageVOLUMEMAPPER_H
#define __VTKCUDADualImageVOLUMEMAPPER_H

#include "vtkCudaVolumeMapper.h"

#include "vtkCudaDualImageTransferFunctionInformationHandler.h"
#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CUDA_containerOutputImageInformation.h"
#include "CUDA_containerDualImageTransferFunctionInformation.h"

/** @brief vtkCudaDualImageVolumeMapper is a volume mapper, taking a set of 3D image data objects, volume and renderer as input and creates a DualImage ray casted projection of the scene which is then displayed to screen
 *
 */
class vtkCudaDualImageVolumeMapper : public vtkCudaVolumeMapper {
public:

	vtkTypeMacro( vtkCudaDualImageVolumeMapper, vtkCudaVolumeMapper );

	/** @brief VTK compatible constructor method
	 *
	 */
	static vtkCudaDualImageVolumeMapper *New();

	virtual void SetInputInternal( vtkImageData * image, int frame);
	virtual void ClearInputInternal();
	virtual void ChangeFrameInternal(unsigned int frame);
	virtual void InternalRender (	vtkRenderer* ren, vtkVolume* vol,
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

protected:
	/** @brief Constructor which initializes the number of frames, rendering type and other constants to safe initial values, and creates the required information handlers
	 *
	 */
	vtkCudaDualImageVolumeMapper();

	/** @brief Destructor which deallocates the various information handlers and matrices
	 *
	 */
	~vtkCudaDualImageVolumeMapper();
	virtual void Reinitialize(int withData = 0);
	virtual void Deinitialize(int withData = 0);
	
	vtkCudaDualImageTransferFunctionInformationHandler* transferFunctionInfoHandler;

	static vtkMutexLock* tfLock;

private:
	vtkCudaDualImageVolumeMapper operator=(const vtkCudaDualImageVolumeMapper&); /**< not implemented */
	vtkCudaDualImageVolumeMapper(const vtkCudaDualImageVolumeMapper&); /**< not implemented */

};

#endif
