/** @file vtkCudaVolumeInformationHandler.h
 *
 *  @brief Header file defining an internal class for vtkCudaVolumeMapper which manages information regarding the volume and transfer function
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
 *  @note First documented on March 28, 2011
 *
 */
#ifndef VTKCUDAVOLUMEINFORMATIONHANDLER_H_
#define VTKCUDAVOLUMEINFORMATIONHANDLER_H_

#include "vtkObject.h"
#include "vtkVolume.h"
#include "vtkImageData.h"
#include "CUDA_containerVolumeInformation.h"

/** @brief vtkCudaVolumeInformationHandler handles all volume and transfer function related information on behalf of the CUDA volume mapper to facilitate the rendering process
 *
 */
class vtkCudaVolumeInformationHandler : public vtkObject {
public:
	
	/** @brief VTK compatible constructor method
	 *
	 */
	static vtkCudaVolumeInformationHandler* New();

	/** @brief Gets the vtkVolume object currently associated with this set of image data
	 *
	 */
	vtkVolume* GetVolume();

	/** @brief Sets the vtkVolume object currently associated with this set of image data
	 *
	 *  @param volume A vtkVolume object not currently used by any other mapper
	 */
	void SetVolume(vtkVolume* volume);

	/** @brief Sets the image data associated with a particular frame
	 *
	 *  @param inputData Input data to be loaded in
	 *  @param index The frame number for this image in the 4D sequence
	 *
	 *  @pre All images added to the volume information handler have the same dimensions and similar intensity and gradient ranges (ie: they are images of the same anatomy from the same imaging modality)
	 *  @pre index is a non-negative integer less than or eaul to the current total number of frames
	 *  @pre index is less than 100
	 */
	void SetInputData(vtkImageData* inputData, int index);
	
	/** @brief Gets the image data associated with a particular frame
	 *
	 *  @param index The frame number for this image in the 4D sequence
	 *
	 *  @pre index is a non-negative integer associated with a valid (a.k.a. populated or set) frame
	 */
	vtkImageData* GetInputData() const { return InputData; }

	/** @brief Gets the CUDA compatible container for volume/transfer function related information needed during the rendering process
	 *
	 */
	const cudaVolumeInformation& GetVolumeInfo() const { return (this->VolumeInfo); }

	/** @brief Clear all information about the volumes
	 *
	 *  @note This also resets the lastModifiedTime that the volume information handler has for the transfer function, forcing an updating in the lookup tables for the first render
	 */
	void ClearInput();
	
	/** @brief Triggers an update for the volume information, checking all subsidary information for modifications
	 *
	 */
	virtual void Update();

protected:
	
	/** @brief Constructor which sets the pointers to the image and volume to null, as well as setting all the constants to safe initial values, and initializes the image holder on the GPU
	 *
	 */
	vtkCudaVolumeInformationHandler();

	/** @brief Destructor which cleans up the volume and image data pointers as well as clearing the GPU array containing the images
	 *
	 */
	~vtkCudaVolumeInformationHandler();

	/** @brief Update information regarding the image currently being displayed, using the index to control switching images
	 *
	 *  @param index The frame number that you wish to update and render
	 *
	 *  @pre index is a non-negative integer less than the current total number of frames
	 *  @pre index is less than 100
	 */
	void UpdateImageData(int index);

private:
	vtkCudaVolumeInformationHandler& operator=(const vtkCudaVolumeInformationHandler&); /**< Not implemented */
	vtkCudaVolumeInformationHandler(const vtkCudaVolumeInformationHandler&); /**< Not implemented */

private:
	
	vtkImageData*			InputData;		/**< The 3D image data currently being renderered */
	vtkVolume*				Volume;			/**< The volume defining how to render this image, such as position in space, etc... */
	cudaVolumeInformation	VolumeInfo;		/**< The CUDA specific structure holding the required volume related information for rendering */

	unsigned long lastModifiedTime;			/**< The last time the transfer function was modified, used to determine when to repopulate the transfer function lookup tables */

};

#endif
