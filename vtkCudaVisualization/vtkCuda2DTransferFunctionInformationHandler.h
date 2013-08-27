/** @file vtkCuda2DTransferFunctionInformationHandler.h
 *
 *  @brief Header file defining an internal class for vtkCudaVolumeMapper which manages information regarding the volume and transfer function
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 28, 2011
 *
 */
#ifndef vtkCuda2DTransferFunctionInformationHandler_H_
#define vtkCuda2DTransferFunctionInformationHandler_H_

#include "vtkObject.h"
#include "vtkImageData.h"
#include "CUDA_container2DTransferFunctionInformation.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkCudaObject.h"

/** @brief vtkCuda2DTransferFunctionInformationHandler handles all volume and transfer function related information on behalf of the CUDA volume mapper to facilitate the rendering process
 *
 */
class vtkCuda2DTransferFunctionInformationHandler : public vtkObject, public vtkCudaObject {
public:
	
	vtkTypeMacro( vtkCuda2DTransferFunctionInformationHandler, vtkObject );
	
	/** @brief VTK compatible constructor method
	 *
	 */
	static vtkCuda2DTransferFunctionInformationHandler* New();

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
	const cuda2DTransferFunctionInformation& GetTransferFunctionInfo() const { return (this->TransInfo); }

	/** @brief Set the transfer function used for determining colour and opacity in the volume rendering process
	 *
	 *  @param func The 2 dimensional transfer function
	 *
	 *  @note This also resets the lastModifiedTime that the volume information handler has for the transfer function, forcing an updating in the lookup tables for the first render
	 */
	void SetTransferFunction(vtkCuda2DTransferFunction* func);

	/** @brief Get the transfer function used for determining colour and opacity in the volume rendering process
	 *
	 */
	vtkCuda2DTransferFunction* GetTransferFunction();

	/** @brief Set the transfer function used for determining colour and opacity in the volume rendering process within the keyhole window
	 *
	 *  @param func The 2 dimensional transfer function
	 *
	 *  @note This also resets the lastModifiedTime that the volume information handler has for the transfer function, forcing an updating in the lookup tables for the first render
	 */
	void SetKeyholeTransferFunction(vtkCuda2DTransferFunction* func);

	/** @brief Get the transfer function used for determining colour and opacity in the volume rendering process within the keyhole window
	 *
	 */
	vtkCuda2DTransferFunction* GetKeyholeTransferFunction();
	
	/** @brief Triggers an update for the volume information, checking all subsidary information for modifications
	 *
	 */
	virtual void Update();

protected:
	
	/** @brief Constructor which sets the pointers to the image and volume to null, as well as setting all the constants to safe initial values, and initializes the image holder on the GPU
	 *
	 */
	vtkCuda2DTransferFunctionInformationHandler();

	/** @brief Destructor which cleans up the volume and image data pointers as well as clearing the GPU array containing the images
	 *
	 */
	~vtkCuda2DTransferFunctionInformationHandler();
	
	/** @brief Update attributes associated with the transfer function after comparing MTimes and determining if the lookup tables have changed since last update
	 *
	 */
	void UpdateTransferFunction();
	
	void Deinitialize(int withData = 0);
	void Reinitialize(int withData = 0);

private:
	vtkCuda2DTransferFunctionInformationHandler& operator=(const vtkCuda2DTransferFunctionInformationHandler&); /**< Not implemented */
	vtkCuda2DTransferFunctionInformationHandler(const vtkCuda2DTransferFunctionInformationHandler&); /**< Not implemented */

private:
	
	vtkImageData*						InputData;		/**< The 3D image data currently being renderered */
	cuda2DTransferFunctionInformation	TransInfo;		/**< The CUDA specific structure holding the required volume related information for rendering */

	vtkCuda2DTransferFunction* function;		/**< The 2 dimensional transfer function used to colour the volume during rendering */
	vtkCuda2DTransferFunction* keyholeFunction;	/**< The 2 dimensional transfer function used to colour the volume during rendering within the keyhole */
	unsigned long lastModifiedTime;			/**< The last time the transfer function was modified, used to determine when to repopulate the transfer function lookup tables */
	int						FunctionSize;	/**< The size of the transfer function which is square */
	double					HighGradient;	/**< The maximum gradient of the current image */
	double					LowGradient;	/**< The minimum gradient of the current image */

};

#endif
