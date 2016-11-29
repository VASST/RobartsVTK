/** @file vtkCudaDualImageTransferFunctionInformationHandler.h
 *
 *  @brief Header file defining an internal class for vtkCudaVolumeMapper which manages information regarding the volume and transfer function
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 28, 2011
 *
 */
#ifndef vtkCudaDualImageTransferFunctionInformationHandler_H_
#define vtkCudaDualImageTransferFunctionInformationHandler_H_

#include "vtkCudaVisualizationExport.h"
#include "vtkVersionMacros.h"

#include "CUDA_containerDualImageTransferFunctionInformation.h"
#include "CudaObject.h"
#include "vtkObject.h"

class vtkCuda2DTransferFunction;
class vtkImageData;

/** @brief vtkCudaDualImageTransferFunctionInformationHandler handles all volume and transfer function related information on behalf of the CUDA volume mapper to facilitate the rendering process
 *
 */
class vtkCudaVisualizationExport vtkCudaDualImageTransferFunctionInformationHandler : public vtkObject, public CudaObject
{
public:

  vtkTypeMacro(vtkCudaDualImageTransferFunctionInformationHandler, vtkObject);

  /** @brief VTK compatible constructor method
   *
   */
  static vtkCudaDualImageTransferFunctionInformationHandler* New();

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
  vtkImageData* GetInputData() const
  {
    return InputData;
  }

  /** @brief Gets the CUDA compatible container for volume/transfer function related information needed during the rendering process
   *
   */
  const cudaDualImageTransferFunctionInformation& GetTransferFunctionInfo() const
  {
    return (this->TransInfo);
  }

  /** @brief Set the transfer function used for determining colour and opacity in the volume rendering process outside the keyhole window
   *
   *  @param func The 2 dimensional transfer function
   *
   *  @note This also resets the lastModifiedTime that the volume information handler has for the transfer function, forcing an updating in the lookup tables for the first render
   */
  void SetTransferFunction(vtkCuda2DTransferFunction* func);

  /** @brief Get the transfer function used for determining colour and opacity in the volume rendering process outside the keyhole window
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
  vtkCudaDualImageTransferFunctionInformationHandler();

  /** @brief Destructor which cleans up the volume and image data pointers as well as clearing the GPU array containing the images
   *
   */
  ~vtkCudaDualImageTransferFunctionInformationHandler();

  /** @brief Update attributes associated with the transfer function after comparing MTimes and determining if the lookup tables have changed since last update
   *
   */
  void UpdateTransferFunction();

  virtual void Deinitialize(bool withData = false);
  virtual void Reinitialize(bool withData = false);

private:
  vtkCudaDualImageTransferFunctionInformationHandler& operator=(const vtkCudaDualImageTransferFunctionInformationHandler&); /**< Not implemented */
  vtkCudaDualImageTransferFunctionInformationHandler(const vtkCudaDualImageTransferFunctionInformationHandler&); /**< Not implemented */

private:

  vtkImageData*            InputData;    /**< The 3D image data currently being renderered */
  cudaDualImageTransferFunctionInformation  TransInfo;    /**< The CUDA specific structure holding the required volume related information for rendering */

  vtkCuda2DTransferFunction* Function;    /**< The 2 dimensional transfer function used to colour the volume during rendering outside the keyhole */
  vtkCuda2DTransferFunction* keyholeFunction;  /**< The 2 dimensional transfer function used to colour the volume during rendering inside the keyhole*/
  unsigned long lastModifiedTime;      /**< The last time the transfer function was modified, used to determine when to repopulate the transfer function lookup tables */
  int            FunctionSize;  /**< The size of the transfer function which is square */

};

#endif
