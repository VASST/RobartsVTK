/** @file vtkCuda1DTransferFunctionInformationHandler.h
 *
 *  @brief Header file defining an internal class for vtkCudaVolumeMapper which manages information regarding the volume and transfer function
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on May 11, 2012
 *
 */

#ifndef vtkCuda1DTransferFunctionInformationHandler_H_
#define vtkCuda1DTransferFunctionInformationHandler_H_

#include "vtkCudaVisualizationExport.h"

#include "CUDA_container1DTransferFunctionInformation.h"
#include "CudaObject.h"
#include "vtkObject.h"

class vtkColorTransferFunction;
class vtkImageData;
class vtkPiecewiseFunction;
class vtkVolume;

/** @brief vtkCuda1DTransferFunctionInformationHandler handles all volume and transfer function related information on behalf of the CUDA volume mapper to facilitate the rendering process
 *
 */
class vtkCudaVisualizationExport vtkCuda1DTransferFunctionInformationHandler : public vtkObject, public CudaObject
{
public:

  vtkTypeMacro(vtkCuda1DTransferFunctionInformationHandler, vtkObject);

  /** @brief VTK compatible constructor method
   *
   */
  static vtkCuda1DTransferFunctionInformationHandler* New();

  /** @brief Sets the image data associated with a particular frame
   *
   *  @param inputData Input data to be loaded in
   *  @param index The frame number for this image in the 4D sequence
   *
   *  @pre All images added to the volume information handler have the same dimensions and similar intensity and gradient ranges (ie: they are images of the same anatomy from the same imaging modality)
   *  @pre index is a non-negative integer less than or equal to the current total number of frames
   *  @pre index is less than 100
   */
  void SetInputData(vtkImageData* inputData, int index);

  /** @brief Gets the image data associated with a particular frame
   *
   *  @param index The frame number for this image in the 4D sequence
   *
   *  @pre index is a non-negative integer associated with a valid (a.k.a. populated or set) frame
   */
  vtkImageData* GetInputData() const;

  /** @brief Gets the CUDA compatible container for volume/transfer function related information needed during the rendering process
   *
   */
  const cuda1DTransferFunctionInformation& GetTransferFunctionInfo() const;

  /** @brief Set the transfer function used for determining colour in the volume rendering process
   *
   *  @param func The 1 dimensional transfer function (from vtkVolumeProperty)
   *
   *  @note This also resets the lastModifiedTime that the volume information handler has for the transfer function, forcing an updating in the lookup tables for the first render
   */
  void SetColourTransferFunction(vtkColorTransferFunction* func);

  /** @brief Set the transfer function used for determining colour in the volume rendering process
   *
   *  @param func The 1 dimensional transfer function (from vtkVolumeProperty)
   *
   *  @note This also resets the lastModifiedTime that the volume information handler has for the transfer function, forcing an updating in the lookup tables for the first render
   */
  void SetOpacityTransferFunction(vtkPiecewiseFunction* func);

  /** @brief Set the transfer function used for determining colour in the volume rendering process
   *
   *  @param func The 1 dimensional transfer function (from vtkVolumeProperty)
   *
   *  @note This also resets the lastModifiedTime that the volume information handler has for the transfer function, forcing an updating in the lookup tables for the first render
   */
  void SetGradientOpacityTransferFunction(vtkPiecewiseFunction* func);

  vtkGetMacro(UseGradientOpacity, bool);
  vtkSetMacro(UseGradientOpacity, bool);

  /** @brief Triggers an update for the volume information, checking all subsidiary information for modifications
   *
   */
  virtual void Update(vtkVolume* vol);

protected:

  /** @brief Constructor which sets the pointers to the image and volume to null, as well as setting all the constants to safe initial values, and initializes the image holder on the GPU
   *
   */
  vtkCuda1DTransferFunctionInformationHandler();

  /** @brief Destructor which cleans up the volume and image data pointers as well as clearing the GPU array containing the images
   *
   */
  ~vtkCuda1DTransferFunctionInformationHandler();

  /** @brief Update attributes associated with the transfer function after comparing MTimes and determining if the lookup tables have changed since last update
   *
   */
  void UpdateTransferFunction();

  virtual void Deinitialize(bool withData = false);
  virtual void Reinitialize(bool withData = false);

protected:
  vtkImageData*                       InputData;    /**< The 3D image data currently being rendered */
  cuda1DTransferFunctionInformation   TransInfo;    /**< The CUDA specific structure holding the required volume related information for rendering */

  vtkPiecewiseFunction*               OpacityFunction;
  vtkPiecewiseFunction*               GradientOpacityFunction;
  vtkColorTransferFunction*           ColourFunction;
  bool                                UseGradientOpacity;

  unsigned long                       LastModifiedTime;      /**< The last time the transfer function was modified, used to determine when to repopulate the transfer function lookup tables */
  int                                 FunctionSize;  /**< The size of the transfer function which is square */
  double                              HighGradient;  /**< The maximum gradient of the current image */
  double                              LowGradient;  /**< The minimum gradient of the current image */

private:
  vtkCuda1DTransferFunctionInformationHandler& operator=(const vtkCuda1DTransferFunctionInformationHandler&); /**< Not implemented */
  vtkCuda1DTransferFunctionInformationHandler(const vtkCuda1DTransferFunctionInformationHandler&); /**< Not implemented */
};

#endif
