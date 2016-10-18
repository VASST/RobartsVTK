/** @file vtkCudaOutputImageInformationHandler.h
 *
 *  @brief Header file defining an internal class for vtkCudaVolumeMapper which manages information regarding the image being outputted
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 28, 2011
 *
 */

#ifndef VTKCUDAOUTPUTIMAGEINFORMATIONHANDLER_H_
#define VTKCUDAOUTPUTIMAGEINFORMATIONHANDLER_H_

#include "vtkCudaVisualizationExport.h"

#include "CUDA_containerOutputImageInformation.h"
#include "CudaObject.h"
#include "vtkObject.h"

class vtkCudaMemoryTexture;
class vtkImageData;
class vtkRayCastImageDisplayHelper;
class vtkRenderer;
class vtkVolume;

/** @brief vtkCudaOutputImageInformationHandler handles all output image, buffering, texturing and OpenGL related information on behalf of the CUDA volume mapper to facilitate the rendering and display process
 *
 */
class vtkCudaVisualizationExport vtkCudaOutputImageInformationHandler : public vtkObject, public CudaObject
{
public:

  vtkTypeMacro( vtkCudaOutputImageInformationHandler, vtkObject );

  /** @brief VTK compatible constructor method
   *
   */
  static vtkCudaOutputImageInformationHandler* New();

  /** @brief Get the renderer that the handler is currently collecting information from
   *
   */
  vtkRenderer* GetRenderer();

  /** @brief Set the renderer that the handler will collect information from
   *
   *  @param renderer A vtkRenderer associated with the mapper in use
   */
  void SetRenderer(vtkRenderer* renderer);

  /** @brief Sets how the image is displayed
   *
   *  @param scaleFactor The factor by which the screen is under-sampled in each direction (must be equal or greater than 1.0f, where 1.0f means full sampling)
   */
  void SetRenderOutputScaleFactor(float scaleFactor);

  /** @brief Sets how the image is displayed
   *
   *  @param type An integer between 0 and 2 inclusive representing the way the image is displayed
   *
   *  @note 0 means using CUDA-OpenGL interoperability, which is fast, but not usable on all computers or in some situations
   *  @note 1 means using full VTK compatibility, which is slower, but more supported
   *  @note 2 means using vtkImageData, so the result is given to a vtkImageData object, rather than the render window
   *
   */
  void SetRenderType(int type);

  /** @brief Gets the CUDA compatible container for the output image buffer location needed during rendering, and the additional information needed after rendering for displaying
   *
   */
  const cudaOutputImageInformation& GetOutputImageInfo();

  /** @brief Prepares the buffers/textures/images before rendering
   *
   */
  void Prepare();

  /** @brief Displays the buffers/textures/images to the render window after the ray casting process
   *
   */
  void Display(vtkVolume* volume, vtkRenderer* renderer);

  /** @brief Accessor method for collecting the image data if the renderer is using render type 2
   *
   *  @pre The render type is equal to 2
   */
  vtkImageData* GetCurrentImageData();

  /** @brief Updates the various available rendering parameters, reconstructing the buffers/textures/images if the render type or output image resolution has changed
   *
   */
  void Update();

  vtkSetMacro(ImageFlipped,bool);
  vtkGetMacro(ImageFlipped,bool);

  void SetTint(unsigned char RGBA[4]);
  void GetTint(unsigned char RGBA[4]);

protected:

  /** @brief Constructor which initializes all the display parameters to safe values, and create a display helper and a CUDA memory texture to help with the display process
   *
   */
  vtkCudaOutputImageInformationHandler();

  /** @brief Destructor that deallocates the displayer helper and memory texture
   *
   */
  ~vtkCudaOutputImageInformationHandler();

  void Deinitialize(int withData = 0);
  void Reinitialize(int withData = 0);

private:
  vtkCudaOutputImageInformationHandler& operator=(const vtkCudaOutputImageInformationHandler&); /**< not implemented */
  vtkCudaOutputImageInformationHandler(const vtkCudaOutputImageInformationHandler&); /**< not implemented */

private:
  cudaOutputImageInformation      OutputImageInfo;      /**< CUDA compatible container for the output image display information */
  vtkRayCastImageDisplayHelper*   Displayer;          /**< A VTK class solely for helping ray casters render a 2D RGBA image to the appropriate section of the render window */
  vtkRenderer*                    Renderer;          /**< The vtkRenderer which information is currently being extracted from */

  int                             OldRenderType;        /**< The render type used previous to the current one, used to clean up information when switching display type */
  uint2                           OldResolution;        /**< The previous window size (used to determine whether or not to recreate buffers) */

  vtkCudaMemoryTexture*           MemoryTexture;        /**< The texture that will be textured to the screen if OpenGL-CUDA interoperability is used */

  uchar4*                         HostOutputImage;                  /**< The image that will be textured to the screen stored on host memory */
  uchar4*                         DeviceOutputImage;                  /**< The image that will be textured to the screen stored on device memory */

  float                           RenderOutputScaleFactor;  /**< The approximate factor by which the screen is resized in order to speed up the rendering process*/
  bool                            ImageFlipped;        /**< Boolean to describe whether the output image is flipped */
  uchar4                          ImageTint;          /**< Tint applied to the output image */

};

#endif
