/** @file vtkCudaRendererInformationHandler.h
 *
 *  @brief Header file defining an internal class for vtkCudaVolumeMapper which manages information regarding the renderer, camera, shading model and other objects
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 29, 2011
 *
 */

#ifndef VTKCUDARENDERERINFORMATIONHANDLER_H_
#define VTKCUDARENDERERINFORMATIONHANDLER_H_

#include "vtkObject.h"
#include "vtkMatrix4x4.h"
#include "vtkRenderer.h"
#include "vtkPlaneCollection.h"
#include "vtkCudaMemoryTexture.h"
#include "vtkImageData.h"

#include "CUDA_containerRendererInformation.h"
#include "vtkCudaObject.h"

/** @brief vtkCudaRendererInformationHandler handles all renderer, shading, geometry and camera related information on behalf of the CUDA volume mapper to facilitate the rendering process
 *
 */
class vtkCudaRendererInformationHandler : public vtkObject, public vtkCudaObject {
public:

  vtkTypeMacro( vtkCudaRendererInformationHandler, vtkObject );

  /** @brief VTK compatible constructor method
   *
   */
  static vtkCudaRendererInformationHandler* New();
  
  /** @brief Get the renderer that the handler is currently collecting information from
   *
   */
  vtkRenderer* GetRenderer();

  /** @brief Set the renderer that the handler will collect information from
   *
   *  @param renderer A vtkRenderer associated with the mapper in use
   */
  void SetRenderer(vtkRenderer* renderer);
  
  /** @brief Gets the CUDA compatible container for renderer/camera/shading/geometry related information needed during the rendering process
   *
   */
  const cudaRendererInformation& GetRendererInfo() { return (this->RendererInfo); }
  
  /** @brief Set the strength and sensitivity parameters of the nonphotorealistic shading model which is given to the renderer information handler
   *
   *  @param darkness Floating point between 0.0f and 1.0f inclusive, where 0.0f means no shading, and 1.0f means maximal shading
   *  @param a The shading start value
   *  @param b The shading stop value
   */
  void SetCelShadingConstants(float darkness, float a, float b);

  /** @brief Set the strength and sensitivity parameters of the nonphotorealistic shading model which is given to the renderer information handler
   *
   *  @param darkness Floating point between 0.0f and 1.0f inclusive, where 0.0f means no shading, and 1.0f means maximal shading
   *  @param a The shading start value
   *  @param b The shading stop value
   */
  void SetDistanceShadingConstants(float darkness, float a, float b);

  /** @brief Sets the view to voxels matrix, which is used in rendering to convert rays in view space to rays in voxel space necessary for ray casting
   *
   *  @param m The 4x4 matrix representing the transformation from view space to voxel space
   */
  void SetViewToVoxelsMatrix(vtkMatrix4x4* m);

  /** @brief Sets the voxels to world matrix, which is used to convert the clipping planes to voxel space, using them to clip the ray in the kernel
   *
   *  @param m The 4x4 matrix representing the transformation from world space to voxel space
   *
   *  @pre m represents an affine transform, meaning that no perspective components are included in the matrix (bottom row)
   */
  void SetVoxelsToWorldMatrix(vtkMatrix4x4* m);

  /** @brief Sets the world to world matrix, which is used to convert the clipping planes to voxel space, using them to clip the ray in the kernel
   *
   *  @param m The 4x4 matrix representing the transformation from view voxel to world space
   *
   *  @pre m represents an affine transform, meaning that no perspective components are included in the matrix (bottom row)
   */
  void SetWorldToVoxelsMatrix(vtkMatrix4x4* m);
  
  /** @brief Gets the Z buffer from the render window, and loads it into a CUDA 2D texture for use during rendering
   *
   */
  void LoadZBuffer();

  /** @brief Sets the user-defining clipping planes used to bound the volume during rendering (Can get the planes from the vtkBoxWidget)
   *
   *  @param planes A set of 6 planes acting as the clipping planes
   *
   *  @pre The 6 planes can be split into three groups, where each plane is parallel to those in the group, and orthogonal to all those not in the group
   */
  void SetClippingPlanes(vtkPlaneCollection* planes);

  /** @brief Sets the user-defining keyhole planes used to exclude a region in the volume during rendering (Can get the planes from the vtkBoxWidget)
   *
   *  @param planes A set of 6 planes acting as the keyhole planes
   *
   *  @pre The 6 planes can be split into three groups, where each plane is parallel to those in the group, and orthogonal to all those not in the group
   */
  void SetKeyholePlanes(vtkPlaneCollection* planes);
  
  /** @brief Figures out how to translate information from the set of planes to the arrays used in rendering
   *
   *  @param planes A set of 6 planes
   *
   *  @pre The 6 planes can be split into three groups, where each plane is parallel to those in the group, and orthogonal to all those not in the group
   */
  void FigurePlanes(vtkPlaneCollection* planes, float* planesArray, int* numberOfPlanes);

  /** @brief Updates the various available rendering parameters, repopulating the information container
   *
   */
  virtual void Update();

protected:
  
  /** @brief Constructor which clears all render information, as well as loading the random offset array into CUDA (used to eliminate striping artifacts)
   *
   */
  vtkCudaRendererInformationHandler();
  
  void Deinitialize(int withData = 0);
  void Reinitialize(int withData = 0);

private:
  vtkCudaRendererInformationHandler& operator=(const vtkCudaRendererInformationHandler&); /**< not implemented */
  vtkCudaRendererInformationHandler(const vtkCudaRendererInformationHandler&); /**< not implemented */

private:
  vtkRenderer*      Renderer;          /**< The vtkRenderer which information is currently being extracted from */

  cudaRendererInformation  RendererInfo;        /**< CUDA compatible container for various renderer/shading/camera/geometry information */

  float          WorldToVoxelsMatrix[16];  /**< Array representing the world to voxels transformation as a matrix */
  float          VoxelsToWorldMatrix[16];  /**< Array representing the voxels to world transformation as a matrix */
  float*          ZBuffer;          /**< Address of the Z Buffer in CPU space */
  unsigned int      clipModified;        /**< Determines whether the clipping plane set has been modified and needs reloading */
};

#endif
