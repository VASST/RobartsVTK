/** @file CUDA_container2DTransferFunctionInformation.h
 *
 *  @brief File for the volume information holding structure used for volume ray casting
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 27, 2011
 *
 *  @note This is primarily an internal file used by the vtkCudaVolumeInformationHandler and CUDA_renderAlgo to store and communicate constants
 *
 */

#ifndef __CUDA2DTRANSFERFUNCTIONINFORMATION_H__
#define __CUDA2DTRANSFERFUNCTIONINFORMATION_H__

#include <cuda_runtime.h>
#include <vector_types.h>

/** @brief A structure located on the CUDA hardware that holds all the information required about the volume being renderered.
 *
 */
typedef struct __align__(16) {
  // The scale and shift to transform intensity and gradient to indices in the transfer functions
  float      intensityLow;      /**< Minimum intensity of the image */
  float      intensityMultiplier;  /**< Scale factor to normalize intensities to between 0 and 1 */
  float      gradientLow;      /**< The minimum logarithmic gradient index including the offset */
  float      gradientMultiplier;    /**< The maximum logarithmic gradient index including the offset */
  float      gradientOffset;      /**< The offset for the logarithmic scaling of the gradient indexes */
  unsigned int  functionSize;      /**< The size of the lookup table */

  //opaque memory back for the transfer function
  cudaArray* alphaTransferArray2D;
  cudaArray* ambientTransferArray2D;
  cudaArray* diffuseTransferArray2D;
  cudaArray* specularTransferArray2D;
  cudaArray* specularPowerTransferArray2D;
  cudaArray* colorRTransferArray2D;
  cudaArray* colorGTransferArray2D;
  cudaArray* colorBTransferArray2D;

  //opaque memory back for the keyhole transfer function
  bool useSecondTransferFunction;
  cudaArray* K_alphaTransferArray2D;
  cudaArray* K_ambientTransferArray2D;
  cudaArray* K_diffuseTransferArray2D;
  cudaArray* K_specularTransferArray2D;
  cudaArray* K_specularPowerTransferArray2D;
  cudaArray* K_colorRTransferArray2D;
  cudaArray* K_colorGTransferArray2D;
  cudaArray* K_colorBTransferArray2D;

} cuda2DTransferFunctionInformation;

#endif
