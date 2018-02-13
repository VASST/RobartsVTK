/** @file CUDA_containerDualImageTransferFunctionInformation.h
 *
 *  @brief File for the volume information holding structure used for volume ray casting
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on May 18th, 2011
 *
 *  @note This is primarily an internal file used by the CUDA_renderAlgo to store and communicate constants
 *
 */

#ifndef __CUDADUALIMAGETRANSFERFUNCTIONINFORMATION_H__
#define __CUDADUALIMAGETRANSFERFUNCTIONINFORMATION_H__

#include <cuda_runtime.h>
#include <vector_types.h>

/** @brief A structure located on the CUDA hardware that holds all the information required about the volume being renderered.
 *
 */
typedef struct __align__(16) {
  // The scale and shift to transform intensity and gradient to indices in the transfer functions
  float      intensity1Low;      /**< Minimum intensity of the first component of the image */
  float      intensity1Multiplier;  /**< Scale factor to normalize first intensities to between 0 and 1 */
  float      intensity2Low;      /**< Minimum intensity of the second component of the image */
  float      intensity2Multiplier;  /**< Scale factor to normalize second intensities to between 0 and 1 */
  unsigned int  functionSize;      /**< The size of the lookup table */
  
  //opague memory back for the transfer function
  cudaArray* alphaTransferArrayDualImage;
  cudaArray* ambientTransferArrayDualImage;
  cudaArray* diffuseTransferArrayDualImage;
  cudaArray* specularTransferArrayDualImage;
  cudaArray* specularPowerTransferArrayDualImage;
  cudaArray* colorRTransferArrayDualImage;
  cudaArray* colorGTransferArrayDualImage;
  cudaArray* colorBTransferArrayDualImage;

  //opague memory back for the keyhole transfer function
  bool      useSecondTransferFunction;
  cudaArray* K_alphaTransferArrayDualImage;
  cudaArray* K_ambientTransferArrayDualImage;
  cudaArray* K_diffuseTransferArrayDualImage;
  cudaArray* K_specularTransferArrayDualImage;
  cudaArray* K_specularPowerTransferArrayDualImage;
  cudaArray* K_colorRTransferArrayDualImage;
  cudaArray* K_colorGTransferArrayDualImage;
  cudaArray* K_colorBTransferArrayDualImage;

} cudaDualImageTransferFunctionInformation;

#endif
