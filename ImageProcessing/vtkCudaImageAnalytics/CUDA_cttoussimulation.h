#ifndef __CTTOUSSIMULATION_H__
#define __CTTOUSSIMULATION_H__

#include "CudaCommon.h"

typedef struct __align__(16)
{

  // The resolution of the rStartering screen.
  uint3 Resolution;
  uint3 VolumeSize;
  float3 spacing;

  //the world to volume transformation
  float WorldToVolume[16];

  //the pose and structure of the US
  float UltraSoundToWorld[16];
  float2 probeWidth;
  float2 fanAngle;
  float StartDepth;
  float EndDepth;
  
  //input scaling to Hounsfield units
  float hounsfieldScale;
  float hounsfieldOffset;

  //output scaling parameters
  float a;
  float alpha;
  float beta;
  float bias;

  //threshold value for total reflection (in Hounsfield Units)
  float reflectionThreshold;

  //whether or not to compute alpha, beta and bias optimally given an input image
  bool optimalParam;
  float crossCorrelation;

} CT_To_US_Information;

void CUDAsetup_unloadCTImage(cudaStream_t* stream);

void CUDAsetup_loadCTImage( float* CTImage, CT_To_US_Information& information, cudaStream_t* stream);

void CUDAsetup_unloadUSImage(cudaStream_t* stream);

void CUDAsetup_loadUSImage( unsigned char* USImage, int resolution[3], cudaStream_t* stream);

void CUDAalgo_simulateUltraSound( float* outputDensity, float* outputTransmission, float* outputReflection, unsigned char* outputUltrasound,
                  CT_To_US_Information& information, cudaStream_t* stream );

#endif //__CTTOUSSIMULATION_H__