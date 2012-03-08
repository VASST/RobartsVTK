#ifndef __CUDAVOLUMEINFORMATION_H__
#define __CUDAVOLUMEINFORMATION_H__

#include "vector_types.h"

//! A datastructure located on the cudacard that holds the information of the volume.
extern "C"
typedef struct __align__(16) {
    void*           SourceData;
    
    // The size and spacing of the volume
    int3            VolumeSize;
	float			Bounds[6];
	float3			SpacingReciprocal;

    //! The scale and shift to transform intensity and gradient to indices in the transfer functions
	float			intensityLow;
	float			intensityMultiplier;
	float			twiceGradientMultiplier;

    //float           Ambient;        //!< Ambient color part
    //float           Diffuse;        //!< Diffuse color part
    //float           Specular;       //!< Specular color part
    //float           SpecularPower;  //!< The power of the specular color

    //! The stepping accuracy to raster along the ray.
    //float           SampleDistance;

} cudaVolumeInformation;
#endif /* __CUDAVOLUMEINFORMATION_H__ */
