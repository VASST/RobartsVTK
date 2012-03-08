#ifndef __CUDARENDERERINFORMATION_H__
#define __CUDARENDERERINFORMATION_H__

#include "vector_types.h"

//! A Datastucture located on the cuda hardware that holds all the information about the renderer.
extern "C"
typedef struct __align__(16)
{
    // The resolution of the rendering screen.
    uint2          Resolution;
    uint2          ActualResolution;

    uchar4*        OutputImage;

	float			ViewToVoxelsMatrix[16];

	// additional clipping planes (6 max)
	int				NumberOfClippingPlanes;
	float			ClippingPlanes[24];

	// gradient shading constants
	float			gradShadeScale;
	float			gradShadeShift;

	// depth shading constants
	float			depthShadeScale;
	float			depthShadeShift;

	//gooch shading constants
	float			darkness;
	float			a;
	float			b;
	float			computedShift;

} cudaRendererInformation;

#endif /* __CUDARENDERERINFORMATION_H__ */
