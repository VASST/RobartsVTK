/* File for holding multiply-used function declarations for buffer operations
 *
 *
 */

#ifndef VTKMAXFLOWSEGMENTATIONUTILITIES_H
#define VTKMAXFLOWSEGMENTATIONUTILITIES_H

#include "vtkRobartsCommonModule.h"

void zeroOutBuffer(float* buffer, int size);
void setBufferToValue(float* buffer, float value, int size);
void translateBuffer(float* bufferOut, float* bufferIn, float shift, float scale, int size);
void sumBuffer(float* bufferOut, float* bufferIn, int size);
void sumScaledBuffer(float* bufferOut, float* bufferIn, float scale, int size);
void copyBuffer(float* bufferOut, float* bufferIn, int size);
void minBuffer(float* bufferOut, float* bufferIn, int size);
void divBuffer(float* bufferOut, float* bufferIn, int size);
void divAndStoreBuffer(float* bufferOut, float* bufferIn, float value, int size);
void lblBuffer( float* label, float* sink, float* cap, int size );
void constrainBuffer( float* sink, float* cap, int size );
void updateLeafSinkFlow(float* sink, float* inc, float* div, float* label, float CC, int size);
void updateLabel(float* sink, float* inc, float* div, float* label, float CC, int size);
void storeSourceFlowInBuffer(float* working, float* sink, float* div, float* label, float CC, int size);
void storeSinkFlowInBuffer(float* working, float* inc, float* div, float* label, float CC, int size);


void dagmf_storeSourceFlowInBuffer(float* working, float* sink, float* div, float* label, float* source, float* exclude, float CC, float multiplicity, int size);
void dagmf_storeSinkFlowInBuffer(float* working, float* inc, float* div, float* label, float* sink, float multiplicity, float CC, int size);
void dagmf_flowGradientStep(float* sink, float* inc, float* div, float* label, float StepSize, float CC, int size);
void dagmf_applyStep(float* div, float* flowX, float* flowY, float* flowZ, int VX, int VY, int VZ, int size);
void dagmf_computeFlowMag(float* div, float* flowX, float* flowY, float* flowZ, float* smooth, float alpha, int VX, int VY, int VZ, int size );
void dagmf_projectOntoSet(float* div, float* flowX, float* flowY, float* flowZ, int VX, int VY, int VZ, int size);

void ghmf_flowGradientStep(float* sink, float* inc, float* div, float* label, float StepSize, float CC, int size);
void ghmf_applyStep(float* div, float* flowX, float* flowY, float* flowZ, int VX, int VY, int VZ, int size);
void ghmf_computeFlowMag(float* div, float* flowX, float* flowY, float* flowZ, float* smooth, float alpha, int VX, int VY, int VZ, int size );
void ghmf_projectOntoSet(float* div, float* flowX, float* flowY, float* flowZ, int VX, int VY, int VZ, int size);



#endif