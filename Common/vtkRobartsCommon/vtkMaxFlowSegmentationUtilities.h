/*=========================================================================

Robarts Visualization Toolkit

Copyright (c) 2016 Virtual Augmentation and Simulation for Surgery and Therapy, Robarts Research Institute

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

=========================================================================*/

#ifndef VTKMAXFLOWSEGMENTATIONUTILITIES_H
#define VTKMAXFLOWSEGMENTATIONUTILITIES_H

#include "vtkRobartsCommonExport.h"

void zeroOutBuffer(float* buffer, int size);
void setBufferToValue(float* buffer, float value, int size);
void translateBuffer(float* bufferOut, float* bufferIn, float shift, float scale, int size);
void sumBuffer(float* bufferOut, float* bufferIn, int size);
void sumScaledBuffer(float* bufferOut, float* bufferIn, float scale, int size);
void copyBuffer(float* bufferOut, float* bufferIn, int size);
void minBuffer(float* bufferOut, float* bufferIn, int size);
void divBuffer(float* bufferOut, float* bufferIn, int size);
void divAndStoreBuffer(float* bufferOut, float* bufferIn, float value, int size);
void lblBuffer(float* label, float* sink, float* cap, int size);
void constrainBuffer(float* sink, float* cap, int size);
void updateLeafSinkFlow(float* sink, float* inc, float* div, float* label, float CC, int size);
void updateLabel(float* sink, float* inc, float* div, float* label, float CC, int size);
void storeSourceFlowInBuffer(float* working, float* sink, float* div, float* label, float CC, int size);
void storeSinkFlowInBuffer(float* working, float* inc, float* div, float* label, float CC, int size);

void dagmf_storeSourceFlowInBuffer(float* working, float* sink, float* div, float* label, float* source, float* exclude, float CC, float multiplicity, int size);
void dagmf_storeSinkFlowInBuffer(float* working, float* inc, float* div, float* label, float* sink, float multiplicity, float CC, int size);
void dagmf_flowGradientStep(float* sink, float* inc, float* div, float* label, float StepSize, float CC, int size);
void dagmf_applyStep(float* div, float* flowX, float* flowY, float* flowZ, int VX, int VY, int VZ, int size);
void dagmf_computeFlowMag(float* div, float* flowX, float* flowY, float* flowZ, float* smooth, float alpha, int VX, int VY, int VZ, int size);
void dagmf_projectOntoSet(float* div, float* flowX, float* flowY, float* flowZ, int VX, int VY, int VZ, int size);

void ghmf_flowGradientStep(float* sink, float* inc, float* div, float* label, float StepSize, float CC, int size);
void ghmf_applyStep(float* div, float* flowX, float* flowY, float* flowZ, int VX, int VY, int VZ, int size);
void ghmf_computeFlowMag(float* div, float* flowX, float* flowY, float* flowZ, float* smooth, float alpha, int VX, int VY, int VZ, int size);
void ghmf_projectOntoSet(float* div, float* flowX, float* flowY, float* flowZ, int VX, int VY, int VZ, int size);

#endif