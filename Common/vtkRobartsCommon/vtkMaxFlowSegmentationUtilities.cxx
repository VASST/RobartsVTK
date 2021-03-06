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

#include "vtkMaxFlowSegmentationUtilities.h"
#include <math.h>

//----------------------------------------------------------------------------
// CPU VERSION OF THE ALGORITHM
//----------------------------------------------------------------------------

void zeroOutBuffer(float* buffer, int size)
{
  for (int x = 0; x < size; x++)
  {
    buffer[x] = 0.0f;
  }
}

void setBufferToValue(float* buffer, float value, int size)
{
  for (int x = 0; x < size; x++)
  {
    buffer[x] = value;
  }
}

void translateBuffer(float* bufferOut, float* bufferIn, float shift, float scale, int size)
{
  for (int x = 0; x < size; x++)
  {
    bufferOut[x] = shift + scale * bufferIn[x];
  }
}

void sumBuffer(float* bufferOut, float* bufferIn, int size)
{
  for (int x = 0; x < size; x++)
  {
    bufferOut[x] += bufferIn[x];
  }
}

void sumScaledBuffer(float* bufferOut, float* bufferIn, float scale, int size)
{
  for (int x = 0; x < size; x++)
  {
    bufferOut[x] += scale * bufferIn[x];
  }
}

void copyBuffer(float* bufferOut, float* bufferIn, int size)
{
  for (int x = 0; x < size; x++)
  {
    bufferOut[x] = bufferIn[x];
  }
}

void minBuffer(float* bufferOut, float* bufferIn, int size)
{
  for (int x = 0; x < size; x++)
  {
    bufferOut[x] = (bufferOut[x] > bufferIn[x]) ? bufferIn[x] : bufferOut[x];
  }
}

void divBuffer(float* bufferOut, float* bufferIn, int size)
{
  for (int x = 0; x < size; x++)
  {
    bufferOut[x] /= bufferIn[x];
  }
}

void divAndStoreBuffer(float* bufferOut, float* bufferIn, float value, int size)
{
  for (int x = 0; x < size; x++)
  {
    bufferOut[x] = bufferIn[x] / value;
  }
}

void lblBuffer(float* label, float* sink, float* cap, int size)
{
  for (int x = 0; x < size; x++)
  {
    label[x] = (sink[x] == cap[x]) ? 1.0f : 0.0f;
  }
}

void constrainBuffer(float* sink, float* cap, int size)
{
  for (int x = 0; x < size; x++)
  {
    sink[x] = (sink[x] > cap[x]) ? cap[x] : sink[x];
  }
}

void updateLeafSinkFlow(float* sink, float* inc, float* div, float* label, float CC, int size)
{
  for (int x = 0; x < size; x++)
  {
    sink[x] = inc[x] - div[x] + label[x] / CC;
  }
}

void updateLabel(float* sink, float* inc, float* div, float* label, float CC, int size)
{
  for (int x = 0; x < size; x++)
  {
    label[x] += CC * (inc[x] - div[x] - sink[x]);
  }
}

void dagmf_storeSourceFlowInBuffer(float* working, float* sink, float* div, float* label, float* source, float* exclude, float CC, float multiplicity, int size)
{
  for (int x = 0; x < size; x++)
  {
    working[x] += (sink[x] + div[x] - source[x] - label[x] / CC) * multiplicity;
  }
}

void storeSinkFlowInBuffer(float* working, float* inc, float* div, float* label, float CC, int size)
{
  for (int x = 0; x < size; x++)
  {
    working[x] = inc[x] - div[x] + label[x] / CC;
  }
}
void dagmf_storeSinkFlowInBuffer(float* working, float* inc, float* div, float* label, float* sink, float multiplicity, float CC, int size)
{
  for (int x = 0; x < size; x++)
  {
    working[x] = inc[x] - div[x] + label[x] / CC + multiplicity * sink[x];
  }
}

void storeSourceFlowInBuffer(float* working, float* sink, float* div, float* label, float CC, int size)
{
  for (int x = 0; x < size; x++)
  {
    working[x] += sink[x] + div[x] - label[x] / CC;
  }
}

void dagmf_flowGradientStep(float* sink, float* inc, float* div, float* label, float StepSize, float CC, int size)
{
  for (int x = 0; x < size; x++)
  {
    div[x] = StepSize * (sink[x] + div[x] - inc[x] - label[x] / CC);
  }
}

void dagmf_applyStep(float* div, float* flowX, float* flowY, float* flowZ, int VX, int VY, int VZ, int size)
{
  for (int x = 0; x < size; x++)
  {
    float currAllowed = div[x];
    float xAllowed = (x % VX) ? div[x - 1] : currAllowed;
    flowX[x] -= (currAllowed - xAllowed);
    float yAllowed = (x / VX % VY) ? div[x - VX] : currAllowed;
    flowY[x] -= (currAllowed - yAllowed);
    float zAllowed = (x >= VX * VY) ? div[x - VX * VY] : currAllowed;
    flowZ[x] -= (currAllowed - zAllowed);
  }
}

void dagmf_computeFlowMag(float* div, float* flowX, float* flowY, float* flowZ, float* smooth, float alpha, int VX, int VY, int VZ, int size)
{
  for (int x = 0; x < size; x++)
  {
    div[x] = flowX[x] * flowX[x] + flowY[x] * flowY[x] + flowZ[x] * flowZ[x];
  }
  for (int x = 0; x < size; x++)
  {
    div[x] += ((x + 1) % VX) ? 0.0f : flowX[x + 1] * flowX[x + 1];
  }
  for (int x = 0; x < size; x++)
  {
    div[x] += (((x + VX) / VX) % VY) ? 0.0f : flowX[x + VX] * flowX[x + VX];
  }
  for (int x = 0; x < size - VX * VY; x++)
  {
    div[x] += flowX[x + VX * VY] * flowX[x + VX * VY];
  }
  for (int x = 0; x < size; x++)
  {
    div[x] = sqrtf(div[x]);
  }
  if (smooth)
    for (int x = 0; x < size; x++)
    {
      div[x] = (div[x] > alpha * smooth[x]) ? alpha * smooth[x] / div[x] : 1.0f;
    }
  else
    for (int x = 0; x < size; x++)
    {
      div[x] = (div[x] > alpha) ? alpha / div[x] : 1.0f;
    }
}

void dagmf_projectOntoSet(float* div, float* flowX, float* flowY, float* flowZ, int VX, int VY, int VZ, int size)
{
  //project flows onto valid smoothness set
  for (int x = 0; x < size; x++)
  {
    float currAllowed = div[x];
    float xAllowed = (x % VX) ? div[x - 1] : -currAllowed;
    flowX[x] *= 0.5f * (currAllowed + xAllowed);
    float yAllowed = (x / VX % VY) ? div[x - VX] : -currAllowed;
    flowY[x] *= 0.5f * (currAllowed + yAllowed);
    float zAllowed = (x >= VX * VY) ? div[x - VX * VY] : -currAllowed;
    flowZ[x] *= 0.5f * (currAllowed + zAllowed);
  }

  //compute divergence
  for (int x = 0; x < size; x++)
  {
    div[x] = flowX[x] + flowY[x] + flowZ[x];
  }
  for (int x = 0; x < size; x++)
  {
    div[x] -= ((x + 1) % VX) ? flowX[x + 1] : 0.0f;
  }
  for (int x = 0; x < size; x++)
  {
    div[x] -= ((x / VX + 1) % VY) ? flowY[x + VX] : 0.0f;
  }
  for (int x = 0; x < size; x++)
  {
    div[x] -= (x < size - VX * VY) ? flowZ[x + VX * VZ] : 0.0f;
  }
}

void ghmf_flowGradientStep(float* sink, float* inc, float* div, float* label, float StepSize, float CC, int size)
{
  for (int x = 0; x < size; x++)
  {
    div[x] = StepSize * (sink[x] + div[x] - inc[x] - label[x] / CC);
  }
}

void ghmf_applyStep(float* div, float* flowX, float* flowY, float* flowZ, int VX, int VY, int VZ, int size)
{
  for (int x = 0; x < size; x++)
  {
    float currAllowed = div[x];
    float xAllowed = (x % VX) ? div[x - 1] : 0.0f;
    flowX[x] *= 0.5f * (currAllowed - xAllowed);
    float yAllowed = (x / VX % VY) ? div[x - VX] : 0.0f;
    flowY[x] *= 0.5f * (currAllowed - yAllowed);
    float zAllowed = (x >= VX * VY) ? div[x - VX * VY] : 0.0f;
    flowZ[x] *= 0.5f * (currAllowed - zAllowed);
  }
}

void ghmf_computeFlowMag(float* div, float* flowX, float* flowY, float* flowZ, float* smooth, float alpha, int VX, int VY, int VZ, int size)
{
  for (int x = 0; x < size; x++)
  {
    div[x] = flowX[x] * flowX[x] + flowY[x] * flowY[x] + flowZ[x] * flowZ[x];
  }
  for (int x = 0; x < size; x++)
  {
    div[x] += ((x + 1) % VX) ? 0.0f : flowX[x + 1] * flowX[x + 1];
  }
  for (int x = 0; x < size; x++)
  {
    div[x] += (((x + VX) / VX) % VY) ? 0.0f : flowX[x + VX] * flowX[x + VX];
  }
  for (int x = 0; x < size - VX * VY; x++)
  {
    div[x] += flowX[x + VX * VY] * flowX[x + VX * VY];
  }
  for (int x = 0; x < size; x++)
  {
    div[x] = sqrtf(div[x]);
  }
  if (smooth)
    for (int x = 0; x < size; x++)
    {
      div[x] = (div[x] > alpha * smooth[x]) ? alpha * smooth[x] / div[x] : 1.0f;
    }
  else
    for (int x = 0; x < size; x++)
    {
      div[x] = (div[x] > alpha) ? alpha / div[x] : 1.0f;
    }
}

void ghmf_projectOntoSet(float* div, float* flowX, float* flowY, float* flowZ, int VX, int VY, int VZ, int size)
{
  //project flows onto valid smoothness set
  for (int x = 0; x < size; x++)
  {
    float currAllowed = div[x];
    float xAllowed = (x % VX) ? div[x - 1] : -currAllowed;
    flowX[x] *= 0.5f * (currAllowed + xAllowed);
    float yAllowed = (x / VX % VY) ? div[x - VX] : -currAllowed;
    flowY[x] *= 0.5f * (currAllowed + yAllowed);
    float zAllowed = (x >= VX * VY) ? div[x - VX * VY] : -currAllowed;
    flowZ[x] *= 0.5f * (currAllowed + zAllowed);
  }

  //compute divergence
  for (int x = 0; x < size; x++)
  {
    div[x] = flowX[x] + flowY[x] + flowZ[x];
  }
  for (int x = 0; x < size; x++)
  {
    div[x] -= (x % VX) ? flowX[x - 1] : 0.0f;
  }
  for (int x = 0; x < size; x++)
  {
    div[x] -= (x / VX % VY) ? flowY[x - VX] : 0.0f;
  }
  for (int x = 0; x < size; x++)
  {
    div[x] -= (x >= VX * VY) ? flowZ[x - VX * VZ] : 0.0f;
  }
}