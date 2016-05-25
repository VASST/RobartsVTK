/*==========================================================================

  Copyright (c) 2014 Xiongbiao Luo, xiongbiao.luo@gmail.com

  Use, modification and redistribution of the software, in source or
  binary forms, are permitted provided that the following terms and
  conditions are met:

  1) Redistribution of the source code, in verbatim or modified
  form, must retain the above copyright notice, this license,
  the following disclaimer, and any notices that refer to this
  license and/or the following disclaimer.

  2) Redistribution in binary form must include the above copyright
  notice, a copy of this license and the following disclaimer
  in the documentation or with other materials provided with the
  distribution.

  3) Modified copies of the source code must be clearly marked as such,
  and must not be misrepresented as verbatim copies of the source code.

  THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE SOFTWARE "AS IS"
  WITHOUT EXPRESSED OR IMPLIED WARRANTY INCLUDING, BUT NOT LIMITED TO,
  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE.  IN NO EVENT SHALL ANY COPYRIGHT HOLDER OR OTHER PARTY WHO MAY
  MODIFY AND/OR REDISTRIBUTE THE SOFTWARE UNDER THE TERMS OF THIS LICENSE
  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, LOSS OF DATA OR DATA BECOMING INACCURATE
  OR LOSS OF PROFIT OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF
  THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGES.
  =========================================================================*/

#include "vtkDisparityMap.h"
#include "vtkObjectFactory.h"

#include "png.h"

#include "vtkXBLImage.h"
#include "vtkSmartPointer.h"

#include <algorithm>
#include <limits>
#include <iostream>
#include <sstream>

//----------------------------------------------------------------------------

vtkStandardNewMacro(vtkDisparityMap);

//----------------------------------------------------------------------------
vtkDisparityMap::vtkDisparityMap(float c, float g, float a, int k, float e)
{
  color_threshold    = c;
  gradient_threshold = g;
  alpha              = a;
  kernel_radius      = k;
  epsilon            = e;
}

//----------------------------------------------------------------------------
vtkDisparityMap::vtkDisparityMap()
  : color_threshold(7),
    gradient_threshold(2),
    alpha(1-0.1f),
    kernel_radius(9),
    epsilon(0.0001*255*255)
{
}

//----------------------------------------------------------------------------
vtkDisparityMap::~vtkDisparityMap()
{

}

//----------------------------------------------------------------------------
void vtkDisparityMap::InverseSymmetric(const float* matrix, float* inverse)
{
  inverse[0] = matrix[4]*matrix[8] - matrix[5]*matrix[7];
  inverse[1] = matrix[2]*matrix[7] - matrix[1]*matrix[8];
  inverse[2] = matrix[1]*matrix[5] - matrix[2]*matrix[4];
  float det = matrix[0]*inverse[0]+matrix[3]*inverse[1]+matrix[6]*inverse[2];
  det = 1/det;
  inverse[0] *= det;
  inverse[1] *= det;
  inverse[2] *= det;
  inverse[3] = inverse[1];
  inverse[4] = (matrix[0]*matrix[8] - matrix[2]*matrix[6]) * det;
  inverse[5] = (matrix[2]*matrix[3] - matrix[0]*matrix[5]) * det;
  inverse[6] = inverse[2];
  inverse[7] = inverse[5];
  inverse[8] = (matrix[0]*matrix[4] - matrix[1]*matrix[3]) * det;
}

//----------------------------------------------------------------------------
vtkXBLImage vtkDisparityMap::PropagateCost(const vtkXBLImage& im1Color, const vtkXBLImage& im2Color, int dispMin, int dispMax)
{
  vtkSmartPointer<vtkXBLImage> im1R = vtkSmartPointer<vtkXBLImage>::New();
  im1Color.r(*im1R);
  vtkSmartPointer<vtkXBLImage> im1G = vtkSmartPointer<vtkXBLImage>::New();
  im1Color.g(*im1G);
  vtkSmartPointer<vtkXBLImage> im1B = vtkSmartPointer<vtkXBLImage>::New();
  im1Color.b(*im1B);

  vtkSmartPointer<vtkXBLImage> im2R = vtkSmartPointer<vtkXBLImage>::New();
  im1Color.r(*im2R);
  vtkSmartPointer<vtkXBLImage> im2G = vtkSmartPointer<vtkXBLImage>::New();
  im1Color.g(*im2G);
  vtkSmartPointer<vtkXBLImage> im2B = vtkSmartPointer<vtkXBLImage>::New();
  im1Color.b(*im2B);

  const int width = im1R->GetWidth();
  const int height = im1R->GetHeight();
  const int r = kernel_radius;

  vtkSmartPointer<vtkXBLImage> disparity = vtkSmartPointer<vtkXBLImage>::New();
  disparity->SetWidth(width);
  disparity->SetHeight(height);
  disparity->AllocateData();

  vtkSmartPointer<vtkXBLImage> cost = vtkSmartPointer<vtkXBLImage>::New();
  cost->SetWidth(width);
  cost->SetHeight(height);
  cost->AllocateData();

  std::fill_n(&(*disparity)(0,0), width*height, dispMin-1);
  std::fill_n(&(*cost)(0,0), width*height, std::numeric_limits<float>::max());

  vtkSmartPointer<vtkXBLImage> im1Gray = vtkSmartPointer<vtkXBLImage>::New();
  cost->SetWidth(width);
  cost->SetHeight(height);
  cost->AllocateData();
  vtkSmartPointer<vtkXBLImage> im2Gray = vtkSmartPointer<vtkXBLImage>::New();
  cost->SetWidth(width);
  cost->SetHeight(height);
  cost->AllocateData();

  Libpng::rgb2gray(&(*im1R)(0,0),&(*im1G)(0,0),&(*im1B)(0,0), width,height, &(*im1Gray)(0,0));
  Libpng::rgb2gray(&(*im2R)(0,0),&(*im2G)(0,0),&(*im2B)(0,0), width,height, &(*im2Gray)(0,0));
  vtkXBLImage gradient1 = im1Gray->XGradient();
  vtkXBLImage gradient2 = im2Gray->XGradient();

  // Compute the mean and variance of each patch, eq. (14)
  vtkXBLImage meanIm1R = im1R.box_filter(r);
  vtkXBLImage meanIm1G = im1G.box_filter(r);
  vtkXBLImage meanIm1B = im1B.box_filter(r);

  vtkXBLImage varIm1RR = Covariance(im1R, meanIm1R, im1R, meanIm1R, r);
  vtkXBLImage varIm1RG = Covariance(im1R, meanIm1R, im1G, meanIm1G, r);
  vtkXBLImage varIm1RB = Covariance(im1R, meanIm1R, im1B, meanIm1B, r);
  vtkXBLImage varIm1GG = Covariance(im1G, meanIm1G, im1G, meanIm1G, r);
  vtkXBLImage varIm1GB = Covariance(im1G, meanIm1G, im1B, meanIm1B, r);
  vtkXBLImage varIm1BB = Covariance(im1B, meanIm1B, im1B, meanIm1B, r);

  vtkXBLImage aR(width,height),aG(width,height),aB(width,height);
  vtkXBLImage dCost(width,height);
  int _count_ = 1;
  for(int d=dispMin; d<=dispMax; d++)
  {
    std::cout << '*' << std::flush;
    ComputeCost(im1R,im1G,im1B, im2R,im2G,im2B, gradient1, gradient2, d, dCost);
    vtkXBLImage meanCost = dCost.box_filter(r); // Eq. (14)

    vtkXBLImage covarIm1RCost = Covariance(im1R, meanIm1R, dCost, meanCost, r);
    vtkXBLImage covarIm1GCost = Covariance(im1G, meanIm1G, dCost, meanCost, r);
    vtkXBLImage covarIm1BCost = Covariance(im1B, meanIm1B, dCost, meanCost, r);

    for(int y=0; y<height; y++)
    {
      for(int x=0; x<width; x++)
      {
        // Computation of (Sigma_k+\epsilon Id)^{-1}
        float S1[3*3] =   // Eq. (21)
        {
          varIm1RR(x,y)+epsilon, varIm1RG(x,y), varIm1RB(x,y),
          varIm1RG(x,y), varIm1GG(x,y)+epsilon, varIm1GB(x,y),
          varIm1RB(x,y), varIm1GB(x,y), varIm1BB(x,y)+epsilon
        };
        float S2[3*3];
        InverseSymmetric(S1, S2);
        // Eq. (19)
        aR(x,y) = covarIm1RCost(x,y) * S2[0] +
                  covarIm1GCost(x,y) * S2[1] +
                  covarIm1BCost(x,y) * S2[2];
        aG(x,y) = covarIm1RCost(x,y) * S2[3] +
                  covarIm1GCost(x,y) * S2[4] +
                  covarIm1BCost(x,y) * S2[5];
        aB(x,y) = covarIm1RCost(x,y) * S2[6] +
                  covarIm1GCost(x,y) * S2[7] +
                  covarIm1BCost(x,y) * S2[8];
      }
    }
    vtkXBLImage b = (meanCost-aR*meanIm1R-aG*meanIm1G-aB*meanIm1B).box_filter(r);
    b += aR.box_filter(r)*im1R+aG.box_filter(r)*im1G+aB.box_filter(r)*im1B;

    // Winner takes all label selection
    for(int y=0; y<height; y++)
    {
      for(int x=0; x<width; x++)
      {
        if(cost(x,y) >= b(x,y))
        {
          cost(x,y) = b(x,y);
          disparity(x,y) = d;
        }
      }
    }

  }
  std::cout << std::endl;
  return disparity;
}

//----------------------------------------------------------------------------
bool vtkDisparityMap::WriteDisparityToPNG(const char* img_name, const vtkXBLImage& disparity, int d_min, int d_max, int gray_min, int gray_max)
{
  const float a=(gray_max-gray_min)/float(d_max-d_min);
  const float b=(gray_min*d_max-gray_max*d_min)/float(d_max-d_min);

  const int w=disparity.width(), h=disparity.height();
  const float* in=&(const_cast<vtkXBLImage&>(disparity))(0,0);
  unsigned char *out = new unsigned char[3*w*h];
  unsigned char *red=out, *green=out+w*h, *blue=out+2*w*h;
  for(size_t i=w*h; i>0; i--, in++, red++)
  {
    if((float)d_min<=*in && *in<=(float)d_max)
    {
      float v = a * *in + b +0.5f;
      if(v<0)
      {
        v=0;
      }
      if(v>255)
      {
        v=255;
      }
      *red = static_cast<unsigned char>(v);
      *green++ = *red;
      *blue++  = *red;
    }
    else
    {
      // Cyan for disparities out of range
      *red=0;
      *green++=255;
      *blue++ = 0;
    }
  }
  bool ok = (Libpng::write_png_u(img_name, out, w, h, 3) == 0);
  delete [] out;
  return ok;
}

//----------------------------------------------------------------------------
vtkXBLImage vtkDisparityMap::Covariance(const vtkXBLImage& im1, const vtkXBLImage& mean1, const vtkXBLImage& im2, const vtkXBLImage& mean2, int r)
{
  return (im1*im2).box_filter(r) - mean1*mean2;
}

//----------------------------------------------------------------------------
float vtkDisparityMap::ComputeColorCost(const vtkXBLImage& im1R, const vtkXBLImage& im1G, const vtkXBLImage& im1B, const vtkXBLImage& im2R, const vtkXBLImage& im2G, const vtkXBLImage& im2B, int x, int y, int d, float maxCost)
{
  float col1[3] = {im1R(x,y), im1G(x,y), im1B(x,y)};
  float col2[3] = {im2R(x+d,y), im2G(x+d,y), im2B(x+d,y)};
  float cost=0;
  for(int i=0; i<3; i++)
  {
    // Eq. (2)
    float tmp = col1[i]-col2[i];
    if(tmp<0)
    {
      tmp=-tmp;
    }
    cost += tmp;
  }
  cost /= 3;
  if(cost > maxCost) // Eq. (3)
  {
    cost = maxCost;
  }
  return cost;
}

//----------------------------------------------------------------------------
float vtkDisparityMap::ComputeGradientCost(const vtkXBLImage& gradient1, const vtkXBLImage& gradient2, int x, int y, int d, float maxCost)
{
  float cost = gradient1(x,y)-gradient2(x+d,y); // Eq. (5)
  if(cost < 0)
  {
    cost = -cost;
  }
  if(cost > maxCost) // Eq. (6)
  {
    cost = maxCost;
  }
  return cost;
}

//----------------------------------------------------------------------------
void vtkDisparityMap::ComputeCost(const vtkXBLImage& im1R, const vtkXBLImage& im1G, const vtkXBLImage& im1B, const vtkXBLImage& im2R, const vtkXBLImage& im2G, const vtkXBLImage& im2B, const vtkXBLImage& gradient1, const vtkXBLImage& gradient2, int d, vtkXBLImage& outCostImage)
{
  const int width=im1R.width(), height=im1R.height();
  for(int y=0; y<height; y++)
  {
    for(int x=0; x<width; x++)
    {
      float costColor = color_threshold; // Color L1 distance
      float costGrad = gradient_threshold; // x-deriv abs diff
      if(0<=x+d && x+d<width)
      {
        costColor = ComputeColorCost(im1R, im1G, im1B, im2R, im2G, im2B, x, y, d, color_threshold);
        costGrad  = ComputeGradientCost(gradient1, gradient2, x, y, d, gradient_threshold);
      }
      // Combination of the two penalties, eq. (7)
      outCostImage(x,y) = (1-alpha)*costColor + alpha*costGrad;
    }
  }
}
