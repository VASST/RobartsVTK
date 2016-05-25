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

#include "vtkXBLImage.h"

#include <numeric>
#include <algorithm>

//----------------------------------------------------------------------------
void vtkXBLImage::FreeMemory()
{
  if(Count && --*Count == 0)
  {
    delete Count;
    delete [] ImageData;
  }
}

//----------------------------------------------------------------------------
vtkXBLImage::vtkXBLImage(int width, int height)
  : Count( new int(1) )
  , ImageData( new float[width*height])
{
  Width = width;
  Height = height;
}

//----------------------------------------------------------------------------
vtkXBLImage::vtkXBLImage(float* pix, int width, int height)
  : Count(0),
    ImageData(pix)
{
  Width = width;
  Height = height;
}

//----------------------------------------------------------------------------
vtkXBLImage::vtkXBLImage(const vtkXBLImage& I)
  : Count(I.Count),
    ImageData(I.ImageData)
{
  Width = I.Width;
  Height = I.Height;
  if(Count)
  {
    ++*Count;
  }
}

//----------------------------------------------------------------------------
vtkXBLImage::~vtkXBLImage()
{
  FreeMemory( );
}

//----------------------------------------------------------------------------
void vtkXBLImage::AllocateData()
{
  FreeMemory();

  Count = new int(1);
  if( this->Width > 0 && this->Height > 0 )
  {
    ImageData = new float[this->Width*this->Height];
  }
}

//----------------------------------------------------------------------------
vtkXBLImage& vtkXBLImage::operator=(const vtkXBLImage& I)
{
  if(Count != I.Count)
  {
    FreeMemory( );
    if(I.Count)
    {
      ++*I.Count;
    }
  }
  Count = I.Count;
  ImageData = I.ImageData;
  Width = I.Width;
  Height = I.Height;
  return *this;
}

//----------------------------------------------------------------------------
vtkXBLImage vtkXBLImage::Clone() const
{
  vtkXBLImage I(Width,Height);
  std::copy(ImageData, ImageData+Width*Height, I.ImageData);
  return I;
}

//----------------------------------------------------------------------------
float vtkXBLImage::CalculateSquareL2Distance(int x1,int y1, int x2,int y2) const
{
  return square((*this)(x1,y1)-(*this)(x2,y2) )+
         square((*this)(x1,y1+  Height)-(*this)(x2,y2+  Height))+
         square((*this)(x1,y1+2*Height)-(*this)(x2,y2+2*Height));
}

//----------------------------------------------------------------------------
void vtkXBLImage::ComputeWeightedHistogram(std::vector<float>& tab, int x, int y, int radius,float vMin, const vtkXBLImage& guidance,float sSpace, float sColor) const
{
  std::fill(tab.begin(), tab.end(), 0);
  for(int dy=-radius; dy<=radius; dy++)
  {
    if(0<=y+dy && y+dy<Height)
    {
      for(int dx=-radius; dx<=radius; dx++)
      {
        if(0<=x+dx && x+dx<Width)
        {
          float w = exp(-(dx*dx+dy*dy)*sSpace - guidance.CalculateSquareL2Distance(x,y,x+dx,y+dy)*sColor);
          tab[(int)((*this)(x+dx,y+dy))-vMin] += w;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
vtkXBLImage vtkXBLImage::WeightedMedianFilter(const vtkXBLImage& guidance, const vtkXBLImage& where, int vMin, int vMax, int radius, float sSpace, float sColor) const
{
  sSpace = 1.0f/(sSpace*sSpace);
  sColor = 1.0f/(sColor*sColor);

  const int size=vMax-vMin+1;
  std::vector<float> tab(size);
  vtkXBLImage M(Width,Height);

#ifdef _OPENMP
  #pragma omp parallel for firstprivate(tab)
#endif
  for(int y=0; y<Height; y++)
    for(int x=0; x<Width; x++)
    {
      if(where(x,y)>=vMin)
      {
        M(x,y)=(*this)(x,y);
        continue;
      }
      ComputeWeightedHistogram(tab, x,y, radius, vMin, guidance, sSpace, sColor);
      M(x,y) = vMin + MedianHistogram(tab);
    }
  return M;
}

//----------------------------------------------------------------------------
int vtkXBLImage::MedianHistogram(const std::vector<float>& tab)
{
  float sum = std::accumulate(tab.begin(), tab.end(), 0.0f)/2;
  int d=-1;
  for(float cumul=0; cumul<sum;)
  {
    cumul += tab[++d];
  }
  return d;
}

//----------------------------------------------------------------------------
float& vtkXBLImage::operator()(int i,int j)
{
  return ImageData[j*Width+i];
}

//----------------------------------------------------------------------------
float vtkXBLImage::operator()(int i,int j) const
{
  return ImageData[j*Width+i];
}

//----------------------------------------------------------------------------
vtkXBLImage vtkXBLImage::r(vtkXBLImage& outImage) const
{
  outImage = vtkXBLImage(ImageData+0*Width*Height,Width,Height);
}

//----------------------------------------------------------------------------
vtkXBLImage vtkXBLImage::g(vtkXBLImage& outImage) const
{
  outImage = vtkXBLImage(ImageData+1*Width*Height,Width,Height);
}

//----------------------------------------------------------------------------
vtkXBLImage vtkXBLImage::b(vtkXBLImage& outImage) const
{
  outImage = vtkXBLImage(ImageData+2*Width*Height,Width,Height);
}

//----------------------------------------------------------------------------
vtkXBLImage& vtkXBLImage::operator+=(const vtkXBLImage& I)
{
  float* out=ImageData;
  const float *in=I.ImageData;
  for(int i=Width*Height-1; i>=0; i--)
  {
    *out++ += *in++;
  }
  return *this;
}

//----------------------------------------------------------------------------
vtkXBLImage vtkXBLImage::operator+(const vtkXBLImage& I) const
{
  vtkXBLImage S(Width,Height);
  float* out=S.ImageData;
  const float *in1=ImageData, *in2=I.ImageData;
  for(int i=Width*Height-1; i>=0; i--)
  {
    *out++ = *in1++ + *in2++;
  }
  return S;
}

//----------------------------------------------------------------------------
vtkXBLImage vtkXBLImage::operator-(const vtkXBLImage& I) const
{
  vtkXBLImage S(Width,Height);
  float* out=S.ImageData;
  const float *in1=ImageData, *in2=I.ImageData;
  for(int i=Width*Height-1; i>=0; i--)
  {
    *out++ = *in1++ - *in2++;
  }
  return S;
}

//----------------------------------------------------------------------------
vtkXBLImage vtkXBLImage::operator*(const vtkXBLImage& I) const
{
  vtkXBLImage S(Width,Height);
  float* out=S.ImageData;
  const float *in1=ImageData, *in2=I.ImageData;
  for(int i=Width*Height-1; i>=0; i--)
  {
    *out++ = *in1++ * *in2++;
  }
  return S;
}

//----------------------------------------------------------------------------
vtkXBLImage vtkXBLImage::XGradient() const
{
  vtkXBLImage D(Width,Height);
  float* out=D.ImageData;
  for(int y=0; y<Height; y++)
  {
    const float* in=ImageData+y*Width;
    *out++ = in[1]-in[0];           // Right - current
    for(int x=1; x+1<Width; x++, in++)
    {
      *out++ = .5f*(in[2]-in[0]);  // Right - left
    }
    *out++ = in[1]-in[0];           // Current - left
  }
  return D;
}

//----------------------------------------------------------------------------
void vtkXBLImage::CalculateMedian(int radius, vtkXBLImage& M) const
{
  int size=2*radius+1;
  size *= size;
  float* v = new float[size];
  for(int y=0; y<Height; y++)
    for(int x=0; x<Width; x++)
    {
      int n=0;
      for(int j=-radius; j<=radius; j++)
        if(0<=j+y && j+y<Height)
          for(int i=-radius; i<=radius; i++)
            if(0<=i+x && i+x<Width)
            {
              v[n++] = (*this)(i+x,j+y);
            }
      std::nth_element(v, v+n/2, v+n);
      M(x,y) = v[n/2];
    }
  delete [] v;
}

//----------------------------------------------------------------------------
vtkXBLImage vtkXBLImage::CalculateMedianColor(int radius) const
{
  vtkXBLImage M(Width,3*Height);
  M.Height=Height;
  vtkXBLImage I;
  M.r(I);
  r().median(radius, I);
  I=M.g();
  g().median(radius, I);
  I=M.b();
  b().median(radius, I);
  return M;
}

//----------------------------------------------------------------------------
vtkXBLImage vtkXBLImage::BoxFilter(int radius) const
{
  double* S = new double[Width*Height]; // Use double to mitigate precision loss
  for(int i=Width*Height-1; i>=0; i--)
  {
    S[i] = static_cast<double>(ImageData[i]);
  }

  //cumulative sum table S, eq. (24)
  for(int y=0; y<Height; y++)   //horizontal
  {
    double *in=S+y*Width, *out=in+1;
    for(int x=1; x<Width; x++)
    {
      *out++ += *in++;
    }
  }
  for(int y=1; y<Height; y++)   //vertical
  {
    double *in=S+(y-1)*Width, *out=in+Width;
    for(int x=0; x<Width; x++)
    {
      *out++ += *in++;
    }
  }

  //box filtered image B
  vtkXBLImage B(Width,Height);
  float *out=B.ImageData;
  for(int y=0; y<Height; y++)
  {
    int ymin = std::max(-1, y-radius-1);
    int ymax = std::min(Height-1, y+radius);
    for(int x=0; x<Width; x++, out++)
    {
      int xmin = std::max(-1, x-radius-1);
      int xmax = std::min(Width-1, x+radius);
      // S(xmax,ymax)-S(xmin,ymax)-S(xmax,ymin)+S(xmin,ymin), eq. (25)
      double val = S[ymax*Width+xmax];
      if(xmin>=0)
      {
        val -= S[ymax*Width+xmin];
      }
      if(ymin>=0)
      {
        val -= S[ymin*Width+xmax];
      }
      if(xmin>=0 && ymin>=0)
      {
        val += S[ymin*Width+xmin];
      }
      *out = static_cast<float>(val/((xmax-xmin)*(ymax-ymin))); //average
    }
  }
  delete [] S;
  return B;
}