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

#ifndef __vtkXBLImage_h__
#define __vtkXBLImage_h__

#include "vtkObject.h"
#include "vtkRobartsVisualizationModule.h"

#include <vector>

class vtkXBLImage : public vtkObject
{
public:
  static vtkXBLImage *New();
  vtkTypeMacro(vtkXBLImage, vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

public:
  vtkXBLImage& operator=(const vtkXBLImage& I);
  vtkXBLImage Clone() const;

  vtkSetMacro(Width, int);
  vtkGetMacro(Width, int);
  vtkSetMacro(Height, int);
  vtkGetMacro(Height, int);

  void AllocateData();

  float  operator()(int i,int j) const;
  float& operator()(int i,int j);

  void r(vtkXBLImage& outImage) const;
  void g(vtkXBLImage& outImage) const;
  void b(vtkXBLImage& outImage) const;

  vtkXBLImage operator+(const vtkXBLImage& I) const;
  vtkXBLImage operator-(const vtkXBLImage& I) const;
  vtkXBLImage operator*(const vtkXBLImage& I) const;
  vtkXBLImage& operator+=(const vtkXBLImage& I);

  ///Derivative along x-axis
  vtkXBLImage XGradient() const;

  /// Median filter, write results in \a M
  void CalculateMedian(int radius, vtkXBLImage& M) const;

  /// Median filter for a color image
  vtkXBLImage CalculateMedianColor(int radius) const;

  /// box filter
  vtkXBLImage BoxFilter(int radius) const;
  /// weighted median filter
  vtkXBLImage WeightedMedianFilter(const vtkXBLImage& guidance, const vtkXBLImage& where, int vMin, int vMax, int radius, float sigmaSpace, float sigmaColor) const;

  ///Averaging filter with box of \a radius

  /// Index in histogram \a tab reaching median.
  static int MedianHistogram(const std::vector<float>& tab);

protected:
  ///Calculate square L2 distance
  float CalculateSquareL2Distance(int x1,int y1, int x2,int y2) const;

  ///Compute weighted histogram of image values
  void ComputeWeightedHistogram(std::vector<float>& tab, int x, int y, int radius, float vMin,
                                const vtkXBLImage& guidance, float sSpace, float sColor) const;

  void FreeMemory();

protected:
  int* Count;
  float* ImageData;
  int Width;
  int Height;

private:
  vtkXBLImage(int width, int height);
  vtkXBLImage(float* pix, int width, int height);
  vtkXBLImage(const vtkXBLImage& I);
  ~vtkXBLImage();
};

#endif //__vtkXBLImage_h__