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

#ifndef __vtkDisparityMap_h__
#define __vtkDisparityMap_h__

#include "vtkRobartsVisualizationModule.h"
#include "vtkObject.h"

#include "vtkXBLImage.h"

class VTKROBARTSVISUALIZATION_EXPORT vtkDisparityMap : public vtkObject
{
public:
  static vtkDisparityMap *New();
  vtkTypeMacro(vtkDisparityMap, vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

public:
  ///Inverse of symmetric 3x3 matrix
  void InverseSymmetric(const float* matrix, float* inverse);

  ///Cost volume filtering
  vtkXBLImage PropagateCost(const vtkXBLImage& im1Color, const vtkXBLImage& im2Color, int dispMin, int dispMax);

  ///save the estimate disparity
  bool WriteDisparityToPNG(const char* img_name, const vtkXBLImage& disparity, int d_min, int d_max, int gray_min, int gray_max);

protected:
  /// Covariance of patches of radius \a r between images, eq. (14).
  vtkXBLImage Covariance(const vtkXBLImage& im1, const vtkXBLImage& mean1, const vtkXBLImage& im2, const vtkXBLImage& mean2, int r);

  /// Compute color cost according to eq. (3).
  float ComputeColorCost(const vtkXBLImage& im1R, const vtkXBLImage& im1G, const vtkXBLImage& im1B, const vtkXBLImage& im2R, const vtkXBLImage& im2G, const vtkXBLImage& im2B, int x, int y, int d, float maxCost);

  /// Compute gradient cost according to eq. (6).
  float ComputeGradientCost(const vtkXBLImage& gradient1, const vtkXBLImage& gradient2, int x, int y, int d, float maxCost);

  /// Compute image of matching costs at disparity d.
  void ComputeCost(const vtkXBLImage& im1R, const vtkXBLImage& im1G, const vtkXBLImage& im1B, const vtkXBLImage& im2R, const vtkXBLImage& im2G,
                   const vtkXBLImage& im2B, const vtkXBLImage& gradient1, const vtkXBLImage& gradient2, int d, vtkXBLImage& outCostImage);

protected:
  float color_threshold;
  float gradient_threshold;
  float alpha;
  int   kernel_radius;
  float epsilon;

private:
  vtkDisparityMap();
  vtkDisparityMap(float c, float g, float a, int k, float e);
  ~vtkDisparityMap();
};

#endif
