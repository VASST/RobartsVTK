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

#ifndef __vtkRetinex_h__
#define __vtkRetinex_h__

#include "vtkRobartsCommon.h"
#include "vtkRobartsVisualizationModule.h"

#include "vtkObject.h"
#include <vector>

class VTKROBARTSVISUALIZATION_EXPORT vtkRetinex : public vtkObject
{
public:
  static vtkRetinex *New();
  vtkTypeMacro(vtkRetinex, vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

public:
  const std::vector< double >& GetScaleValues() const;
  void SetScaleValues(const std::vector< double >& val);

  /// Gaussian convolution based on the Fast Fourier transform
  double *GaussianConvolution( double *img_data, double *out_data, size_t w, size_t h, double scale );

  /// multi scale Retinex processing
  double *MultiscaleRetinex( double *img_data, double *out_data, int w, int h, double omega );

  /// color restoration
  double *ColorRestore( double *img_data, double *out_data, int w, int h, double *gray );

  /// histogram equalization
  double *HistogramEqualizer( double *img_data, double *out_data, int w, int h, float p_left, float p_right );

protected:
  /// scale values
  std::vector< double > ScaleValues;

private:
  vtkRetinex();
  vtkRetinex( const std::vector< double >& n_scales );
  ~vtkRetinex();
};

#endif // __vtkRetinex_h__