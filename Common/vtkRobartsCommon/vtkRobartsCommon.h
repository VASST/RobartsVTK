/*==========================================================================

  Copyright (c) 2016 Adam Rankin, arankin@robarts.ca

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

#ifndef __vtkRobartsCommon_h__
#define __vtkRobartsCommon_h__

template<class T>
int CompareValues( const void *x, const void *y )
{
  double dx, dy;

  dx = *(T *)x;
  dy = *(T *)y;

  if (dx < dy)
  {
    return -1;
  }
  else if (dx > dy)
  {
    return +1;
  }

  return 0;
}

template<class T>
void SplitImageIntoChannels(unsigned char *data_in, int img_size, T* R, T* G, T* B )
{
  for ( int i = 0; i < img_size; i++)
  {
    R[i] = (T)data_in[i]+1.;
    G[i] = (T)data_in[ img_size + i ] + 1.0;
    B[i] = (T)data_in[ 2 * img_size + i ] + 1.0;
  }
}

template<class T>
void RGBToGreyscale( T* R, T* G, T* B, int img_size, T* gray )
{
  for( int i = 0; i < img_size; i++ )
  {
    gray[i]=( R[i] + G[i] + B[i] )/3.0;
  }
}

template<class T>
void RGBToRGBCuchar(T* R, T* G, T* B, unsigned char *output, int img_size )
{
  for ( int i = 0; i < img_size; i++ )
  {
    output[i] = (unsigned char)( R[i] + 0.5f );
    output[img_size + i] = (unsigned char)( G[i] + 0.5f );
    output[2*img_size + i] = (unsigned char)( B[i] + 0.5f );
  }
}

#endif //__vtkRobartsCommon_h__