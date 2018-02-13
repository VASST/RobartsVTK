/*==========================================================================

Copyright (c) 2018 Adam Rankin, arankin@robarts.ca

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

#include "vtkRobartsCommon.h"

#include <locale>

namespace
{
  //----------------------------------------------------------------------------
  bool icompare_pred(unsigned char a, unsigned char b)
  {
    return ::tolower(a) == ::tolower(b);
  }

  //----------------------------------------------------------------------------
  bool icompare_pred_w(wchar_t a, wchar_t b)
  {
    return ::towlower(a) == ::towlower(b);
  }
}

//----------------------------------------------------------------------------
int CompareValues(const void *x, const void *y)
{
  double dx, dy;

  dx = *(double*)x;
  dy = *(double*)y;

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

//----------------------------------------------------------------------------
bool IsEqualInsensitive(std::string const& a, std::string const& b)
{
  if (a.length() == b.length())
  {
    return std::equal(b.begin(), b.end(), a.begin(), icompare_pred);
  }
  else
  {
    return false;
  }
}

//----------------------------------------------------------------------------
bool IsEqualInsensitive(std::wstring const& a, std::wstring const& b)
{
  if (a.length() == b.length())
  {
    return std::equal(b.begin(), b.end(), a.begin(), icompare_pred_w);
  }
  else
  {
    return false;
  }
}
