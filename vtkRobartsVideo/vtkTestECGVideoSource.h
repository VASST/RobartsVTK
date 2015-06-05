/*=========================================================================

  Program:   Heart Signal Box for VTK
  Module:    $RCSfile: vtkTestECGSignalBox.h,v $
  Creator:   Chris Wedlake <cwedlake@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: cwedlake $
  Date:      $Date: 2009/09/23 17:31:37 $
  Version:   $Revision: 1.0 $

==========================================================================

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
vBE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, LOSS OF DATA OR DATA BECOMING INACCURATE
OR LOSS OF PROFIT OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF
THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.

=========================================================================*/
// .NAME vtkTestECGSignalBox - signal box for the heart phantom

#ifndef __vtkTestECGVideoSource_h
#define __vtkTestECGVideoSource_h

#include "vtkVideoSource.h"
#include "vtkUnsignedCharArray.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

class VTK_EXPORT vtkTestECGVideoSource : public vtkVideoSource
{
public:
  static vtkTestECGVideoSource *New();
#if (VTK_MAJOR_VERSION <= 5)
  vtkTypeRevisionMacro(vtkTestECGVideoSource,vtkVideoSource);
#else
  vtkTypeMacro(vtkTestECGVideoSource,vtkVideoSource);
#endif
  void PrintSelf(ostream& os, vtkIndent indent);
  // Description:
  // Request a particular frame size (set the third value to 1).
  //void SetFrameSize(int x, int y, int z);

  // Description:
  // Request a particular output format (default: VTK_RGB).
 // void SetOutputFormat(int format);
  void InternalGrab();

  void SetECGPhase(int newPhase);
  int GetECGPhase();
  void SetNumberOfECGPhases(int newTotal);

  void Grab();
  void Record();
  void Stop();
  void Pause();
  void UnPause();


protected:
  vtkTestECGVideoSource();
  ~vtkTestECGVideoSource();

private:

  int phase;
  int totalPhases;

  int pauseFeed;

  vtkTestECGVideoSource(const vtkTestECGVideoSource&);  // Not implemented.
  void operator=(const vtkTestECGVideoSource&);  // Not implemented.
};

#endif
