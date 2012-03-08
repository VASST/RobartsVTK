/*=========================================================================

  Program:   USB Data ECG Box for VTK
  Module:    $RCSfile: vtkVideoLinkedUSBECGBox.h,v $
  Creator:   Chris Wedlake <cwedlake@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: cwedlake $
  Date:      $Date: 2008/12/15 18:36:11 $
  Version:   $Revision: 1.2 $

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
// .NAME vtkVideoLinkedUSBECGBox - interfaces VTK with real-time ECG signal

#ifndef __vtkVideoLinkedUSBECGBox_h
#define __vtkVideoLinkedUSBECGBox_h

#include "vtkObject.h"
#include "vtkCriticalSection.h"

#include "vtkDoubleArray.h"
#include "vtkIntArray.h"
#include "vtkSignalBox.h"

class vtkDoubleArray;
class vtkIntArray;
class vtkMultiThreader;

#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include ".\cbw\cbw.h"
#include <vector>

class VTK_EXPORT vtkVideoLinkedUSBECGBox : public vtkUSBECGBox
{
public:
  static vtkVideoLinkedUSBECGBox *New();
  vtkTypeMacro(vtkVideoLinkedUSBECGBox,vtkSignalBox);
  void PrintSelf(ostream& os, vtkIndent indent);



protected:
  vtkVideoLinkedUSBECGBox();
  ~vtkVideoLinkedUSBECGBox();
//BTX
  vtkMILECGVideoSource2 * videoSource;
//ETX

private:
  vtkVideoLinkedUSBECGBox(const vtkVideoLinkedUSBECGBox&);
  void operator=(const vtkVideoLinkedUSBECGBox&);  
};

#endif