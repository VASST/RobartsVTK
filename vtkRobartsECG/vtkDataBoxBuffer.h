/*=========================================================================

  Program:   Data Acquisition box for the USB 9800 series for VTK
  Module:    $RCSfile: vtkDataBoxBuffer.h,v $
  Creator:   Chris Wedlake <cwedlake@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: cwedlake $
  Date:      $Date: 2007/04/19 12:48:52 $
  Version:   $Revision: 1.1 $

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
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, LOSS OF DATA OR DATA BECOMING INACCURATE
OR LOSS OF PROFIT OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF
THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.

=========================================================================*/
// .NAME vtkDataBoxBuffer - maintain a circular buffer of matrices

// .SECTION Description
// vtkDataBoxBuffer maintains a list of matrices and their associated
// flags, collected from a vtkTrackerTool.  The list is circular, meaning
// that it has a set maximum size and, after the number of added entries
// is greater than that maximum size, earlier entries are overwritten
// in a first-in, first-out manner.

// .SECTION see also
// vtkTrackerTool vtkTracker

#ifndef __vtkDataBoxBuffer_h
#define __vtkDataBoxBuffer_h

#include <vector>

#include "vtkDoubleArray.h"
#include "vtkIntArray.h"
#include "vtkObject.h"
#include "vtkCriticalSection.h"

class vtkDoubleArray;
class vtkIntArray;

class VTK_EXPORT vtkDataBoxBuffer : public vtkObject
{
public:
  vtkTypeMacro(vtkDataBoxBuffer,vtkObject);
  static vtkDataBoxBuffer *New();

  void PrintSelf(ostream& os, vtkIndent indent);


	void AddItem(double signal, int channel, double time);
	int GetSignal(int i, int channel);
	//void GetVec(double *vec, int i);
	void WriteToFile(const char *filename);

  // Description:
  // Set the size of the buffer, all new transforms are set to unity.
  void SetBufferSize(int n);
  int GetBufferSize() { return this->BufferSize; };

  // Description:
  // Get the number of items in the list (this is not the same as
  // the buffer size, but is rather the number of transforms that
  // have been added to the list).  This will never be greater than
  // the BufferSize.
  int GetNumberOfItems() { return this->NumberOfItems; };

  // Description:
  // Lock the buffer: this should be done before changing or accessing
  // the data in the buffer if the buffer is being used from multiple
  // threads.
  void Lock() { this->Mutex->Lock(); };
  void Unlock() { this->Mutex->Unlock(); };

  // Description:
  // Get the timestamp (in seconds since Jan 1, 1970) for the matrix.
  double GetTimeStamp(int i);

   // Description:
  // Given a timestamp, compute the nearest index.  This assumes that
  // the times monotonically increase as the index decreases.
  int GetIndexFromTime(double time);

  // Description:
  // Make this buffer into a copy of another buffer.  You should
  // Lock both of the buffers before doing this.
  void DeepCopy(vtkDataBoxBuffer *buffer);

  void SetNumberOfChannels(int channels);
  int GetNumberOfChannels();

  /*
	int GetNumCurrentChannels(){return this->numCurrentChannels;}
	void SetNumCurrentChannels(int channels){	
		this->numCurrentChannels = channels;
	  this->MatrixArray->SetNumberOfComponents(this->numCurrentChannels);
	}
*/
protected:
  vtkDataBoxBuffer();
  ~vtkDataBoxBuffer();

  vtkDoubleArray *SignalArray;
  vtkDoubleArray *TimeStampArray;

  vtkCriticalSection *Mutex;

  int BufferSize;
  int NumberOfItems;
  int CurrentIndex;
  double CurrentTimeStamp;
  int numberOfChannels;
  char OutFile[255];
  //BTX
  std::vector<vtkDoubleArray *>  SignalArrayVector;
  //ETX

private:
  vtkDataBoxBuffer(const vtkDataBoxBuffer&);
  void operator=(const vtkDataBoxBuffer&);
};

#endif
