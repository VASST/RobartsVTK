/*=========================================================================

  File: vtkReplayImageVideoSource.h
  Author: Chris Wedlake <cwedlake@imaging.robarts.ca>
  Language: C++

==========================================================================

  Copyright (c) Chris Wedlake, cwedlake@robarts.ca

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
// .NAME vtkReplayImageVideoSource - Replays images as a video feed
// .SECTION Description
//  
// .SECTION Caveats
//  
// .SECTION see also
//  
//

#ifndef __vtkReplayImageVideoSource_h
#define __vtkReplayImageVideoSource_h

//#include <windows.h>
#include <vector>

#include "vtkVideoSource.h"  

#include "vtkMultiThreader.h"

class VTK_EXPORT vtkReplayImageVideoSource : public vtkVideoSource
{
public:

  vtkTypeMacro(vtkReplayImageVideoSource,vtkVideoSource);
  static vtkReplayImageVideoSource *New();
  void PrintSelf(ostream& os, vtkIndent indent);   
  
  // Initialize (source must be set first)
  void Initialize();
  
  // Internal use only
  void ReleaseSystemResources();
  //void UpdateFrameBuffer();
  void InternalGrab();
  
  void Record();
  void Play();
  void Stop();

  int GetRecording() { return this->Playing;}

  void Pause();
  void UnPause();

  void Restart();
  
  void LoadFile(char * filename);
  int LoadFolder(char * folder, char * filetype);
  //  int LoadFolder2(char * folder, char * filetype);
  void Clear();
  void SetClipRegion(int x0, int x1, int y0, int y1, int z0, int z1);

protected:
  vtkReplayImageVideoSource();
  ~vtkReplayImageVideoSource();

  int status;

  std::vector<vtkImageData *> loadedData;
  int imageIndex;

  bool SetFrameSizeAutomatically;

  int pauseFeed;
  int currentLength;
};

#endif
