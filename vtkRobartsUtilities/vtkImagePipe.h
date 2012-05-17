/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkImagePipe.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkImagePipe - Video-for-Windows video digitizer
// .SECTION Description
// vtkImagePipe grabs or pushes frames or streaming video over
// a TCP/IP socket, allowing for multiple process VTK pipelines
// .SECTION Caveats
// Not quite sure how endianess will be handled at the moment... Must
// look into that more carefully.
//

#ifndef __vtkImagePipe_h
#define __vtkImagePipe_h

#include "vtkAlgorithm.h"
#include "vtkImageData.h"
#include "vtkMutexLock.h"
#include "vtkReadWriteLock.h"
#include "vtkServerSocket.h"
#include "vtkClientSocket.h"
#include "vtkSocketController.h"
#include "vtkMultiThreader.h"

#include <vector>

class vtkImagePipe : public vtkAlgorithm
{

public:
  static vtkImagePipe *New();
  void PrintSelf(ostream& os, vtkIndent indent);   

  //Internal methods: do not call
  void UpdateFrameBuffer();
  void InternalGrab();

  // Description:
  // Input media to be communicated across the pipe
  void SetInput( vtkImageData* in );
  vtkImageData* GetOutput();
  void Update();

  // Description:
  // Sets the connection properties, such as server status
  // and connected address.
  // Must be called before Initialize()!
  void SetAsServer( bool isServer );
  void SetSourceAddress( char* ipAddress, int portNumber );

  // Description:
  // Initialize the driver (this is called automatically when the
  // first grab is done).
  void Initialize();

  // Description:
  // Free the driver (this is called automatically inside the
  // destructor).
  void ReleaseSystemResources();

protected:
  vtkImagePipe();
  ~vtkImagePipe();

  bool Initialized;
  bool isServer;
  bool serverSet;
  int portNumber;
  char* IPAddress;
  
  vtkMultiThreader* threader;
  int mainServerThread;
  vtkSocketController* controller;
  vtkServerSocket*	serverSocket;
  vtkClientSocket*	clientSocket;
  
  vtkImageData* buffer;
  vtkMutexLock* newThreadLock;
  vtkReadWriteLock* rwBufferLock;
  
  //structures for the read/write lock

  int		ImageSize;

private:
  vtkImagePipe(const vtkImagePipe&);  // Not implemented.
  void operator=(const vtkImagePipe&);  // Not implemented.
  
  void ClientSideUpdate();
  static void* ServerSideUpdate(vtkMultiThreader::ThreadInfo *data);
  static void* FirstServerSideUpdate(vtkMultiThreader::ThreadInfo *data);

};

#endif





