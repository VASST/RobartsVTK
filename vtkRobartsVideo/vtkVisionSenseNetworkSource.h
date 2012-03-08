/*=========================================================================

  File: vtkVisionSenseNetworkSource.h
  Author: Kyle Charbonneau <kcharbon@imaging.robarts.ca>
  Language: C++

=========================================================================*/

#ifndef __vtkVisionSenseNetworkSource_h
#define __vtkVisionSenseNetworkSource_h

#include "vtkObject.h"   


#ifdef _WIN32
  #include <winsock2.h>           //ws2_32.lib must be linked in

#elif defined(__FreeBSD__) || defined(__linux__) || defined(sgi)
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>

  #define SOCKET_ERROR -1

#endif

struct FRAME_HEADER {
  int width;
  int height;
  int bpp;
  short channels;   
};

class vtkMultiThreader;

class VTK_EXPORT vtkVisionSenseNetworkSource : public vtkObject
{
public:

  vtkTypeMacro(vtkVisionSenseNetworkSource, vtkObject);

  static vtkVisionSenseNetworkSource *New();
  void PrintSelf(ostream& os, vtkIndent indent);   

  // Connect to remote address/port
  int Connect();
  // Disconnect from the camera
  void Disconnect();

  // Start streaming images over the network
  void Start();
  // Manually stop streaming images, disconnect will auotmatically stop
  void Stop();

  // Get the height and widht of the images in the buffer
  int GetWidth();
  int GetHeight();

  // Get the current location for the right and left image buffers
  char *GetRightBuffer();
  char *GetLeftBuffer();

  // Set/Get the remote port
  vtkSetMacro(RemotePort, int);
  vtkGetMacro(RemotePort, int);

  // Set/Get the remote address
  void SetRemoteAddress(char *);
  char *GetRemoteAddress() {return this->RemoteAddress;};
  
  // Internal use only
  void GetFrame();

  // Set the max frame rate images should be downloaded at
  vtkSetMacro(MaxFrameRate, int);
  vtkGetMacro(MaxFrameRate, int);

protected:
  vtkVisionSenseNetworkSource();
  ~vtkVisionSenseNetworkSource();

  // Get information from socket
  int RecieveIntoBuffer(char *buffer, int length);
  
  // Switches between buffers to avoid downloading into the same
  // buffer that is being displayed and causing tearing
  void SwitchBuffers();

  // Holds the local socket
  int DataSocket;

  // Connection information
  int RemotePort;
  char* RemoteAddress;

  // Image information and buffers
  FRAME_HEADER ImageHeader;
  int ImageSize;
  char *ImageBuffer1;
  char *ImageBuffer2;

  // Flags to make sure buffer isn't over ridden prematurely
  int SwitchLeft;
  int SwitchRight;
  
  // Current frame downlodaed
  int CurrentFrame;
  
  // Thread info for streaming download
  int ThreadId;
  vtkMultiThreader *Threader;
  
  // State information
  int Connected;
  int Streaming;
  int CurrentBuffer;
  
  int MaxFrameRate;
};

#endif
