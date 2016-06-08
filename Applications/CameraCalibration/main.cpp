/*=========================================================================

Program:   tracking with GUI
Module:    $RCSfile: main.cpp,v $
Creator:   Elvis C. S. Chen <chene@robarts.ca>
Language:  C++
Author:    $Author: Elvis Chen $  
Date:      $Date: 2011/07/04 15:28:30 $
Version:   $Revision: 0.99 $

==========================================================================

Copyright (c) Elvis C. S. Chen, elvis.chen@gmail.com

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

#include <QApplication>
#include <cv.h>

#include <vtksys/CommandLineArguments.hxx>
#include "CameraCalibrationMainWindow.h"

int appMain(int argc, char *argv[])
{
  Q_INIT_RESOURCE(PlusCommonWidgets);

  // Check command line arguments.
  bool printHelp(false);
  std::string configFile;
  std::string trackingChannel;

  vtksys::CommandLineArguments args;
  args.Initialize( argc, argv );

  args.AddArgument("--help", vtksys::CommandLineArguments::NO_ARGUMENT, &printHelp, "Print this help.");
  args.AddArgument( "--config-file", vtksys::CommandLineArguments::EQUAL_ARGUMENT, &configFile, "Name of the PLUS configuration file." );
  args.AddArgument("--tracking-channel", vtksys::CommandLineArguments::EQUAL_ARGUMENT, &trackingChannel, "Name of the tracking data channel.");

  if ( !args.Parse() )
  {
    std::cerr << "Problem parsing arguments." << std::endl;
    std::cout << "Help: " << args.GetHelp() << std::endl;
    exit(EXIT_FAILURE);
  }

  if ( printHelp )
  {
    std::cout << args.GetHelp() << std::endl;
    exit(EXIT_SUCCESS);
  }

  QApplication app(argc, argv);
  app.setOrganizationName("Robarts Research Institute, Canada");
  app.setApplicationName("Camera Calibration");

  CameraCalibrationMainWindow* mainWin = new CameraCalibrationMainWindow();
  mainWin->SetPLUSTrackingChannel(trackingChannel);
  mainWin->show();
  return app.exec();
}

#ifdef _WIN32
#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <tchar.h>

// TODO: remove these two functions when VTK is updated to a version that contains vtksys::Encoding
size_t vtksys_Encoding_wcstombs(char* dest, const wchar_t* str, size_t n)
{
  if(str == 0)
  {
    return (size_t)-1;
  }
  return WideCharToMultiByte(CP_ACP, 0, str, -1, dest, (int)n, NULL, NULL) - 1;
}
vtksys_stl::string vtksys_Encoding_ToNarrow(const vtksys_stl::wstring& wcstr)
{
  vtksys_stl::string str;
  size_t length = vtksys_Encoding_wcstombs(0, wcstr.c_str(), 0) + 1;
  if(length > 0)
  {
    std::vector<char> chars(length);
    if(vtksys_Encoding_wcstombs(&chars[0], wcstr.c_str(), length) > 0)
    {
      str = &chars[0];
    }
  }
  return str; 
}

int WINAPI WinMain(HINSTANCE hInstance,
                   HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine,
                   int nCmdShow)
{
  Q_UNUSED(hInstance);
  Q_UNUSED(hPrevInstance);
  Q_UNUSED(nCmdShow);

  // CommandLineToArgvW has no narrow-character version, so we get the arguments in wide strings
  // and then convert to regular string.
  int argc=0;
  LPWSTR* argvStringW = CommandLineToArgvW(GetCommandLineW(), &argc);

  std::vector< const char* > argv(argc); // usual const char** array used in main() functions
  std::vector< std::string > argvString(argc); // this stores the strings that the argv pointers point to
  for(int i=0; i<argc; i++)
  {
    // TODO: replace this by vtksys::Encoding::ToNarrow when VTK is updated to a version that contains vtksys::Encoding
    argvString[i] = vtksys_Encoding_ToNarrow(argvStringW[i]);
    argv[i] = argvString[i].c_str();
  }

  LocalFree(argvStringW);

  return appMain(argc, const_cast< char** >(&argv[0]));
}
#else

int main(int argc, char *argv[])
{
  return appMain(argc, argv);
}

#endif