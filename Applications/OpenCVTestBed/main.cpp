/*====================================================================
Copyright(c) 2016 Adam Rankin

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files(the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and / or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
====================================================================*/

// Local includes
#include "OpenCVTestBedMainWindow.h"

// QT includes
#include <QApplication>

// VTK includes
#include <vtksys/CommandLineArguments.hxx>
#include <vtksys/Encoding.hxx>

// PlusLib includes
#include <PlusCommon.h>

// OpenCV includes
#include <cv.h>

// STL includes
#include <iostream>

int appMain(int argc, char* argv[])
{
  // Check command line arguments.
  bool printHelp(false);
  std::string configFile;
  std::string trackingChannel;

  vtksys::CommandLineArguments args;
  args.Initialize(argc, argv);

  args.AddArgument("--help", vtksys::CommandLineArguments::NO_ARGUMENT, &printHelp, "Print this help.");

  if (!args.Parse())
  {
    LOG_ERROR("Unable to parse arguments.");
    exit(EXIT_FAILURE);
  }

  if (printHelp)
  {
    LOG_INFO(args.GetHelp());
    exit(EXIT_SUCCESS);
  }

  QApplication app(argc, argv);
  app.setOrganizationName("Robarts Research Institute, Canada");
  app.setApplicationName("OpenCV Test Bed");

  OpenCVTestBedMainWindow* mainWin = new OpenCVTestBedMainWindow();
  mainWin->show();
  return app.exec();
}

#ifdef _WIN32
#include <shellapi.h>
#include <stdlib.h>
#include <string.h>
#include <tchar.h>
#include <windows.h>

int WINAPI WinMain(HINSTANCE hInstance,
                   HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine,
                   int nCmdShow)
{
  Q_UNUSED(hInstance);
  Q_UNUSED(hPrevInstance);
  Q_UNUSED(nCmdShow);

  // CommandLineToArgvW has no narrow-character version, so we get the arguments in wide strings and then convert to regular string.
  int argc = 0;
  LPWSTR* argvStringW = CommandLineToArgvW(GetCommandLineW(), &argc);

  std::vector<const char*> argv(argc);
  std::vector<std::string> argvString(argc);
  for (int i = 0; i < argc; i++)
  {
    argvString[i] = vtksys::Encoding::ToNarrow(argvStringW[i]);
    argv[i] = argvString[i].c_str();
  }

  LocalFree(argvStringW);

  return appMain(argc, const_cast< char** >(&argv[0]));
}

#else

int main(int argc, char* argv[])
{
  return appMain(argc, argv);
}

#endif