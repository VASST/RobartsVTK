# Windows

Path length is still a problem on Windows. To avoid this, have your root development directory be VERY short (for example, C:/d/) and have your RobartsVTK binary directory be very short as well (for example. C:/d/RVTK-bin)

If you are building a debug configuration on Windows, you will run into problems with OpenCV and python27_d.lib (if you've simply installed the python.org python release). Unfortunately, you'll have to disable OpenCV_BUILD_PYTHON2 in the OpenCV CMake configuration.