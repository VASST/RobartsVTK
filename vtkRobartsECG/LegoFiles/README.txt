About NXT_USB:
This is a C++ library used to control a LEGO Mindstorms NXT robot over a USB connection.

This code was modified from:
    -NXT++: http://nxtpp.sourceforge.net
    -Device::USB: http://search.cpan.org/~gwadej/Device-USB-0.21

Reqirements:
1.  This code can be used only on a Linux distribution with root access.  Remember to run any software utilizing this code as root.
2.  libusb: http://libusb.sourceforge.net/

Instructions:

Download libusb from:
http://libusb.sourceforge.net
(version 0.1.12)

Then run:
tar zxvf libusb-0.1.12.tar.gz
cd libusb-0.1.12/
./configure CFLAGS=-funsigned-char --prefix=<your libusb directory> --exec-prefix=<your libsub directory>
make
make install

Then add the libusb directory to your $LD_LIBRARY_PATH variable.
Note that the INSTALL.libusb file has information about the installation process.

Another alternative is to use CMake:

SET(CMAKE_CXX_FLAGS
  "-funsigned-char"
)

INCLUDE_DIRECTORIES(
  ...
  <your libusb directory (can be a relative path)>
)

ADD_LIBRARY(<your library name>
  ...
  NXT_USB.cxx
  NXT_USB_linux.cxx
)

FIND_LIBRARY(LIBUSB_LIB
  NAMES libusb-lib
  PATHS
     <your libusb directory (can be a relative path)>/lib
)

TARGET_LINK_LIBRARIES(Lego
  ...
  <your libusb directory (should be an absolute path)>/lib/libusb.so
)

The simplest way to use this library is to keep the four NXT_USB files (NXT_USB.h, NXT_USB.cxx, NXT_USB_linux.h, and NXT_USB_linux.cxx) in the same directory as your program.  Simply #include "NXT_USB.h" and compile as usual.

Contact information:
I would be more than happy to reply to any questions and comments emailed to dpace [at] bwh.harvard.edu.
