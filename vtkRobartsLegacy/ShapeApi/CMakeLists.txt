PROJECT(VTKSHAPEAPI)

INCLUDE_DIRECTORIES(${VTKSHAPEAPI_SOURCE_DIR})
INCLUDE_DIRECTORIES(${VTKSHAPEAPI_BINARY_DIR})

SET(ShapeTape_SRCS

3dmath.cpp
Arm.cpp
Back.cpp
BodyPart.cpp
calibrate.cpp
collect.cpp
filedata.cpp
Head.cpp
Leg.cpp
quaternion.cpp
rawdatafile.cpp
serial.cpp
shapeAPI.cpp
StdAfx.cpp
StdAfx.h
tapeAPI.cpp

)

ADD_LIBRARY(ShapeApi ${ShapeTape_SRCS})
#TARGET_LINK_LIBRARIES( vtkShapeApi )
#INSTALL_FILES(/include/vtk .h ${ShapeTape_SRCS})

