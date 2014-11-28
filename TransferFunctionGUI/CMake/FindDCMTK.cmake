#
# try to find DCMTK libraries
#

# DCMTK_INCLUDE_DIR   - Directories to include to use DCMTK
# DCMTK_LIBRARIES     - Files to link against to use DCMTK
# DCMTK_FOUND         - If false, don't try to use DCMTK
# DCMTK_DIR           - (optional) Source directory for DCMTK
#
# DCMTK_DIR can be used to make it simpler to find the various include
# directories and compiled libraries if you've just compiled it in the
# source tree. Just set it to the root of the tree where you extracted
# the source.
#
# Written for VXL by Amitha Perera.
# 

#
# Modified 07-12-05 to suit vtkAtamaiIO by Glen Lehmann 
#

#
# Modified 06-07-06 to support dcmtk-3.5.4 includes
#

SET(DCMTK_DEBUG FALSE)#TRUE)

FIND_PATH( DCMTK_config_INCLUDE_DIR dcmtk/config/osconfig.h
  ${DCMTK_DIR}/config/include
  ${DCMTK_DIR}/include 
)

FIND_PATH( DCMTK_ofstd_INCLUDE_DIR dcmtk/ofstd/ofstdinc.h
  ${DCMTK_DIR}/ofstd/include
  ${DCMTK_DIR}/include
)

FIND_LIBRARY( DCMTK_ofstd_LIBRARY ofstd
  ${DCMTK_DIR}/ofstd/libsrc
  ${DCMTK_DIR}/ofstd/libsrc/Release
  ${DCMTK_DIR}/ofstd/libsrc/Debug
  ${DCMTK_DIR}/ofstd/Release
  ${DCMTK_DIR}/ofstd/Debug
  ${DCMTK_DIR}/lib
)

FIND_PATH( DCMTK_dcmdata_INCLUDE_DIR dcmtk/dcmdata/dctypes.h
  ${DCMTK_DIR}/dcmdata/include
  ${DCMTK_DIR}/include
)

FIND_LIBRARY( DCMTK_dcmdata_LIBRARY dcmdata
  ${DCMTK_DIR}/dcmdata/libsrc
  ${DCMTK_DIR}/dcmdata/libsrc/Release
  ${DCMTK_DIR}/dcmdata/libsrc/Debug
  ${DCMTK_DIR}/dcmdata/Release
  ${DCMTK_DIR}/dcmdata/Debug
  ${DCMTK_DIR}/lib
)

FIND_PATH( DCMTK_dcmimgle_INCLUDE_DIR dcmtk/dcmimgle/dcmimage.h
  ${DCMTK_DIR}/dcmimgle/include
  ${DCMTK_DIR}/include
)

FIND_LIBRARY( DCMTK_dcmimgle_LIBRARY dcmimgle
  ${DCMTK_DIR}/dcmimgle/libsrc
  ${DCMTK_DIR}/dcmimgle/libsrc/Release
  ${DCMTK_DIR}/dcmimgle/libsrc/Debug
  ${DCMTK_DIR}/dcmimgle/Release
  ${DCMTK_DIR}/dcmimgle/Debug
  ${DCMTK_DIR}/lib
)

#FIND_LIBRARY(DCMTK_dcmnet_LIBRARY dcmnet 
#${DCMTK_DIR}/dcmnet/libsrc/Release
#${DCMTK_DIR}/dcmnet/libsrc/Debug
#${DCMTK_DIR}/dcmnet/libsrc/
#)

FIND_PATH( DCMTK_dcmimage_INCLUDE_DIR dcmtk/dcmimage/diregist.h
  ${DCMTK_DIR}/dcmimage/include
  ${DCMTK_DIR}/include
)

FIND_LIBRARY( DCMTK_dcmimage_LIBRARY dcmimage
  ${DCMTK_DIR}/dcmimage/libsrc
  ${DCMTK_DIR}/dcmimage/libsrc/Release
  ${DCMTK_DIR}/dcmimage/libsrc/Debug
  ${DCMTK_DIR}/dcmimage/Release
  ${DCMTK_DIR}/dcmimage/Debug
  ${DCMTK_DIR}/lib
)

FIND_PATH( DCMTK_dcmjpeg_INCLUDE_DIR dcmtk/dcmjpeg/djdecode.h
  ${DCMTK_DIR}/dcmjpeg/include
  ${DCMTK_DIR}/include
)

FIND_LIBRARY( DCMTK_dcmjpeg_LIBRARY dcmjpeg 
  ${DCMTK_DIR}/dcmjpeg/libsrc
  ${DCMTK_DIR}/dcmjpeg/libsrc/Release
  ${DCMTK_DIR}/dcmjpeg/libsrc/Debug
  ${DCMTK_DIR}/dcmjpeg/Release
  ${DCMTK_DIR}/dcmjpeg/Debug
  ${DCMTK_DIR}/lib
)

FIND_LIBRARY( DCMTK_ijg8_LIBRARY ijg8
  ${DCMTK_DIR}/dcmjpeg/libijg8
  ${DCMTK_DIR}/dcmjpeg/libijg8/Release
  ${DCMTK_DIR}/dcmjpeg/libijg8/Debug
  ${DCMTK_DIR}/dcmjpeg/Release
  ${DCMTK_DIR}/dcmjpeg/Debug
  ${DCMTK_DIR}/lib
)

FIND_LIBRARY( DCMTK_ijg12_LIBRARY ijg12
  ${DCMTK_DIR}/dcmjpeg/libijg12
  ${DCMTK_DIR}/dcmjpeg/libijg12/Release
  ${DCMTK_DIR}/dcmjpeg/libijg12/Debug
  ${DCMTK_DIR}/dcmjpeg/Release
  ${DCMTK_DIR}/dcmjpeg/Debug
  ${DCMTK_DIR}/lib
)

FIND_LIBRARY( DCMTK_ijg16_LIBRARY ijg16
  ${DCMTK_DIR}/dcmjpeg/libijg16
  ${DCMTK_DIR}/dcmjpeg/libijg16/Release
  ${DCMTK_DIR}/dcmjpeg/libijg16/Debug
  ${DCMTK_DIR}/dcmjpeg/Release
  ${DCMTK_DIR}/dcmjpeg/Debug
  ${DCMTK_DIR}/lib
)

# Include only
FIND_PATH( DCMTK_dcmpstat_INCLUDE_DIR dcmtk/dcmpstat/dcmpstat.h
  ${DCMTK_DIR}/dcmpstat/include
  ${DCMTK_DIR}/include
)

IF(DCMTK_DEBUG)
  MESSAGE(STATUS "DCMTK_config_INCLUDE_DIR ${DCMTK_config_INCLUDE_DIR}")
  MESSAGE(STATUS "DCMTK_ofstd_INCLUDE_DIR ${DCMTK_ofstd_INCLUDE_DIR}")
  MESSAGE(STATUS "DCMTK_ofstd_LIBRARY ${DCMTK_ofstd_LIBRARY}")
  MESSAGE(STATUS "DCMTK_dcmdata_INCLUDE_DIR ${DCMTK_dcmdata_INCLUDE_DIR}")
  MESSAGE(STATUS "DCMTK_dcmdata_LIBRARY ${DCMTK_dcmdata_LIBRARY}")
  MESSAGE(STATUS "DCMTK_dcmimgle_INCLUDE_DIR ${DCMTK_dcmimgle_INCLUDE_DIR}")
  MESSAGE(STATUS "DCMTK_dcmimgle_LIBRARY ${DCMTK_dcmimgle_LIBRARY}")
  MESSAGE(STATUS "DCMTK_dcmimage_INCLUDE_DIR ${DCMTK_dcmimage_INCLUDE_DIR}")
  MESSAGE(STATUS "DCMTK_dcmimage_LIBRARY ${DCMTK_dcmimage_LIBRARY}")
  MESSAGE(STATUS "DCMTK_dcmjpeg_INCLUDE_DIR ${DCMTK_dcmjpeg_INCLUDE_DIR}")
  MESSAGE(STATUS "DCMTK_dcmjpeg_LIBRARY ${DCMTK_dcmjpeg_LIBRARY}")
  MESSAGE(STATUS "DCMTK_ijg8_LIBRARY ${DCMTK_ijg8_LIBRARY}")
  MESSAGE(STATUS "DCMTK_ijg12_LIBRARY ${DCMTK_ijg12_LIBRARY}")
  MESSAGE(STATUS "DCMTK_ijg16_LIBRARY ${DCMTK_ijg16_LIBRARY}")
  MESSAGE(STATUS "DCMTK_dcmpstat_INCLUDE_DIR ${DCMTK_dcmpstat_INCLUDE_DIR}")
ENDIF(DCMTK_DEBUG)


# vtkAtamaiIO
# Required Libraries: dcmimage, dcmjpeg, ijg8/12/16, dcmimgle, dcmdata, ofstd 
# Additional Required Includes: dcmpstat
IF( DCMTK_config_INCLUDE_DIR )
IF( DCMTK_ofstd_INCLUDE_DIR )
IF( DCMTK_ofstd_LIBRARY )
IF( DCMTK_dcmdata_INCLUDE_DIR )
IF( DCMTK_dcmdata_LIBRARY )
IF( DCMTK_dcmimgle_INCLUDE_DIR )
IF( DCMTK_dcmimgle_LIBRARY )
IF( DCMTK_dcmimage_INCLUDE_DIR )
IF( DCMTK_dcmimage_LIBRARY )
IF( DCMTK_dcmjpeg_INCLUDE_DIR )
IF( DCMTK_dcmjpeg_LIBRARY )
IF( DCMTK_ijg8_LIBRARY )
IF( DCMTK_ijg12_LIBRARY )
IF( DCMTK_ijg16_LIBRARY )
IF( DCMTK_dcmpstat_INCLUDE_DIR)

  SET( DCMTK_FOUND "YES" )

  IF(DCMTK_DEBUG)
    MESSAGE ( STATUS "VTKDCMTK IS FOUND" )
  ENDIF(DCMTK_DEBUG)

  SET( DCMTK_INCLUDE_DIR
    ${DCMTK_config_INCLUDE_DIR}
    ${DCMTK_ofstd_INCLUDE_DIR}
    ${DCMTK_dcmdata_INCLUDE_DIR}
    ${DCMTK_dcmimgle_INCLUDE_DIR}
    ${DCMTK_dcmimage_INCLUDE_DIR}
    ${DCMTK_dcmjpeg_INCLUDE_DIR}
    ${DCMTK_dcmpstat_INCLUDE_DIR}
  )

  SET( DCMTK_LIBRARIES
    ${DCMTK_dcmimage_LIBRARY}
    ${DCMTK_dcmjpeg_LIBRARY}
    ${DCMTK_ijg16_LIBRARY}
    ${DCMTK_ijg12_LIBRARY}
    ${DCMTK_ijg8_LIBRARY}
    ${DCMTK_dcmimgle_LIBRARY}
    ${DCMTK_dcmdata_LIBRARY}
    ${DCMTK_ofstd_LIBRARY}
  )

  #IF(DCMTK_dcmnet_LIBRARY)
  # SET( DCMTK_LIBRARIES
  # ${DCMTK_LIBRARIES}
  # ${DCMTK_dcmnet_LIBRARY}
  # )
  #ENDIF(DCMTK_dcmnet_LIBRARY)

  IF( WIN32 )
    SET( DCMTK_LIBRARIES ${DCMTK_LIBRARIES} ws2_32 netapi32 wsock32)
  ENDIF( WIN32 )

ENDIF( DCMTK_dcmpstat_INCLUDE_DIR)
ENDIF( DCMTK_ijg16_LIBRARY )
ENDIF( DCMTK_ijg12_LIBRARY )
ENDIF( DCMTK_ijg8_LIBRARY )
ENDIF( DCMTK_dcmjpeg_LIBRARY )
ENDIF( DCMTK_dcmjpeg_INCLUDE_DIR )
ENDIF( DCMTK_dcmimage_LIBRARY )
ENDIF( DCMTK_dcmimage_INCLUDE_DIR )
ENDIF( DCMTK_dcmimgle_LIBRARY )
ENDIF( DCMTK_dcmimgle_INCLUDE_DIR )
ENDIF( DCMTK_dcmdata_LIBRARY )
ENDIF( DCMTK_dcmdata_INCLUDE_DIR )
ENDIF( DCMTK_ofstd_LIBRARY )
ENDIF( DCMTK_ofstd_INCLUDE_DIR )
ENDIF( DCMTK_config_INCLUDE_DIR )

IF( NOT DCMTK_FOUND )
  SET( DCMTK_DIR "" CACHE PATH "Root of DCMTK source tree (optional)." )
  MARK_AS_ADVANCED( DCMTK_DIR )
ENDIF( NOT DCMTK_FOUND )
