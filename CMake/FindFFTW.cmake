# - Find fftw
# Find the native fftw includes and library
#
#  fftw_INCLUDE_DIRS    - Where to find fftw3.h
#  fftw_LIBRARIES       - List of libraries when using fftw.
#  fftw_FOUND           - True if fftw found.

FIND_PATH(fftw_INCLUDE_DIRS fftw3.h PATHS ${fftw_ROOT_DIR})

FIND_LIBRARY(fftw3-3_LIB NAMES libfftw3-3 PATHS ${fftw_ROOT_DIR})
FIND_LIBRARY(fftw3f-3_LIB NAMES libfftw3f-3 PATHS ${fftw_ROOT_DIR})
FIND_LIBRARY(fftw3l-3_LIB NAMES libfftw3l-3 PATHS ${fftw_ROOT_DIR})

SET(fftw_LIBRARIES ${fftw3-3_LIB} ${fftw3f-3_LIB} ${fftw3l-3_LIB})

# handle the QUIETLY and REQUIRED arguments and set fftw_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (fftw DEFAULT_MSG fftw_LIBRARIES fftw_INCLUDE_DIRS)

IF(fftw3-3_LIB)
  ADD_LIBRARY(fftw SHARED IMPORTED)
  GET_FILENAME_COMPONENT(fftw_LIB_FOLDER ${fftw3-3_LIB} DIRECTORY)
  SET_PROPERTY(TARGET fftw PROPERTY IMPORTED_IMPLIB ${fftw_LIB_FOLDER}/fftw3-3${CMAKE_STATIC_LIBRARY_SUFFIX})
  SET_PROPERTY(TARGET fftw PROPERTY IMPORTED_LOCATION ${fftw_LIB_FOLDER}/fftw3-3${CMAKE_SHARED_LIBRARY_SUFFIX})
ENDIF()
IF(fftw3f-3_LIB)
  ADD_LIBRARY(fftwf SHARED IMPORTED)
  GET_FILENAME_COMPONENT(fftw_LIB_FOLDER ${fftw3f-3_LIB} DIRECTORY)
  SET_PROPERTY(TARGET fftw PROPERTY IMPORTED_IMPLIB ${fftw_LIB_FOLDER}/fftw3f-3${CMAKE_STATIC_LIBRARY_SUFFIX})
  SET_PROPERTY(TARGET fftw PROPERTY IMPORTED_LOCATION ${fftw_LIB_FOLDER}/fftw3f-3${CMAKE_SHARED_LIBRARY_SUFFIX})
ENDIF()
IF(fftw3l-3_LIB)
  ADD_LIBRARY(fftwl SHARED IMPORTED)
  GET_FILENAME_COMPONENT(fftw_LIB_FOLDER ${fftw3l-3_LIB} DIRECTORY)
  SET_PROPERTY(TARGET fftw PROPERTY IMPORTED_IMPLIB ${fftw_LIB_FOLDER}/fftw3l-3${CMAKE_STATIC_LIBRARY_SUFFIX})
  SET_PROPERTY(TARGET fftw PROPERTY IMPORTED_LOCATION ${fftw_LIB_FOLDER}/fftw3l-3${CMAKE_SHARED_LIBRARY_SUFFIX})
ENDIF()

mark_as_advanced (fftw_LIBRARIES fftw_INCLUDE_DIRS fftw3-3_LIB fftw3f-3_LIB fftw3l_LIB)