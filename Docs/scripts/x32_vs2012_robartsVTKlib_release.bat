mkdir build
cd build
cmake -G "Visual Studio 11" -DVTK_DIR:PATH=F:\arankin\RVTK-bin-x32\Deps\vtk-bin -DITK_DIR:PATH=F:\arankin\RVTK-bin-x32\Deps\itk-bin -DPlusLib_DIR:PATH=F:\arankin\RVTK-bin-x32\Deps\Plus-bin\PlusLib-bin\src -DQt5_DIR:PATH=F:\Qt\qt-5.5.0-x86-msvc2012-rev0\qt-5.5.0-x86-msvc2012-rev0\lib\cmake\Qt5 -DRobartsVTK_USE_QT:BOOL=ON -DRobartsVTK_USE_ITK:BOOL=ON -DRobartsVTK_USE_PLUS:BOOL=ON -DRobartsVTK_USE_REGISTRATION:BOOL=ON -DRobartsVTK_USE_COMMON:BOOL=ON -DRobartsVTK_USE_CUDA:BOOL=ON -DRobartsVTK_USE_CUDA_VISUALIZATION:BOOL=ON -DRobartsVTK_USE_CUDA_ANALYTICS:BOOL=ON -DRobartsVTK_BUILD_EXAMPLES:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=ON -DBUILD_TESTING:BOOL=OFF -DBUILD_DOCUMENTATION:BOOL=OFF -DPYTHON_INCLUDE_DIR:PATH=C:/Python27/include -DRobartsVTK_WRAP_PYTHON:BOOL=ON -DPYTHON_LIBRARY:FILEPATH=C:/Python27/libs/python27.lib -DPYTHON_EXECUTABLE:FILEPATH=C:/Python27/python ..
"C:\Program Files (x86)\Microsoft Visual Studio 11.0\Common7\IDE\devenv.com" ./RobartsVTK.sln /Project ALL_BUILD /Build Release