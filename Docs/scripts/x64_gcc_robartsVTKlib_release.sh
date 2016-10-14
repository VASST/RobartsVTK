mkdir build
cd build
cmake -G "Unix Makefiles" -DVTK_DIR:PATH=/home/builduser/devel/RobartsVTKSuperBuild-bin/Deps/vtk-bin -DITK_DIR:PATH=/home/builduser/devel/RobartsVTKSuperBuild-bin/Deps/itk-bin -DPlusLib_DIR:PATH=/home/builduser/devel/RobartsVTKSuperBuild-bin/Deps/Plus-bin/PlusLib-bin/src -DQt5_DIR:PATH=/usr/lib/x86_64-linux-gnu/cmake/Qt5 -DRobartsVTK_USE_QT:BOOL=ON -DRobartsVTK_USE_ITK:BOOL=ON -DRobartsVTK_USE_PLUS:BOOL=ON -DRobartsVTK_USE_REGISTRATION:BOOL=ON -DRobartsVTK_USE_COMMON:BOOL=ON -DRobartsVTK_USE_CUDA:BOOL=ON -DRobartsVTK_USE_CUDA_VISUALIZATION:BOOL=ON -DRobartsVTK_USE_CUDA_ANALYTICS:BOOL=ON -DRobartsVTK_BUILD_APPS:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DBUILD_TESTING:BOOL=OFF -DBUILD_DOCUMENTATION:BOOL=OFF -DCMAKE_CXX_FLAGS:STRING=" -std=c++11" ..
make -j 24
