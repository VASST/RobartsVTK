<img src="Docs/readme/images/robartsvtk.png" alt="RobartsVTK" height="125px"/>

RobartsVTK is a software package for image processing and visualization.

# Download RobartsVTK

Using a [git](https://en.wikipedia.org/wiki/Git_(software)) client, clone the repo link above.
* Windows? Try [TortoiseGit](https://tortoisegit.org/download/)
* Ubuntu? Try [RabbitVCS](http://rabbitvcs.org/), [SmartGit](http://www.syntevo.com/smartgit/) or [git-cola](http://git-cola.github.io/downloads.html)
* Mac? Try [GitHub](https://desktop.github.com/)

# Building RobartsVTK
For convenient building, please see the RobartsVTK [super build](http://git.imaging.robarts.ca/vasst/RobartsVTKBuild) project.

## Known Configurations
RobartsVTK has been built on the following configurations:
* Windows 10 x64, Visual Studio 2012, 32/64bit builds (see [Windows build tips](Docs/readme/windows.md))
* Ubuntu 15.10, Unix Makefiles/Eclipse CDT (see [Ubuntu build tips](Docs/readme/ubuntu.md))
* Ubuntu 15.04, Unix Makefiles/Eclipse CDT

## Dependencies
Dependencies must be built or installed before RobartsVTK can be built. Please visit the respective links to download the appropriate packages.
* [CMake 3.4](https://cmake.org/download/) - installed
* [VTK 6.3](http://www.vtk.org/download/) - built
   * If Python wrapping of RobartsVTK is desired, VTK must be built with WRAP_PYTHON enabled.
* [QT 4/5](http://download.qt.io/archive/qt/) - built (optional, please follow Qt build instructions)
* [ITK 4.7.2](http://www.itk.org/ITK/resources/software.html) - built (optional)
* [CUDA 7](https://developer.nvidia.com/cuda-downloads) - installed (optional)
* [PLUS 2.3](http://plustoolkit.org) - built (optional)
* [Python 2.7 x64](https://www.python.org/downloads/release/python-2711/) - installed (optional)

## CMake Configuration
The following variables should be set when configuring RobartsVTK
* RobartsVTK_BUILD_APPS:BOOL = `ON`
* RobartsVTK_Include_Outdated_Registration:BOOL = `OFF`
* ITK_DIR:PATH = `<path/to/your/itk-bin/dir>`
* PlusLib_DIR:PATH = `<path/to/your/plus-bin/dir>`
* QT4 - QT_QMAKE_EXECUTABLE:FILEPATH = `<path/to/your/qt-build>/bin/qmake.exe`
* QT5 - as above OR - Qt5_DIR:PATH = `<path/to/your/qt-build>/lib/cmake/Qt5`
* VTK_DIR:PATH = `<path/to/your/vtk-bin/dir>`
    * If you're wrapping with python:
        * PYTHON_INCLUDE_DIR:PATH = `<path/to/python-install>/include`
        * PYTHON_LIBRARY:PATH = `<path/to/python-install>/libs/python27.lib`

# Continuous Integration
Continuous integration is enabled for this project and has workers running on happy.imaging.robarts.ca (Ubuntu 15.04). Migration to a permanent build system is on-going.

## Set up a runner
* Follow installation instructions for the [gitlab-ci-multi-runner](https://gitlab.com/gitlab-org/gitlab-ci-multi-runner). 
    * During the installation, make the following choices (unless you are an expert):
        * coordinator URL: http://git.imaging.robarts.ca/ci
        * gitlab-ci token: SEE Settings->Runners (in left menu)
        * description: please provide a descriptive name of your configuration (institution, OS, architecture)
        * tags:
            * Windows x64: windows, cmake, x64, visual-studio-11
            * Windows x32: windows, cmake, x32, visual-studio-11
            * Ubuntu: cmake, gcc
        * executor: shell
    * Edit the config.toml file that was created in the same directory
        * After the line
        
        `executor = "shell"`
        
        add
        
        `shell = "cmd"` for Windows or 
        
        `shell = "bash"` for Ubuntu (typically /etc/gitlab-runner/config.toml)
    * Copy the appropriate script file to any location specified in your PATH environment variable
        * Win32: [x32_vs2012_robartsVTKlib_release.bat](Docs/scripts/x32_vs2012_robartsVTKlib_release.bat)
        * Win64: [x64_vs2012_robartsVTKlib_release.bat](Docs/scripts/x64_vs2012_robartsVTKlib_release.bat)
        * Win32: [x64_gcc_robartsVTKlib_release.sh](Docs/scripts/x64_gcc_robartsVTKlib_release.sh)
    * Install and start the runner (instructions provided by gitlab runner installation script)

## Add a new type of runner
To add a new build type, An entry must be made to [.gitlab-ci.yml](../../.gitlab-ci.yml). Please duplicate the existing structure of an entry but customize the build script and tag entries.

# License
Please see the [license](LICENSE.md) file.

# Acknowledgments
The Robarts Research Institute VASST Lab would like to thank the creator and maintainers of [GitLab](https://about.gitlab.com/).