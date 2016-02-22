# Ubuntu 15.10

## Dependencies
* To install the Qt5 development environment, do the following:
   ```
   > sudo apt-get install qt5-default qttools5-dev libqt5webkit5-dev
   ```

## Issues
I've run into a number of issue building on Ubuntu 15.10. Here are some of the fixes I've had to apply:

* After installing CUDA, you may get the error 
    ```
    #error -- unsupported GNU version! gcc versions later than 4.9 are not supported
    ```
    To fix this, simply comment out that line in ```/usr/local/cuda/include/host_config.h```