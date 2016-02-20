# Ubuntu 15.10

I've run into a number of issue building on Ubuntu 15.10. Here are some of the fixes I've had to apply:

* After installing CUDA, you may get the error 
    ```
    #error -- unsupported GNU version! gcc versions later than 4.9 are not supported
    ```
    To fix this, simply comment out that line in ```/usr/local/cuda/include/host_config.h```