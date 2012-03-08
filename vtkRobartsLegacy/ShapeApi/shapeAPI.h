
// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the SHAPEAPI_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// SHAPEAPI_API functions as being imported from a DLL, wheras this DLL sees symbols
// defined with this macro as being exported.
#ifdef SHAPEAPI_EXPORTS
#define SHAPEAPI_API __declspec(dllexport)
#else
//#define SHAPEAPI_API __declspec(dllimport) // CAUSES PROBLEMS WITH VKT/PYTHON
#define SHAPEAPI_API __declspec(dllexport)
#endif


