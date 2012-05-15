#ifndef ULTERIUS_H
#define ULTERIUS_H

#include "ulterius_def.h"
#include <vector>

#ifndef EXPORT_ULTERIUS_DLL
//    #define EXPORT_ULTERIUS_DLL UMC_PLATFORM_IMPORT
    #define EXPORT_ULTERIUS_DLL __declspec(dllexport)
#endif

////////////////////////////////////////////////////////////////////////////////
/// ulterius is an API that allows connection to the SONIX RP software for data
/// collection and high-level system programming.
///
/// Features:
/// - Network interface
/// - Start / Stop imaging functionality
/// - Imaging mode and preset selection
/// - Transducer selection
/// - Data collection
/// - Parameter value retreival
/// - Parameter value modification
/// - Image injection to SONIX RP software
////////////////////////////////////////////////////////////////////////////////
class EXPORT_ULTERIUS_DLL ulterius
{
public:
    ulterius();
    ~ulterius();

    bool connect(char * addr);
    bool disconnect();
    bool isConnected() const;

    void setCallback(ULTERIUS_CALLBACK fn);
    void setParamCallback(ULTERIUS_PARAM_CALLBACK fn);
    void setTimeout(int timeOut);
    bool getLastError(char * err, int sz);
    void setMessaging(bool status);

    bool getProbes(char * probes, int sz);
    bool getPresets(char * presets, int sz);
    bool getPatientInfo(char * ptinfo, int sz);
    bool getActiveProbe(char * probe, int sz);
    bool getActivePreset(char * preset, int sz);
    int getActiveImagingMode();
    int getFreezeState();
    bool selectMode(int mode);
    bool selectProbe(int connector);
    bool selectPreset(char* preset);
    bool toggleFreeze();
    bool saveScreenImage();

    bool getParamsList(std::vector<uParam> &vars);
    bool getParamValue(const char * id, int &value);
    bool getParamValue(const char * id, uRect &value);
    bool getParamValue(const char * id, uCurve &value);
    bool getParamValue(const char * id, uTGC &value);
    bool setParamValue(const char * id, int value);
    bool setParamValue(const char * id, uRect value);
    bool setParamValue(const char * id, uCurve value);
    bool setParamValue(const char * id, uTGC value);
    bool incParam(const char * id);
    bool decParam(const char * id);

    bool setDataToAcquire(int dataMask);
    int getDataToAcquire();
    bool isDataAvailable(uData type);
    bool getDataDescriptor(uData type, uDataDesc& desc);
    int getCineDataCount(uData type);
    int getMaxCineFrames(uData type);
    bool getCineData(uData type, int frame, bool useCallback = true, void * data = 0, int szData = 0);
    bool setCompressionStatus(int status);
    int getCompressionStatus();
    bool setSharedMemoryStatus(int status);
    int getSharedMemoryStatus();

    bool setInjectMode(bool enabled);
    bool getInjectMode();
    bool injectImage(void * data, int w, int h, int ss, bool scanConvert);

    bool streamScreen();
    bool stopStream();
    bool getStreamStatus();

    bool setFusionModeOff();
    bool addFusionMRTarget(int x, int y);
    bool clearAllFusionMRTargets();
    bool removeLastFusionMRTarget();
    bool startFusionMRTarget();
};

#endif
