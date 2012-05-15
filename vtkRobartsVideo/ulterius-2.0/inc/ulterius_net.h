#ifndef ULTERIUS_NET_H
#define ULTERIUS_NET_H

#include "ulterius_def.h"

// network ports
#define ULTERIUS_COMM_PORT       (525)
#define ULTERIUS_DATA_PORT       (526)
// header versions
#define ULTERIUS_COMMHDR_VERSION (0x00000001)
#define ULTERIUS_DATAHDR_VERSION (0x00000001)

#define ULTERIUS_SZ_HDRDATA      (80)

#define ULTERIUS_MEM_NAME        _T("ULTERIUSSHMEM")

////////////////////////////////////////////////////////////////////////////////
/// Network communication functions.
////////////////////////////////////////////////////////////////////////////////
enum uCommFunction
{
    // Used only for event purposes
    ufConnectComm = 0,
    ufConnectData = 1,

    // Actions API
    ufGetProbes = 1,
    ufGetPresets = 2,
    ufGetFreezeState,
    ufGetActiveImagingMode,
    ufGetActiveProbe,
    ufGetActivePreset,
    ufSelectMode,
    ufSelectProbe,
    ufSelectPreset,
    ufToggleFreeze,
    ufSaveScreenImage,

    // Params API
    ufGetParamsList,
    ufGetParamValue,
    ufSetParamValue,
    ufIncParam,
    ufDecParam,

    // Data API
    ufSetDataToAcquire,
    ufGetDataToAcquire,
    ufIsDataAvailable,
    ufGetDataDesc,
    ufGetCineDataCount,
    ufGetMaxCineFrames,
    ufGetCineData,
    ufSetCompression,
    ufGetCompression,
    ufSetSharedMemory,
    ufGetSharedMemory,

    // Inject API
    ufSetInjectMode,
    ufGetInjectMode,
    ufInjectImage,

    // Streaming
    ufStreamScreen,
    ufStopStream,
    ufGetStreamStatus,

    ufGetPatientInfo,

    // Parameter Status
    ufGetParamStatus,

    // Fusion Controls
    ufSetFusionModeOff,
    ufAddFusionMRTarget,
    ufFusionStartMRTarget,
    ufClearAllFusionMRTargets,
    ufFusionRemoveLastMRTarget
};

////////////////////////////////////////////////////////////////////////////////
/// Communications header.
////////////////////////////////////////////////////////////////////////////////
class uCommHdr
{
public:
    int version;
    uCommFunction function;
    char charprm[ULTERIUS_SZ_HDRDATA];
    int prm1;
    int prm2;
    int prm3;
    int prm4;
    char data[ULTERIUS_SZ_HDRDATA];
};

////////////////////////////////////////////////////////////////////////////////
/// Data header.
////////////////////////////////////////////////////////////////////////////////
class uDataHdr
{
public:
    int version;
    uData type;
    int size;
    bool cine;
    bool compressed;
    int frmnum;
};

#endif
