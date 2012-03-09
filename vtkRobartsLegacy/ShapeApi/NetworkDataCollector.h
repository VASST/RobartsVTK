// NetworkDataCollector.h: interface for the NetworkDataCollector class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_NETWORKDATACOLLECTOR_H__4653A64F_9A8E_48B0_AFF7_D2040C7E6687__INCLUDED_)
#define AFX_NETWORKDATACOLLECTOR_H__4653A64F_9A8E_48B0_AFF7_D2040C7E6687__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "winsock.h"
#include "tapeAPI.h"

#define MAXBUFFERSIZE 32768
#define MAXTAPEBUFSIZE 117
#define MAXCOLLECTORPORTS 12
#define ICUBE_PACKETSIZE 18

//structure for describing what type of device is connected to a datalogger serial port
struct SERIAL_DEVICE {
	unsigned short ser_num[MAXCOLLECTORPORTS];
	BYTE num_LEDs[MAXCOLLECTORPORTS];
	int nType[MAXCOLLECTORPORTS];
};

struct TAPEDATAINFO {
	unsigned char dataptr[MAXTAPEBUFSIZE];
	int bufferlength;
	int sernum;
};

struct THREEDMG_DATA {
	double dRoll;
	double dPitch;
	double dYaw;
	double accX; 
	double accY;
	double accZ;
};

//device types
#define MEAS_SHAPETAPE 1
#define MICRO_3DMG 2


class SHAPEAPI_API NetworkDataCollector  
{
public:
	int GetOrientation(int deviceindex, THREEDMG_DATA *pOrientation);
	int ParseNextSample(TAPEDATAINFO *tapedata, int numtapes, DWORD &dwTimestamp);
	int Startup(int desired_freq=0);
	int ReceiveSampleBytes(unsigned char buffer[MAXBUFFERSIZE], int &buffersize);
	int Shutdown();
	int GetPortInfo(SERIAL_DEVICE *pDevice);
	int ConnectToDataCollector(char *dataIP, unsigned short dataPort);
	NetworkDataCollector();
	virtual ~NetworkDataCollector();

private:
	int TestCheckSum(unsigned char *tapedata, int buffersize, int sernum);
	int SetOrientation(int bufindex, int deviceindex, int bufsize);
	int SaveTimeStamp(int index, int numtapes);
	int SavePortData(int bufindex, int portnum);
	int m_nSamplesize;
	int m_nInbuffersize;
	unsigned char m_inbuffer[MAXBUFFERSIZE];
	int Send3dmgStartupSequence();
	bool m_bConnected;
	bool m_bStreaming; //boolean flag to indicate whether or not data collector is presently streaming data
	SOCKET m_socket;
	SERIAL_DEVICE *m_pDevice;
	BYTE m_ipaddr_bytes[4];
	TAPEDATAINFO m_data[MAXCOLLECTORPORTS];
};

#endif // !defined(AFX_NETWORKDATACOLLECTOR_H__4653A64F_9A8E_48B0_AFF7_D2040C7E6687__INCLUDED_)
