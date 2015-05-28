// NetworkDataCollector.cpp: implementation of the NetworkDataCollector class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "NetworkDataCollector.h"
#include "3dmath.h"
#include <fstream>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

/*
Name:    NetworkDataCollector 
Purpose:  Constructor 
Accepts:  none
Returns:    void
*/
NetworkDataCollector::NetworkDataCollector()
{
  WORD wRequestVersion = 2; //request version 2.0
  WSADATA wsData;
  WSAStartup(wRequestVersion,&wsData);
  m_socket = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);

  //set options for socket
  int delay_val=1;
  int rxlowat=1;
  int rxbufsize=16384;
  int sendat=2;
  int nTest=0;
  int nTimeout=1000; //1000 ms
  nTest=setsockopt(m_socket,IPPROTO_TCP,TCP_NODELAY,(char *)&delay_val,sizeof(int));
  if (!nTest) {
    nTest=WSAGetLastError();
  }
  nTest=setsockopt(m_socket,SOL_SOCKET,SO_RCVLOWAT,(char *)&rxlowat,sizeof(int));
  if (!nTest) {
    nTest=WSAGetLastError();
  }
  nTest=setsockopt(m_socket,SOL_SOCKET,SO_RCVBUF,(char *)&rxbufsize,sizeof(int));
  if (!nTest) {
    nTest=WSAGetLastError();
  }
  nTest=setsockopt(m_socket,SOL_SOCKET,SO_SNDLOWAT,(char *)&sendat,sizeof(int));
  if (!nTest) {
    nTest=WSAGetLastError();
  }
  nTest=setsockopt(m_socket,SOL_SOCKET,SO_RCVTIMEO,(char *)&nTimeout,sizeof(int));
  if (!nTest) {
    nTest=WSAGetLastError();
  }
  nTest=setsockopt(m_socket,SOL_SOCKET,SO_SNDTIMEO,(char *)&nTimeout,sizeof(int));
  if (!nTest) {
    nTest=WSAGetLastError();
  }

  m_bStreaming = false;
  m_bConnected = false;
  m_pDevice = NULL;
  for (int i=0;i<4;i++) m_ipaddr_bytes[i]=0;
  m_nInbuffersize=0;
  for (i=0;i<MAXBUFFERSIZE;i++) m_inbuffer[i]=0;
  m_nSamplesize=0;
  //initialize m_data structures
  for (i=0;i<MAXCOLLECTORPORTS;i++) {
    m_data[i].bufferlength=0;
    m_data[i].sernum=0;
    for (int j=0;j<MAXTAPEBUFSIZE;j++) m_data[i].dataptr[j]=0;
  }
}

/*
Name:    ~NetworkDataCollector 
Purpose:  Destructor
Accepts:  none
Returns:    void
*/
NetworkDataCollector::~NetworkDataCollector()
{
  WSACleanup();  
  if (m_pDevice) delete m_pDevice;
}


/*
Name:    ConnectToDataCollector 
Purpose:  Makes a network connection to a Measurand data collector that is being used to collect
      data from one or more serial port devices (i.e. either ShapeTapes, serializers, or 
      orientation sensors).
Accepts:  dataIP = character string describing the IP address of the data collector (ex: 10.0.0.253).
      dataPort = network port number that the data collector server is using (usually 2000).
Returns:    < 0 indicates error:
        -1 = invalid IP address
        -2 = could not connect to the data collector
        -3 = unable to get a response from the data collector
      0 indicates a successful connection to the data collector
*/
int NetworkDataCollector::ConnectToDataCollector(char *dataIP, unsigned short dataPort) {
  //Note: dataPort should be 2000 for ipEngine based systems
  unsigned char ip_bytes[4];
  int temp_bytes[4];
  int nScan = sscanf(dataIP,"%d.%d.%d.%d",&temp_bytes[0],&temp_bytes[1],&temp_bytes[2],
    &temp_bytes[3]);
  if (nScan!=4) return -1; //invalid ip address
  for (int i=0;i<4;i++) ip_bytes[i]=(unsigned char)temp_bytes[i];
  
  sockaddr_in dataCollectaddr;
  dataCollectaddr.sin_family = AF_INET;
  dataCollectaddr.sin_port = htons(dataPort);
  dataCollectaddr.sin_addr.S_un.S_un_b.s_b1 = ip_bytes[0];
  dataCollectaddr.sin_addr.S_un.S_un_b.s_b2 = ip_bytes[1];
  dataCollectaddr.sin_addr.S_un.S_un_b.s_b3 = ip_bytes[2];
  dataCollectaddr.sin_addr.S_un.S_un_b.s_b4 = ip_bytes[3];
  for (i=0;i<8;i++) dataCollectaddr.sin_zero[i]=0;

  int nConnectOK = connect(m_socket,(sockaddr *)&dataCollectaddr,sizeof(sockaddr));
  if (nConnectOK!=0) {
    int nErrorcode = WSAGetLastError();  
    return -2; //error could not connect to the Data Collector
  }
  for (i=0;i<4;i++) m_ipaddr_bytes[i]=ip_bytes[i];
  m_bConnected=true;
  return 0;  
}


/*
Name:    GetPortInfo
Purpose:  Gets information about how many and what type of devices are connected to the data collector.
Accepts:  pDevice = pointer to a structure of type SERIAL_DEVICE that holds information about the 
      various serial port devices that are connected to the data collector.
Returns:    < 0 indicates error:
        -1 = invalid socket
        -2 = error sending data to data collector
        -3 = error trying to receive configuration data
      0 indicates a successful connection to the data collector
*/
int NetworkDataCollector::GetPortInfo(SERIAL_DEVICE *pDevice) {
  char send_bytes[2] = {0,0}; //byte codes to send that signify a request for 
                     //which devices are connected to the serial ports
                     //of the data collector
  if (!m_socket) return -1; //invalid socket
  if (send(m_socket,send_bytes,2,0)<2) {
    //error sending data
    int nErrorcode = WSAGetLastError();
    return -2;
  }
  //receive 25 byte response from the data collector
  int nRetcode = 0;
  bool bFoundportInfoSync = false;
  unsigned char response[25];
  while (!bFoundportInfoSync) {
    nRetcode = recv(m_socket,(char *)response,25,0);
    if (nRetcode==25) {
      //look for sync pattern 9 0xFF's, followed by 1 0x00 byte.
      for (int i=0;i<16;i++) {
        if   (response[i]==0xff&&response[i+1]==0xff&&response[i+2]==0xff&&
          response[i+3]==0xff&&response[i+4]==0xff&&response[i+5]==0xff&&
          response[i+6]==0xff&&response[i+7]==0xff&&response[i+8]==0xff&&
          response[i+9]==0x00) {
          bFoundportInfoSync=true;
          if (i!=0) { //need to get some more bytes
            unsigned char extrabytes[25];
            nRetcode = recv(m_socket,(char *)extrabytes,i,0);
            if (nRetcode!=i) return -3; //unable to get response
            //shift bytes into response buffer
            for (int j=10;j<25-i;j++) response[j]=response[j+i];
            for (j=0;j<i;j++) response[25-i+j]=extrabytes[j];
          }  
          //parse port, device info in response buffer
          for (int j=0;j<5;j++) {
            pDevice->ser_num[j] = (response[10+3*j]<<8)+response[11+3*j];
            pDevice->num_LEDs[j] = response[12+3*j];
            if (pDevice->num_LEDs[j]>0&&pDevice->num_LEDs[j]<0xff)
              pDevice->nType[j]=MEAS_SHAPETAPE;
            else if (pDevice->num_LEDs[j]==0xff) pDevice->nType[j]=MICRO_3DMG;
            else pDevice->nType[j]=0;
          }
          break;
        }
      }
    }
    else {
      //error trying to receive configuration data
      int nErrorcode = WSAGetLastError();
      return -3; //unable to get valid response
    }
  }
  //copy info into m_pdevice
  if (m_pDevice) delete m_pDevice;
  m_pDevice = new SERIAL_DEVICE;
  for (int i=0;i<MAXCOLLECTORPORTS;i++) {
    m_pDevice->nType[i]=pDevice->nType[i];
    m_pDevice->num_LEDs[i]=pDevice->num_LEDs[i];
    m_pDevice->ser_num[i]=pDevice->ser_num[i];
  }
  return 0;
}


/*
Name:    Shutdown
Purpose:  Shuts down a data collector so that it stops sending data over the network.
Accepts:  None
Returns:    < 0 indicates error:
        -1 = invalid socket
        -2 = error sending data to data collector
      0 indicates that bytes were successfully sent over the network, telling the data collector 
        to shutdown.
*/
int NetworkDataCollector::Shutdown() {
  unsigned char send_bytes[2] = {0xff,0xff}; //byte codes to send that signify the code for shutting down
  if (!m_socket) return -1; //invalid socket
  if (send(m_socket,(char *)send_bytes,2,0)<2) {
    //error sending data
    int nErrorcode = WSAGetLastError();
    return -2;
  }
  m_bConnected=false;
  return 0;
}

/*
Name:    ReceiveSampleBytes
Purpose:  Receives raw binary data from the data collector.
Accepts:  buffer = input buffer where raw binary data from the data collector is stored.
        buffersize = length of data to try looking for, as an input it is safest to set this to 
             MAXBUFFERSIZE. The value of buffersize is set by the function to be the 
             actual number of bytes read from the data collector.
Returns:    < 0 indicates error:
        -1 = invalid socket
        -2 = not connected to the data collector
        -3 = error receiving bytes from data collector
        -4 = length of data requested is too long
      0 indicates that bytes were successfully sent over the network, telling the data collector 
        to shutdown.
*/
int NetworkDataCollector::ReceiveSampleBytes(unsigned char buffer[MAXBUFFERSIZE], int &buffersize) {
  //buffer should be allocated with MAXBUFFERSIZE bytes
  //check to make sure that buffersize is not too large
  if (buffersize>MAXBUFFERSIZE) return -4;
    if (!m_socket) return -1; //invalid socket
  if (!m_bConnected) return -2; //not connected to the data collector
  if (!m_bStreaming) Startup(); //need to tell data collector to start collecting data
  buffersize = recv(m_socket,(char *)buffer,buffersize,0);
  if (buffersize<=0) {
    //error receiving bytes
    int nErrorcode = WSAGetLastError();
    return -3;
  }
  //copy received bytes over to input buffer
  int nLimit = min(buffersize,MAXBUFFERSIZE-m_nInbuffersize);
  for (int i=0;i<nLimit;i++) {
    m_inbuffer[i+m_nInbuffersize]=buffer[i];
  }
  m_nInbuffersize+=nLimit;
  return buffersize;
}


/*
Name:    Startup
Purpose:  Tells the data collector to start collecting data from any serial port devices and 
      send it out over the network.
Accepts:  desired_freq = frequency in Hz at which data is collected. If this value is 0 then data is 
               collected as quickly as possible.
Returns:    < 0 indicates error:
        -1 = invalid socket
        -2 = not connected to the data collector
        -3 = already connected
        -4 = could not send startup sequence for intializing any orientation sensors
        -5 = error sending bytes to the data collected
      0 indicates that bytes were successfully sent over the network, telling the data collector 
        to begin sending data.
*/
int NetworkDataCollector::Startup(int desired_freq/*=0*/) {
  //tells data collector to start collecting data
  //desired_freq==0 corresponds to sampling as fast as possible
  if (!m_socket) return -1; //invalid socket
  if (!m_bConnected) return -2; //need to have valid network connection first
  //initialize collection of orientation sensor data
  if (m_bStreaming) return -3; //already started, should Shutdown first establish a new connection and then startup again.
  if (Send3dmgStartupSequence()<0) return -4;
  if (!m_pDevice) {
    m_pDevice = new SERIAL_DEVICE;
    GetPortInfo(m_pDevice);
  }
  unsigned char outbytes[7];
  int numtapes=0;
  outbytes[0]=1; //corresponds to collecting 1 sample at a time
  outbytes[1]=desired_freq&0xff;
  for (int j=0;j<MAXCOLLECTORPORTS;j++) {
    if (m_pDevice->nType[j]==MEAS_SHAPETAPE) {
      outbytes[2+numtapes]=j+1;
      numtapes++;
    }
  }
  int nBytessent=send(m_socket,(char *)outbytes,2+numtapes,0);
  if (nBytessent!=2+numtapes) return -5; //error sending bytes
  m_bStreaming=true; //should now be streaming data from data collector
  return 0;
}


/*
Name:    Send3dmgStartupSequence
Purpose:  Tells the data collector to initialize any 3DM-G orientation sensors that might be
      present on any of its serial ports.
Accepts:  None
Returns:    < 0 indicates error:
        -1 = not connected to the data collector
        -2 = could not set socket option to reuse ip addresses
        -3 = error could not make 2nd connection to the data collector
        -4 = error sending bytes to the data collector
      0 indicates that bytes were successfully sent over the network, telling the data collector 
        to initialize any 3DM-G orientation sensors that might be present
*/
int NetworkDataCollector::Send3dmgStartupSequence() {
  //tells data collector to collect data from any 3dm-g orientation sensors that might 
  //be connected to its serial ports
  if (!m_bConnected) return -1; //need to have valid network connection first to know which
                    //ip address to use
  const unsigned short msgPort = 2001; //network port used for sending network messages to 
                     //data collector
  
  unsigned char send_bytes[1]; 
  send_bytes[0]=0xf7; //command to indicate that a 3dm-g is present
  SOCKET msgSock=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);

  //set socket option to reuse ip address
  int nReuse=1;
  int nTest=setsockopt(msgSock,SOL_SOCKET,SO_REUSEADDR,(char *)&nReuse,sizeof(int));
  if (nTest<0) {
    nTest=WSAGetLastError();
    return -2; //could not set socket option to reuse ip addresses
  }
  sockaddr_in dataCollectaddr;
  dataCollectaddr.sin_family = AF_INET;
  dataCollectaddr.sin_port = htons(msgPort);
  dataCollectaddr.sin_addr.S_un.S_un_b.s_b1 = m_ipaddr_bytes[0];
  dataCollectaddr.sin_addr.S_un.S_un_b.s_b2 = m_ipaddr_bytes[1];
  dataCollectaddr.sin_addr.S_un.S_un_b.s_b3 = m_ipaddr_bytes[2];
  dataCollectaddr.sin_addr.S_un.S_un_b.s_b4 = m_ipaddr_bytes[3];
  for (int i=0;i<8;i++) dataCollectaddr.sin_zero[i]=0;

  int nConnectOK = connect(msgSock,(sockaddr *)&dataCollectaddr,sizeof(sockaddr));
  if (nConnectOK!=0) {
    int nErrorcode = WSAGetLastError();  
    return -3; //error could not connect to the data collector
  }
  int nBytesSent = send(msgSock,(char *)send_bytes,1,0);
  if (nBytesSent<1) {
    int nErrorcode = WSAGetLastError();  
    return -4; //error sending bytes to the data collector
  }
  return 0;
}


/*
Name:    ParseNextSample
Purpose:  Parses the next sample of data contained within the m_inbuffer input buffer.
Accepts:  tapedata = an array of structures of type TAPEDATAINFO that hold the relevant data for 
             for each tape. The first numtapes elements of the array are filled by this 
             function.
      numtapes = the number of ShapeTapes connected to the data collector.
      dwTimestamp = The time in ms since the data collector was turned on (output of the function).
Returns:    < 0 indicates error:
        -1 = no data has been read in yet to parse or no tapes have been specified
        -2 = could not find sync pattern anywhere in the data buffer
        -3 = no data present after sync pattern
        -4 = an error occurred trying to save device data
        -5 = unexpected end of buffer

      0 indicates that the sample was parsed successfully. Sample data for each tape is 
        contained within the various elements of tapedata.
*/
int NetworkDataCollector::ParseNextSample(TAPEDATAINFO *tapedata, int numtapes, DWORD &dwTimestamp) {
  if (m_nInbuffersize<=0||numtapes<=0) return -1; //no data has been read in yet to parse or no tapes have been specified
  //find first available sync pattern in data (9 0xFF's in a row)
  int currenttape=0;
  int data_index=-1;
  for (int i=0;i<m_nInbuffersize-8;i++) {
    if ((m_inbuffer[i]&m_inbuffer[i+1]&m_inbuffer[i+2]&m_inbuffer[i+3]&m_inbuffer[i+4]&
      m_inbuffer[i+5]&m_inbuffer[i+6]&m_inbuffer[i+7]&m_inbuffer[i+8])==0xff) {
      //found sync
      data_index=i+9;
      //skip past any extra 0xff's 
      while (data_index<m_nInbuffersize&&m_inbuffer[data_index]==0xff) data_index++;
      break;
    }
  }
  if (data_index<0) return -2; //could not find sync pattern anywhere in data buffer
  if (data_index>=m_nInbuffersize) return -3; //no data present after sync pattern 
  bool bExitloop=false;
  do {
    int current_port=m_inbuffer[data_index];
    data_index++;
    if (current_port==0) {
      //general data collector info (ignore and move on to next sample)
      int newdata_index=-1;
      for (i=data_index;i<m_nInbuffersize-8;i++) {
        if ((m_inbuffer[i]&m_inbuffer[i+1]&m_inbuffer[i+2]&m_inbuffer[i+3]&m_inbuffer[i+4]&
          m_inbuffer[i+5]&m_inbuffer[i+6]&m_inbuffer[i+7]&m_inbuffer[i+8])==0xff) {
          //found new sync
          newdata_index=i+9;
          //skip past any extra 0xff's 
          while (newdata_index<m_nInbuffersize&&m_inbuffer[newdata_index]==0xff) newdata_index++;
          break;
        }
      }
      if (newdata_index<0) return -2; //could not find sync pattern anywhere in data buffer
      if (newdata_index>=m_nInbuffersize) return -3; //no data present after sync pattern 
      data_index=newdata_index;
    }
    else {
      int nbytes_saved=SavePortData(data_index,current_port);
      if (nbytes_saved<0) {
        bExitloop=true;
        if (nbytes_saved==-1) {//an error occurred trying to save device data, so flush out the input buffer
          m_nInbuffersize=0;
          return -4;
        }
        break;
      }
      else data_index+=nbytes_saved;
      currenttape++;
      if (currenttape==numtapes) {
        int nTs=SaveTimeStamp(data_index,numtapes);
        if (nTs<0) { //unexpected end of buffer
          m_nInbuffersize=0; 
          return -5;
        }
        bExitloop=true;
        data_index+=nTs;
        //check to see if any orientation data is available
        if (data_index<=m_nInbuffersize-ICUBE_PACKETSIZE) {
          if (m_inbuffer[data_index]!=0||m_inbuffer[data_index+1]!=0||
            m_inbuffer[data_index+2]!=0||m_inbuffer[data_index+3]!=0) {
            SetOrientation(data_index,currenttape,ICUBE_PACKETSIZE);
          }
          data_index+=ICUBE_PACKETSIZE;
        }
        //see if there might be some additional 3dm-g's connected to the data collector
        bool bUpcomingSync=false;
        int count=0;
        while (bUpcomingSync==false&&data_index<=m_nInbuffersize-13&&
          count<MAXCOLLECTORPORTS) {
          if (m_inbuffer[data_index]!=0xff||m_inbuffer[data_index+1]!=0xff||
            m_inbuffer[data_index+2]!=0xff||m_inbuffer[data_index+3]!=0xff) {
            count++;
            SetOrientation(data_index,currenttape+count,13);
            data_index+=13;
          }
          else bUpcomingSync=true;
        }
      }
    }
  }  while (data_index<m_nInbuffersize&&!bExitloop);
  //copy info to tapedata
  for (i=0;i<MAXCOLLECTORPORTS;i++) {
    tapedata[i].bufferlength=m_data[i].bufferlength;
    tapedata[i].sernum=m_data[i].sernum;
    for (int j=0;j<tapedata[i].bufferlength;j++) tapedata[i].dataptr[j]=m_data[i].dataptr[j];
  }
  //remove old items from buffer, and shift unused items down
  for (i=data_index;i<m_nInbuffersize;i++) m_inbuffer[i-data_index]=m_inbuffer[i];
  m_nInbuffersize-=data_index;
  return 0;
}


/*
Name:    SavePortData
Purpose:  Parses data for an individual tape.
Accepts:  bufindex = index within m_inbuffer at which to start looking at ShapeTape data.
      portnum = data collector serial port # of tape. 
Returns:    < 0 indicates error:
        -1 = serial port # is invalid
        -2 = insufficient data available
        -3 = invalid data for tape (checksum error)
      otherwise returns the number of bytes saved for the individual tape
*/
int NetworkDataCollector::SavePortData(int bufindex, int portnum) {
  int num_bytes_required=0;
  int bytes_available=m_nInbuffersize-bufindex;
  if (portnum>MAXCOLLECTORPORTS||portnum<=0) return -1; //don't save anything
  if (bytes_available<16) return -2; //insufficent data available

  m_data[portnum-1].sernum=((m_inbuffer[bufindex+1])<<8)+m_inbuffer[bufindex+2];
  
  if (m_data[portnum-1].sernum<4096) num_bytes_required = 16*(m_inbuffer[bufindex]&0x0f)+4;
  else num_bytes_required=16*(m_inbuffer[bufindex]&0x0f)+17;
  
  if (num_bytes_required>bytes_available)  return -2; //insufficient data available
  //store tape data 
  if (num_bytes_required>0&&num_bytes_required<MAXTAPEBUFSIZE) 
    m_data[portnum-1].bufferlength=num_bytes_required;
  else return -1; //error
  if (TestCheckSum(&m_inbuffer[bufindex],num_bytes_required,m_data[portnum-1].sernum)<0) 
    return -3; //checksum error
  //fill tape buffer
  for (int i=0;i<num_bytes_required;i++) m_data[portnum-1].dataptr[i]=m_inbuffer[bufindex+i];
  return num_bytes_required;
}


/*
Name:    SaveTimeStamp
Purpose:  saves the time stamp for a given sample to the appropriate locations in the m_data
      member variable.
Accepts:  index = index within m_inbuffer where 4 byte time stamp is known to be located.
      numtapes = number of serial port tapes connected to data collector.
Returns:    < 0 indicates error:
        -1 = size of m_inbuffer is too small to read in time stamp
      4 if all 4 timestamp bytes were saved successfully
*/
int NetworkDataCollector::SaveTimeStamp(int index, int numtapes) {
  if (m_nInbuffersize-index<4) return -1;//buffer too small
  for (int i=0;i<numtapes;i++) {
    if (m_data[i].sernum>0) {
      m_data[i].dataptr[m_data[i].bufferlength]=m_inbuffer[index];
      m_data[i].dataptr[m_data[i].bufferlength+1]=m_inbuffer[index+1];
      m_data[i].dataptr[m_data[i].bufferlength+2]=m_inbuffer[index+2];
      m_data[i].dataptr[m_data[i].bufferlength+3]=m_inbuffer[index+3];
      m_data[i].bufferlength+=4;
    }
  }  
  return 4;
}

/*
Name:    SetOrientation
Purpose:  saves raw orientation data bytes located in m_inbuffer to the appropriate
      m_data location.
Accepts:  bufindex = index within m_inbuffer where the raw orientation bytes are located
      deviceindex = index of m_data where the orientation info will be stored
      bufsize = size in bytes of orientation data 
Returns:    < 0 indicates error:
        -1 = size of m_inbuffer is too small to read in orientation data
      otherwise returns the number of orientation bytes successfully saved
*/
int NetworkDataCollector::SetOrientation(int bufindex, int deviceindex, int bufsize) {
  if (m_nInbuffersize-bufindex<bufsize) return -1;//buffer too small
  m_data[deviceindex].bufferlength=bufsize;
  for (int i=0;i<bufsize;i++) m_data[deviceindex].dataptr[i]=m_inbuffer[bufindex+i];
  return bufsize;
}


/*
Name:    GetOrientation
Purpose:  extracts orientation info from a given m_data location
Accepts:  deviceindex = index within m_data where orientation bytes are stored.
      pData = pointer to a THREEDMG_DATA structure that stores all of the relevant 
          orientation info.
Returns:    < 0 indicates error:
        -1 = invalid device index
        -2 = first byte is not 0x05 --> must be 0x05 for 3dm-g output
        -3 = invalid checksum
      0 if orientation is successfully returned within the pData structure
*/
int NetworkDataCollector::GetOrientation(int deviceindex, THREEDMG_DATA *pData) {
  
  short stab_q0, stab_q1, stab_q2, stab_q3; //stabilized quaternion values. //need to be
                             //divided by 8192 to get unit quaternion 
                             //components
  unsigned short timerticks; //timer ticks from 3dm-g
  unsigned short checksum; //16 bit checksum
  int computed_checksum;
  if (deviceindex<0||deviceindex>=MAXCOLLECTORPORTS) return -1; //invalid device index
  if (m_data[deviceindex].dataptr[0]!=0x05) return -2; //first byte must be 0x05 for 3dm-g output.
  //get values
  stab_q0 = (m_data[deviceindex].dataptr[1]<<8)+m_data[deviceindex].dataptr[2];
  stab_q1 = (m_data[deviceindex].dataptr[3]<<8)+m_data[deviceindex].dataptr[4];
  stab_q2 = (m_data[deviceindex].dataptr[5]<<8)+m_data[deviceindex].dataptr[6];
  stab_q3 = (m_data[deviceindex].dataptr[7]<<8)+m_data[deviceindex].dataptr[8];
  timerticks = (m_data[deviceindex].dataptr[9]<<8)+m_data[deviceindex].dataptr[10];
  checksum=(m_data[deviceindex].dataptr[11]<<8)+m_data[deviceindex].dataptr[12];
  //test checksum
  computed_checksum=0x05;
  if (stab_q0<0) computed_checksum+=65536+stab_q0;
  else computed_checksum+=stab_q0;
  if (stab_q1<0) computed_checksum+=65536+stab_q1;
  else computed_checksum+=stab_q1;
  if (stab_q2<0) computed_checksum+=65536+stab_q2;
  else computed_checksum+=stab_q2;
  if (stab_q3<0) computed_checksum+=65536+stab_q3;
  else computed_checksum+=stab_q3;
  computed_checksum+=timerticks;
  computed_checksum=computed_checksum%65536;
  if (checksum!=computed_checksum) return -3; //invalid checksum
    
  real qw, qx, qy, qz;
  qw=((real)stab_q0)/8192;
  qx=((real)stab_q3)/8192;
  qy=((real)stab_q1)/8192;
  qz=((real)stab_q2)/8192;
  quaternion2 quat(qw,qx,qy,qz);
  quat.normalize();
  //rotate up by 90 deg
  quaternion2 rotateup(0.0,-90.0,0.0);
  quat=rotateup*quat;
  tmatrix matrix=quat.getRotMatrix();

  matrix.getRPY(pData->dRoll,pData->dPitch,pData->dYaw);
  //convert to degrees
  const double dPI = 3.14159265359;
  pData->dRoll*=180/dPI;
  pData->dPitch*=180/dPI;
  pData->dYaw*=180/dPI;
  
  //check for accelerometer data 
  if (m_data[deviceindex].dataptr[13]!=0||m_data[deviceindex].dataptr[14]!=0||
    m_data[deviceindex].dataptr[15]!=0||m_data[deviceindex].dataptr[16]!=0) {
    short accX=0, accY=0, accZ=0;
    double dAccX=0.0, dAccY=0.0, dAccZ=0.0;
    accX = (m_data[deviceindex].dataptr[9]<<8)+m_data[deviceindex].dataptr[10];
    accY = (m_data[deviceindex].dataptr[13]<<8)+m_data[deviceindex].dataptr[14];
    accZ = (m_data[deviceindex].dataptr[15]<<8)+m_data[deviceindex].dataptr[16];
    dAccX=accX/8192.0; dAccY=accY/8192.0; dAccZ=accZ/8192.0;
    pData->accX=-dAccZ;
    pData->accY=-dAccX;
    pData->accZ=-dAccY;
  }
  return 0;  
}


/*
Name:    TestCheckSum
Purpose:  tests the checksum for a given sample of ShapeTape data
Accepts:  tapedata = buffer where ShapeTape bytes are stored
      buffersize = size of buffer where ShapeTape bytes are stored
      sernum = serial number of ShapeTape.
Returns:    < 0 indicates error:
        -1 = buffer size is too small
        -2 = checksum is invalid
        -3 = all bytes are zero (checksum works, but data is still invalid).
      0 if checksum tests out ok
*/
int NetworkDataCollector::TestCheckSum(unsigned char *tapedata, int buffersize, int sernum) {
  //check to make sure that potential packet is valid
  if (buffersize<16||buffersize>120) return -1;
  int startpos;
  if (sernum>4095) startpos=16;
  else startpos=3;
  int endpos=buffersize-1; //don't include CRC byte
  int i;
  unsigned char checksum=0;
  int totalval=0;
  for (i=startpos;i<endpos;i++) {
    checksum+=tapedata[i];
    totalval+=tapedata[i];
  }
  if (checksum!=tapedata[endpos])  return -2;
  else if (totalval==0) return -3;//check for case in which all data elements are 0
  return 0;
}
