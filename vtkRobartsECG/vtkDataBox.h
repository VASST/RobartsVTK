/*=========================================================================

  Program:   Data Acquisition box for the USB 9800 series for VTK
  Module:    $RCSfile: vtkDataBox.h,v $
  Creator:   Chris Wedlake <cwedlake@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: cwedlake $
  Date:      $Date: 2007/04/19 12:48:52 $
  Version:   $Revision: 1.1 $

==========================================================================

Copyright (c) 2000-2005
All rights reserved.

THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE SOFTWARE "AS IS"
WITHOUT EXPRESSED OR IMPLIED WARRANTY INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  IN NO EVENT SHALL ANY COPYRIGHT HOLDER OR OTHER PARTY WHO MAY
MODIFY AND/OR REDISTRIBUTE THE SOFTWARE UNDER THE TERMS OF THIS LICENSE
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, LOSS OF DATA OR DATA BECOMING INACCURATE
OR LOSS OF PROFIT OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF
THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES. 

=========================================================================*/
#ifndef __vtkDataBox_h
#define __vtkDataBox_h

#include "vtkObject.h"
//#include "vtkDataBoxStructureH.h"
#include "vtkDataBoxBuffer.h"

#include <string.h>
#include "./databox/OLDAAPI.h"
#include "./databox/OLDADEFS.h"
#include "./databox/OLTYPES.h"
#include "./databox/Olmem.h"
#include "./databox/OLERRORS.h"
#include "./databox/Graphs.h"
#include "vtkCriticalSection.h"
#include "vtkMutexLock.h"

#define NUM_BUFFERS 6
#define THERMOGAIN  100
#define STRLEN 80        /* string size for general text manipulation   */
#define MAXCHANNELS 16

#define DATABOX_TOBUFFER 0
#define DATABOX_TOFILE 1

/* simple structure used with board */

typedef struct tag_board {
   HDEV  hdrvr;        /* device handle            */
   HDASS hdass;        /* sub system handle        */
   ECODE status;       /* board error status       */
   char name[STRLEN];  /* string for board name    */
   char entry[STRLEN]; /* string for board name    */
} BOARD;

typedef BOARD* LPBOARD;

class VTK_EXPORT vtkDataBox : public vtkObject
{
   //BTX
public:
     //ETX
  vtkCriticalSection *UpdateMutex;
  vtkTimeStamp UpdateTime;
  double InternalUpdateRate; 

  void GetSingleValue(long * value, unsigned int channel, double gain);
  long GetSingleValue(unsigned int channel, double gain);
  void PutSingleValue(long value, unsigned int channel, double gain);

  static vtkDataBox *New();
  vtkTypeMacro(vtkDataBox,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  int OpenBox();
  int OpenBoxOrg();
  int CloseBox();
  void StartBox();
  void StopBox();
  //void StopBoxBuffer();
  //void StartBoxBuffer();

  void SetNumberOfChannels(int channels);
  int GetNumberOfChannels();
/*
  //Not likely going to be accessable!
  int GetNumCurrentChannels(){return this->numCurrentChannels;}
  //Not likely going to be accessable!
  void SetNumCurrentChannels(int channels){
    this->numCurrentChannels = channels;
    data->SetNumCurrentChannels(channels);
  }
*/

  FILE * GetFileOut();
  void SetFileOut(char *);

  char * GetDriverVersion();
  char * GetErrorString();
  char * GetSDKVersion();
  char * GetDeviceName();

  void SetGain(double gain);
  double GetGain();
  void SetDMA(unsigned int dma);
  unsigned int GetDMA();
  void SetFrequency(double freq);
  double GetFrequency();
  void SetBufferLength(double bufferLength);
  double GetBufferLength();
  double GetChannelsCurrentValue(int channel);
//  vtkDataBoxStructure * GetAllCurrentValues();
  void CodeToVolts(unsigned int resolution, unsigned int encoding, long value, double * voltage);

  // NOT LIKELY!
  // Write the tracking information to a file
//  void WriteToFile2D(const char *filename){ data->WriteToFile(filename); }

  int PrintCapabilitiesAll();
  vtkDataBoxBuffer * GetBuffer(); 
  void SetNotificationProcedureToFile();
  void SetNotificationProcedureToBuffer();

protected:
  vtkDataBox();
  ~vtkDataBox();

  int Probe();

  bool CheckError(int errorCode);
  void CreateBuffers(unsigned long size, unsigned int samplesize);
  void FreeBuffers();

  void ConfigureNotificationProcedure();

  void GetNotificationProcedure(OLNOTIFYPROC procedure);

  void GetInternalTimeStamp(HBUF buffer, LPTDS timestamp);
  
  unsigned long GetInternalTimeStamp(HBUF buffer);


private:
  vtkDataBox(const vtkDataBox&);
  void operator=(const vtkDataBox&);

  void ClearString();
  void GetDeviceHandle(unsigned int numberADs);

  int AllocateSubSystem(unsigned int currentAD);

  void SetChannelTypeToSingle();
  void SetChannelTypeToDifferential();

  void SetEncodingToBinary();
  void SetEncodingTo2sComplements();

  void GetBuffers(HBUF * buffer);
  void PutBuffers(HBUF buffer);

  bool CheckCapabilitiesSingleValue();
  bool CheckCapabilitiesContinuousValue();
  bool CheckCapabilitiesContinuousPretriggerValue();
  bool CheckCapabilitiesContinuousPrePosttriggerValue();
  bool CheckCapabilitiesSimultaneousOperations();
  bool CheckCapabilitiesPausingOperations();
  bool CheckCapabilitiesAsynchronousOperations();
  bool CheckCapabilitiesBufferingOperations();
  bool CheckCapabilitiesSingleBufferWrapping();
  bool CheckCapabilitiesMultiBufferWrapping();
  bool CheckCapabilitiesInProcessFlush();
  unsigned int CheckNumberOfDMAChannels();
  bool CheckCapabilitiesGapFreeNoDMA();
  bool CheckCapabilitiesGapFreeSingleDMA();
  bool CheckCapabilitiesGapFreeDualDMA();
  bool CheckCapabilitiesTriggeredScans();
  unsigned int CheckMaximumMultiScans();
  bool CheckCapabilitiesScanPerTrigger();
  bool CheckCapabilitiesInternalTrigger();
  bool CheckCapabilitiesExtraTrigger();
  unsigned int CheckNumberOfChannelGainEntries();
  bool CheckCapabilitiesRandomChannelGainListSetup();
  bool CheckCapabilitiesSequentialChannelGainListSetup();
  bool CheckCapabilitiesSequentialChannelGainListSetupOnZero();
  bool CheckCapabilitiesSimultaneousSampleAndHold();
  bool CheckCapabilitiesChannelGainListInhibition();
  bool CheckCapabilitiesProgrammableGain();
  unsigned int CheckNumberOfGainSelections();
  bool CheckCapabilitiesAutoranging();
  bool CheckCapabilitiesSynchronousDigitalOutput();
  unsigned int CheckMaximumDigialChannelList();
  unsigned int NumberOfChannels();
  bool CheckCapabilitiesChannellExpansionWithDT2896();
  bool CheckCapabilitiesChannellExpansionWithDT727();
  bool CheckCapabilitiesSingleEndedInputs();
  unsigned int NumberOfSingleEndedChannels();
  bool CheckCapabilitiesDifferentialInputs();
  unsigned int NumberOfDifferentialChannels();
  bool CheckCapabilitiesFilteringPerChannel();
  unsigned int NumberOfFilterSelections();
  bool CheckCapabilitiesRangePerChannel();
  unsigned int NumberOfRangeSelections();
  bool CheckCapabilitiesProgrammableResolution();
  unsigned int NumberOfProgrammableResolutions();
  bool CheckCapabilitiesBinaryEncoding();
  bool CheckCapabilitiesTwosComplementEncoding();
  bool CheckCapabilitiesInternalSoftwareTrigger();
  bool CheckCapabilitiesExternalDigitalTrigger();
  bool CheckCapabilitiesPositiveAnalogTrigger();
  bool CheckCapabilitiesNegativeAnalogTrigger();
  bool CheckCapabilitiesAnalogEventTrigger();
  bool CheckCapabilitiesDigitalEventTrigger() ;
  bool CheckCapabilitiesTimerEventTrigger();
  unsigned int NumberOfExtraTriggerSources();
  bool CheckCapabilitiesInternalClock();
  bool CheckCapabilitiesExternalClock();
  unsigned int NumberOfExtraClockSources();
  bool CheckCapabilitiesCascading();
  bool CheckCapabilitiesEventCounting();
  bool CheckCapabilitiesRateGeneration();
  bool CheckCapabilitiesOneShotMode() ;
  bool CheckCapabilitiesRepetitiveOneShotMode();
  bool CheckCapabilitiesHigh2LowPulse() ;
  bool CheckCapabilitiesLow2HighPulse();
  bool CheckCapabilitiesInternalGate();
  bool CheckCapabilitiesHighLevelGate();
  bool CheckCapabilitiesLowLevelGate();
  bool CheckCapabilitiesHighEdgeGate();
  bool CheckCapabilitiesLowEdgeGate();
  bool CheckCapabilitiesLevelChangeGate();
  bool CheckCapabilitiesHighLevelGateWithDebounce();
  bool CheckCapabilitiesLowLevelGateWithDebounce();
  bool CheckCapabilitiesHighEdgeGateWithDebounce();
  bool CheckCapabilitiesLowEdgeGateWithDebounce();
  bool CheckCapabilitiesLevelChangeGateWithDebounce();
  bool CheckCapabilitiesIOInterrupt();
  bool CheckCapabilitiesFIFO();
  bool CheckCapabilitiesProcessorOnBoard();
  bool CheckCapabilitiesSoftwareCalibration();

  void SetFlowToSingleValue();
  void SetFlowToContinuous();
  void SetFlowToContinuousPretrigged();
  void SetFlowToContinuousPrePosttrigged();
  unsigned int GetFlowType();
  void SetBufferWrapModeToNone();
  void SetBufferWrapModeToMultiple();
  void SetBufferWrapModeToSingle();
  unsigned int GetBufferWrapMode();
  unsigned int GetNumberOfDMAChannels(); 
  void SetNumberOfDMAChannels(unsigned int value);
  void SetTriggeredScanUsageOn();
  void SetTriggeredScanUsageOff();
  bool GetTriggeredScanUsage();
  void SetMultiscanCount(unsigned int value);
  unsigned int GetMultiscanCount();

  void SetTriggerModeSoft();
  void SetTriggerModeExtern();
  void SetTriggerModeThresh();
  void SetTriggerModeAnalogEvent();
  void SetTriggerModeDigitalEvent();
  void SetTriggerModeTimerEvent();
  unsigned int GetTriggerMode();
  void SetTriggerInternalFrequency(unsigned int value);
  unsigned int GetTriggerInternalFrequency();
  
  void SetChannelListSize(unsigned int value);
  unsigned int GetChannelListSize();
  void SetChannelListEntry(unsigned int entry, unsigned int channel);
  unsigned int GetChannelListEntry(unsigned int entry);
  void SetGainListEntry(unsigned int entry, unsigned int channel);
  double GetGainListEntry(unsigned int entry);
  void SetChannelListEntryInhibit(unsigned int entry, bool inhibit);
  bool GetChannelListEntryInhibit(unsigned int entry);
  void SetDigitalIOListEntry(unsigned int entry, unsigned int value);
  unsigned int GetDigitalIOListEntry(unsigned int entry);
  void SetSynchronousDigitalIOUsage(bool use);
  bool GetSynchronousDigitalIOUsage();
  void SetChannelType(unsigned int value);
  unsigned int GetChannelType();
  void SetChannelFilter(unsigned int channel, double cutOffFrequency);
  double GetChannelFilter(unsigned int channel);
  void SetRange(double max, double min);
  void GetRange(double *max, double *in);
  void SetChannelRange(unsigned int channel, double max, double min);
  void GetChannelRange(unsigned int channel, double max, double min);
  void SetResolution(unsigned int bits);
  unsigned int GetResolution();
  void SetEncoding(unsigned int encode);
  unsigned int GetEncoding();
  void SetTrigger(unsigned int trigger);
  unsigned int GetTrigger();
  void SetPretriggerSource(unsigned int trigger);
  unsigned int GetPretriggerSource();
  void SetRetrigger(unsigned int trigger);
  unsigned int GetRetrigger();
  void SetClockSource(unsigned int clock);
  unsigned int GetClockSource();
  void SetClockFrequency(double frequency) ;
  double GetClockFrequency();
  void SetExternalClockDivider(unsigned long divider);
  unsigned long GetExternalClockDivider();
  void SetCTMode(unsigned int ctMode);
  unsigned int GetCTMode();
  void SetCascadeMode(unsigned int cascadeMode);
  unsigned int GetCascadeMode();
  void SetGateType(unsigned int gate);
  unsigned int GetGateType();
  void SetPulseType(unsigned int pulse);
  unsigned int GetPulseType();
  void SetPulseWidth(double pulseWidthPercent);
  double GetPulseWidth();
  void PutSingleValue2(long value, unsigned int channel, double gain);
  long GetSingleValue2(unsigned int channel, double gain);
  void GetSingleValueEx(unsigned int channel, int autoRange, double gain, long valueCounts, double valueVolts);
  void ConfigOperations();
  void StartOperations();
  void PauseOperations();
  void ContinueOperations();
  void StopOperations();
  void AbortOperations();
  void ResetOperations();
  void FlushBuffers();
  unsigned long ReadEvents();
  /* ATTRIBUTES */

  double Gain;
  unsigned int DMA;
  double Freq;
  double BufferLength;

  // Purpose?
  double retriggerFrequency;
  int numCurrentChannels;

  HBUF buffers[NUM_BUFFERS];

  BOARD board;
  
  char str[STRLEN];        /* global string for general text manipulation */

  int deviceOpen;
  int deviceStarted;

  vtkTimeStamp timestamp;
  LPTDS internalTimestamp;
  unsigned long internalTime;
  bool singleValues;
  vtkDataBoxBuffer * data;
  
  int deviceSingle;

  int CutOffFreqFilter;
  int NotificationProcedure;
  int bufferSize;
  FILE * FileOut;
  int FileOpen;

  //BTX
  friend int CALLBACK UpdateInformationToBuffer(UINT message, WPARAM wParam, vtkDataBox * lParam );
  friend int CALLBACK UpdateInformationToFile(UINT message, WPARAM wParam, vtkDataBox * lParam );

  //ETX

  int bufferStarted;
};


#endif