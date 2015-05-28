/*=========================================================================

  Program:   Data Acquisition box for the USB 9800 series for VTK
  Module:    $RCSfile: vtkDataBox.cxx,v $
  Creator:   Chris Wedlake <cwedlake@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: cwedlake $
  Date:      $Date: 2008/06/04 12:35:34 $
  Version:   $Revision: 1.2 $

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

#include <limits.h>
#include <float.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
//#include <iostream.h>
#include <stdio.h>
#include <sys/types.h>
#include <string.h>
#include <sys/timeb.h>
#include <windows.h>

#include "vtkDataBox.h"
#include "vtkObjectFactory.h"


#ifndef __CALLBACKGetDriver
#define __CALLBACKGetDriver
int CALLBACK GetDriver( char * lpszName, char * lpszEntry, LPARAM lParam )   
/*
this is a callback function of olDaEnumBoards, it gets the 
strings of the Open Layers board and attempts to initialize
the board.  If successful, enumeration is halted.
*/

{
   LPBOARD lpboard = (LPBOARD)(LPVOID)lParam;
   
   // fill in board strings 

   lstrcpyn(lpboard->name,lpszName,STRLEN);
   lstrcpyn(lpboard->entry,lpszEntry,STRLEN);

   // try to open board 

   lpboard->status = olDaInitialize(lpszName,&lpboard->hdrvr);
   if   (lpboard->hdrvr != NULL)
      return FALSE;          // false to stop enumerating 
   else                      
      return TRUE;           // true to continue          
}
#endif __CALLBACKGetDriver

//----------------------------------------------------------------------------
vtkDataBox* vtkDataBox::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkDataBox");
  if(ret)
    {
    return (vtkDataBox*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkDataBox;
}

bool vtkDataBox::CheckError(int errorCode) {
  this->board.status = errorCode;
  if (this->board.status != OLNOERROR) {
    if (deviceOpen==1) {
       cout << "Error:" <<errorCode<< "\t" <<this->GetErrorString() << endl;
       return 0;
    }
    else {
       cout << "Error:" <<errorCode<< "\t" <<this->GetErrorString() << endl;
       olDaReleaseDASS(this->board.hdass);
       olDaTerminate(this->board.hdrvr);
    }
    return 0;
  }

  return 1;  // no error
}

vtkDataBox::vtkDataBox()
{
  this->data = vtkDataBoxBuffer::New();

  this->deviceOpen = 0;
  this->deviceStarted = 0;
  this->Gain = 1.0;

  this->CutOffFreqFilter = 250.0;

  // Never used
  this->BufferLength = 100000;
  // Update Frequency
  this->Freq = 96000;
  // Single or Continues Device
  this->deviceSingle=0;
  this->internalTimestamp = new TDS();

  this->UpdateMutex = vtkCriticalSection::New();

   //
  this->bufferStarted = 0;
  this->bufferSize=20;

  this->FileOut=NULL;
  this->FileOpen=0;
  
  // Need accessor methods to set 
  this->retriggerFrequency = 975;
  this->numCurrentChannels = 16;
  data->SetNumberOfChannels(this->numCurrentChannels);
  this->NotificationProcedure = DATABOX_TOBUFFER;
}

//----------------------------------------------------------------------------
vtkDataBox::~vtkDataBox() 
{
  if (this->deviceOpen == 1) {
    this->CloseBox();
  }

    this->UpdateMutex->Delete();
  if (this->FileOpen) {
    fclose(this->FileOut);
    this->FileOpen=0;
  }
}

void vtkDataBox::ClearString() {
  for (int i=0; i< STRLEN; i++) {
    str[i]= '\0';
  }
}

FILE * vtkDataBox::GetFileOut() {
  return this->FileOut;
}


void vtkDataBox::SetFileOut(char * filename) {
  FILE * file = fopen(filename, "w");
  this->FileOpen=1;
  this->FileOut = file;
}

int CALLBACK UpdateInformationToFile(UINT message, WPARAM wParam, vtkDataBox * lParam )
{
  static int PerformanceCounterInit=0;

  unsigned int encoding;
  unsigned int resolution;
  PWORD  pBuffer = NULL;
  HBUF buffer = NULL;
  double LatestValues[MAXCHANNELS];
  double * temp= new double();
  vtkDataBox *box;
  box = lParam;

  unsigned long numValidSamples;
  double timestampCurrent;

  // high accuracy timing
  //============================================================================
  static LONGLONG QPart1;
  static double freqCPU,timestampStart;
  LARGE_INTEGER litmp; 


  if(!PerformanceCounterInit){  //Initialization
    PerformanceCounterInit=1;
    // Get starting time

    struct _timeb timeTmp;
    _ftime(&timeTmp);  // seconds from Jan. 1, 1970
    timestampStart = timeTmp.time + 0.001*timeTmp.millitm;

    QueryPerformanceFrequency(&litmp);
    freqCPU = (double)litmp.QuadPart;
    QueryPerformanceCounter(&litmp);
    QPart1 = litmp.QuadPart;  // starting counter

    double tmp = QPart1/freqCPU;
    //printf("QueryPerformanceCounter=%ld %ld %lf\n", litmp.HighPart, litmp.LowPart, tmp);
  }

  QueryPerformanceCounter(&litmp);
  double diffTime = (double)(litmp.QuadPart-QPart1) / freqCPU;

  timestampCurrent = timestampStart + diffTime;
  //============================================================================

  switch (message) {
    case OLDA_WM_BUFFER_REUSED:   /* message: buffer reused  */
      cout << "Reuses?" << endl;
      break;
    case OLDA_WM_BUFFER_DONE:     /* message: buffer done  */
      if (box->deviceStarted) {
        box->GetBuffers(&buffer);
        if (buffer) {
          encoding = box->GetEncoding();
          resolution = box->GetResolution();
          box->CheckError(olDmGetBufferPtr( buffer,(LPVOID*)&pBuffer));

          int numChannels = box->numCurrentChannels;
          if (numChannels > MAXCHANNELS) {
            numChannels = MAXCHANNELS;
          }

          box->CheckError(olDmGetValidSamples( buffer,&numValidSamples));

          box->UpdateMutex->Lock();
          box->timestamp.Modified();
              
          int numSamples = numValidSamples/numChannels;
          FILE * fileOut = box->GetFileOut();
//          if(box->bufferStarted==1){
            double samplingFreq = box->retriggerFrequency;
            for(int iSample=0; iSample < numSamples; iSample++){
              double timestampSample = timestampCurrent - (numSamples-1-iSample)/samplingFreq;
              fprintf(fileOut,"%f , ", timestampSample);
              for (int i=0; i < numChannels; i++){
                LatestValues[i] = (WORD)pBuffer[i+iSample*numChannels];
                box->CodeToVolts(resolution, encoding, LatestValues[i], temp);
                LatestValues[i] = *temp;
                fprintf(fileOut,"%f , ", LatestValues[i]);
              }
              fprintf(fileOut,"\n");
            }
//          }
          box->UpdateMutex->Unlock();
           box->PutBuffers(buffer);
        }
      }  
      return (TRUE);   /* Did process a message */       

    case OLDA_WM_QUEUE_DONE:
    case OLDA_WM_QUEUE_STOPPED:
      cout << "ALL DONE" << endl;
      return (TRUE);   /* Did process a message */    
    case OLDA_WM_TRIGGER_ERROR:
      cout << "ERROR" << endl;
      box->GetErrorString();
      return (TRUE);   /* Did process a message */        
    case OLDA_WM_OVERRUN_ERROR:
      cout << "ERROR" << endl;
      box->GetErrorString();
      return (TRUE);   /* Did process a message */ 
  }                               
  return (FALSE);               /* Didn't process a message */
}



int CALLBACK UpdateInformationToBuffer(UINT message, WPARAM wParam, vtkDataBox * lParam )
{
  static int PerformanceCounterInit=0;

  unsigned int encoding;
  unsigned int resolution;
  PWORD  pBuffer = NULL;
  HBUF buffer = NULL;
  double LatestValues[MAXCHANNELS];
  double * temp= new double();
  vtkDataBox *box;
  box = lParam;

  long flags = 0;
  unsigned long numValidSamples;
  double timestampCurrent;

  // high accuracy timing
  //============================================================================
  static LONGLONG QPart1;
  static double freqCPU,timestampStart;
  LARGE_INTEGER litmp; 


  if(!PerformanceCounterInit){  //Initialization
    PerformanceCounterInit=1;
    // Get starting time

    struct _timeb timeTmp;
    _ftime(&timeTmp);  // seconds from Jan. 1, 1970
    timestampStart = timeTmp.time + 0.001*timeTmp.millitm;

    QueryPerformanceFrequency(&litmp);
    freqCPU = (double)litmp.QuadPart;
    QueryPerformanceCounter(&litmp);
    QPart1 = litmp.QuadPart;  // starting counter

    double tmp = QPart1/freqCPU;
    //printf("QueryPerformanceCounter=%ld %ld %lf\n", litmp.HighPart, litmp.LowPart, tmp);
  }

  QueryPerformanceCounter(&litmp);
  double diffTime = (double)(litmp.QuadPart-QPart1) / freqCPU;

  timestampCurrent = timestampStart + diffTime;
  //============================================================================

  switch (message) {
    case OLDA_WM_BUFFER_REUSED:   /* message: buffer reused  */
      cout << "Reuses?" << endl;
      break;
    case OLDA_WM_BUFFER_DONE:     /* message: buffer done  */
      if (box->deviceStarted) {
        box->GetBuffers(&buffer);
        if (buffer) {
          encoding = box->GetEncoding();
          resolution = box->GetResolution();
          box->CheckError(olDmGetBufferPtr( buffer,(LPVOID*)&pBuffer));

          int numChannels = box->numCurrentChannels;
          if (numChannels > MAXCHANNELS) {
            numChannels = MAXCHANNELS;
          }
          box->CheckError(olDmGetValidSamples( buffer,&numValidSamples));

          box->UpdateMutex->Lock();
          box->timestamp.Modified();
              
          int numSamples = numValidSamples/numChannels;

          double samplingFreq = box->retriggerFrequency;
          for(int iSample=0; iSample < numSamples; iSample++){
            double timestampSample = timestampCurrent - (numSamples-1-iSample)/samplingFreq;
            for (int i=0; i < numChannels; i++){
              LatestValues[i] = (WORD)pBuffer[i+iSample*numChannels];
              box->CodeToVolts(resolution, encoding, LatestValues[i], temp);
              LatestValues[i] = *temp;
              box->data->AddItem(LatestValues[i], i, timestampSample);
            }
          }
          box->UpdateMutex->Unlock();
           box->PutBuffers(buffer);
        }
      }  
      return (TRUE);   /* Did process a message */       

    case OLDA_WM_QUEUE_DONE:
    case OLDA_WM_QUEUE_STOPPED:
      cout << "ALL DONE" << endl;
      return (TRUE);   /* Did process a message */    
    case OLDA_WM_TRIGGER_ERROR:
      cout << "ERROR" << endl;
      box->GetErrorString();
      return (TRUE);   /* Did process a message */        
    case OLDA_WM_OVERRUN_ERROR:
      cout << "ERROR" << endl;
      box->GetErrorString();
      return (TRUE);   /* Did process a message */ 
  }                               
  return (FALSE);               /* Didn't process a message */
}

void vtkDataBox::ConfigureNotificationProcedure() {
   if (this->NotificationProcedure == DATABOX_TOFILE) {
    this->CheckError(olDaSetNotificationProcedure(this->board.hdass, (OLNOTIFYPROC)UpdateInformationToFile, (long)&*this));
   }
   else if (this->NotificationProcedure == DATABOX_TOBUFFER) {
    this->CheckError(olDaSetNotificationProcedure(this->board.hdass, (OLNOTIFYPROC)UpdateInformationToBuffer, (long)&*this));
   }
   this->ConfigOperations();
}

void vtkDataBox::SetNotificationProcedureToFile(){
  this->NotificationProcedure = DATABOX_TOFILE;  
}

void vtkDataBox::SetNotificationProcedureToBuffer(){
  this->NotificationProcedure = DATABOX_TOBUFFER;
}

void vtkDataBox::GetNotificationProcedure(OLNOTIFYPROC procedure){
  this->CheckError(olDaGetNotificationProcedure(this->board.hdass, &procedure));
  this->ConfigOperations();
}

char * vtkDataBox::GetErrorString(){
  ClearString();
  olDaGetErrorString(this->board.status, str, STRLEN);
  return str;
}

char * vtkDataBox::GetDriverVersion(){
  ClearString();
  this->CheckError(olDaGetDriverVersion(this->board.hdrvr, str, STRLEN));
  return str;
}

char * vtkDataBox::GetSDKVersion(){
  ClearString();
  this->CheckError(olDaGetVersion(str, STRLEN));
  return str;
}

char * vtkDataBox::GetDeviceName(){
  ClearString();
  this->CheckError(olDaGetDeviceName(this->board.hdrvr, str, STRLEN));
  return str;
}

void vtkDataBox::PrintSelf(ostream& os, vtkIndent indent)
{
  cout << "Device: " << &board << endl;
  cout << "\tName:" << board.name << endl;
  cout << "\tEntry:" << board.entry << endl;
  cout << "\tError Code:" << board.status << endl;
  cout << "\tError String:" << this->GetErrorString() << endl;
  cout << "\tHandle:" << board.hdrvr << endl;
  cout << "\tSubsystem Handle:" << board.hdass << endl;
  cout << "Open:" << deviceOpen << endl;
  cout << "Start:" << deviceStarted << endl;
  cout << "Driver Version:" << this->GetDriverVersion() << endl;
  cout << "SDK Version:" << this->GetSDKVersion() << endl;
  cout << endl;
}

/*  olDaGetDevCaps */

void vtkDataBox::SetFlowToSingleValue() {
    if (this->deviceStarted==0)
      this->deviceSingle=1;
    else
     cout << "Device already started... stop, call this method then reconfigure" <<endl;
  this->CheckError(olDaSetDataFlow(this->board.hdass,OL_DF_SINGLEVALUE));
}

void vtkDataBox::SetFlowToContinuous() {
   if (this->deviceStarted==0)
      this->deviceSingle=0;
   else
     cout << "Device already started... stop, call this method then reconfigure" <<endl;

  this->CheckError(olDaSetDataFlow(this->board.hdass,OL_DF_CONTINUOUS));
}

void vtkDataBox::SetFlowToContinuousPretrigged() {
  this->CheckError(olDaSetDataFlow(this->board.hdass,OL_DF_CONTINUOUS_PRETRIG));
}

void vtkDataBox::SetFlowToContinuousPrePosttrigged() {
  this->CheckError(olDaSetDataFlow(this->board.hdass,OL_DF_CONTINUOUS_ABOUTTRIG));
}

unsigned int vtkDataBox::GetFlowType() {
  unsigned int value;
  this->CheckError(olDaGetDataFlow(this->board.hdass,&value));
  return value;
}

void vtkDataBox::SetBufferWrapModeToNone(){ 
  this->CheckError(olDaSetWrapMode(this->board.hdass, OL_WRP_NONE));
}

void vtkDataBox::SetBufferWrapModeToMultiple(){ 
  this->CheckError(olDaSetWrapMode(this->board.hdass, OL_WRP_MULTIPLE));
}

void vtkDataBox::SetBufferWrapModeToSingle(){ 
  this->CheckError(olDaSetWrapMode(this->board.hdass, OL_WRP_SINGLE));
}

unsigned int vtkDataBox::GetBufferWrapMode() {
  unsigned int value;
  this->CheckError(olDaGetWrapMode(this->board.hdass,&value));
  return value;
}

void vtkDataBox::SetNumberOfDMAChannels(unsigned int value) {
  if (this->CheckError(olDaSetDmaUsage(this->board.hdass,value))){
    this->DMA=value;
  }
}

unsigned int vtkDataBox::GetNumberOfDMAChannels() {
  unsigned int value;
  this->CheckError(olDaGetDmaUsage(this->board.hdass,&value));
  return value;
}

void vtkDataBox::SetNumberOfChannels(int channels) {
  this->numCurrentChannels = channels;
  this->data->SetNumberOfChannels(this->numCurrentChannels);
}

int vtkDataBox::GetNumberOfChannels() {
  return this->numCurrentChannels;
}


void vtkDataBox::SetTriggeredScanUsageOn() {
  this->CheckError(olDaSetTriggeredScanUsage(this->board.hdass,TRUE));
}

void vtkDataBox::SetTriggeredScanUsageOff() {
  this->CheckError(olDaSetTriggeredScanUsage(this->board.hdass,FALSE));
}

bool vtkDataBox::GetTriggeredScanUsage() {
  int value;
  this->CheckError(olDaGetTriggeredScanUsage(this->board.hdass,&value));
  if (value == 1)
    return true;
  return false;
}

void vtkDataBox::SetMultiscanCount(unsigned int value) {
  this->CheckError(olDaSetMultiscanCount(this->board.hdass,value));
}

unsigned int vtkDataBox::GetMultiscanCount() {
  unsigned int value;
  this->CheckError(olDaGetMultiscanCount(this->board.hdass,&value));
  return value;  
}

void vtkDataBox::SetTriggerModeSoft() {
  this->CheckError(olDaSetRetrigger(this->board.hdass,OL_TRG_SOFT));
}

void vtkDataBox::SetTriggerModeExtern() {
  this->CheckError(olDaSetRetrigger(this->board.hdass,OL_TRG_EXTERN));
}

void vtkDataBox::SetTriggerModeThresh() {
  this->CheckError(olDaSetRetrigger(this->board.hdass,OL_TRG_THRESH));
}

void vtkDataBox::SetTriggerModeAnalogEvent() {
  this->CheckError(olDaSetRetrigger(this->board.hdass,OL_TRG_ANALOGEVENT));
}

void vtkDataBox::SetTriggerModeDigitalEvent() {
  this->CheckError(olDaSetRetrigger(this->board.hdass,OL_TRG_DIGITALEVENT));
}

void vtkDataBox::SetTriggerModeTimerEvent() {
  this->CheckError(olDaSetRetrigger(this->board.hdass,OL_TRG_TIMEREVENT));
}

unsigned int vtkDataBox::GetTriggerMode() {
  unsigned int value;
  this->CheckError(olDaGetRetrigger(this->board.hdass,&value));
  return value;  
}

void vtkDataBox::SetTriggerInternalFrequency(unsigned int value) {
  this->CheckError(olDaSetRetriggerFrequency(this->board.hdass,value));
}

unsigned int vtkDataBox::GetTriggerInternalFrequency() {
  double value;
  this->CheckError(olDaGetRetriggerFrequency(this->board.hdass,&value));
  return value;  
}

void vtkDataBox::SetChannelListSize(unsigned int value) {
  this->CheckError(olDaSetChannelListSize(this->board.hdass,value));
}

unsigned int vtkDataBox::GetChannelListSize() {
  unsigned int value;
  this->CheckError(olDaGetChannelListSize(this->board.hdass,&value));
  return value;  
}

void vtkDataBox::SetChannelListEntry(unsigned int entry, unsigned int channel) {
  this->CheckError(olDaSetChannelListEntry(this->board.hdass,entry, channel));
}

unsigned int vtkDataBox::GetChannelListEntry(unsigned int entry) {
  unsigned int value;
  this->CheckError(olDaGetChannelListEntry(this->board.hdass,entry, &value));
  return value;  
}

void vtkDataBox::SetGainListEntry(unsigned int entry, unsigned int channel) {
  this->CheckError(olDaSetGainListEntry(this->board.hdass,entry, channel));
}

double vtkDataBox::GetGainListEntry(unsigned int entry) {
  double value;
  this->CheckError(olDaGetGainListEntry(this->board.hdass,entry, &value));
  return value;  
}

void vtkDataBox::SetChannelListEntryInhibit(unsigned int entry, bool inhibit) {
  this->CheckError(olDaSetChannelListEntryInhibit(this->board.hdass,entry, inhibit));
}

bool vtkDataBox::GetChannelListEntryInhibit(unsigned int entry) {
  int inhibit;
  this->CheckError(olDaGetChannelListEntryInhibit(this->board.hdass,entry, &inhibit));
  if (inhibit == 0)
    return false;
  return true;  
}

void vtkDataBox::SetDigitalIOListEntry(unsigned int entry, unsigned int value) {
  this->CheckError(olDaSetDigitalIOListEntry(this->board.hdass,entry, value));
}

unsigned int vtkDataBox::GetDigitalIOListEntry(unsigned int entry) {
  unsigned int value;
  this->CheckError(olDaGetDigitalIOListEntry(this->board.hdass,entry, &value));
  return value;  
}

void vtkDataBox::SetSynchronousDigitalIOUsage(bool use) {
  this->CheckError(olDaSetSynchronousDigitalIOUsage(this->board.hdass, use));
}

bool vtkDataBox::GetSynchronousDigitalIOUsage() {
  int use;
  this->CheckError(olDaGetSynchronousDigitalIOUsage(this->board.hdass, &use));
  if (use == 0)
    return false;
  return true;
}

void vtkDataBox::SetChannelType(unsigned int value) {
  this->CheckError(olDaSetChannelType(this->board.hdass, value));
}

void vtkDataBox::SetChannelTypeToSingle() {
  this->CheckError(olDaSetChannelType(this->board.hdass, OL_CHNT_SINGLEENDED));
}

void vtkDataBox::SetChannelTypeToDifferential() {
  this->CheckError(olDaSetChannelType(this->board.hdass, OL_CHNT_DIFFERENTIAL));
}

unsigned int vtkDataBox::GetChannelType() {
  unsigned int value;
  this->CheckError(olDaGetChannelType(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::SetChannelFilter(unsigned int channel, double cutOffFrequency) {
  this->CheckError(olDaSetChannelFilter(this->board.hdass,channel, cutOffFrequency));
}

double vtkDataBox::GetChannelFilter(unsigned int channel) {
  double value;
  this->CheckError(olDaGetChannelFilter(this->board.hdass,channel, &value));
  return value;  
}

void vtkDataBox::SetRange(double max, double min) {
  this->CheckError(olDaSetRange(this->board.hdass, max, min));
}

void vtkDataBox::GetRange(double *max, double *min) {
  this->CheckError(olDaGetRange(this->board.hdass, max, min));
}

void vtkDataBox::SetChannelRange(unsigned int channel, double max, double min) {
  this->CheckError(olDaSetChannelRange(this->board.hdass, channel, max, min));
}

void vtkDataBox::GetChannelRange(unsigned int channel, double max, double min) {
  this->CheckError(olDaGetChannelRange(this->board.hdass, channel, &max, &min));
}

void vtkDataBox::SetResolution(unsigned int bits) {
  this->CheckError(olDaSetResolution(this->board.hdass, bits));
}

unsigned int vtkDataBox::GetResolution() {
  unsigned int value;
  this->CheckError(olDaGetResolution(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::SetEncoding(unsigned int encode) {
  if ((encode == OL_ENC_BINARY) || (encode == OL_ENC_2SCOMP))  {
    this->CheckError(olDaSetEncoding(this->board.hdass,encode));
  }
  else {
    cout << "Invalid Encoding Type" << endl;
  }

}

void vtkDataBox::SetEncodingToBinary() {
  this->CheckError(olDaSetEncoding(this->board.hdass,OL_ENC_BINARY));
}

void vtkDataBox::SetEncodingTo2sComplements() {
  this->CheckError(olDaSetEncoding(this->board.hdass,OL_ENC_2SCOMP));
}

unsigned int vtkDataBox::GetEncoding() {
  unsigned int value;
  this->CheckError(olDaGetEncoding(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::SetTrigger(unsigned int trigger) {
  this->CheckError(olDaSetTrigger(this->board.hdass,trigger));
}

unsigned int vtkDataBox::GetTrigger() {
  unsigned int value;
  this->CheckError(olDaGetTrigger(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::SetPretriggerSource(unsigned int trigger) {
  this->CheckError(olDaSetPretriggerSource(this->board.hdass,trigger));
}

unsigned int vtkDataBox::GetPretriggerSource() {
  unsigned int value;
  this->CheckError(olDaGetPretriggerSource(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::SetRetrigger(unsigned int trigger) {
  this->CheckError(olDaSetRetrigger(this->board.hdass,trigger));
}

unsigned int vtkDataBox::GetRetrigger() {
  unsigned int value;
  this->CheckError(olDaGetRetrigger(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::SetClockSource(unsigned int clock) {
  this->CheckError(olDaSetClockSource(this->board.hdass,clock));
}

unsigned int vtkDataBox::GetClockSource() {
  unsigned int value;
  this->CheckError(olDaGetClockSource(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::SetClockFrequency(double frequency) {
  if (this->CheckError(olDaSetClockFrequency(this->board.hdass,frequency))) 
    this->Freq= frequency;
}

double vtkDataBox::GetClockFrequency() {
  double value;
  this->CheckError(olDaGetClockFrequency(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::SetExternalClockDivider(unsigned long divider) {
  this->CheckError(olDaSetExternalClockDivider(this->board.hdass,divider));
}

unsigned long vtkDataBox::GetExternalClockDivider() {
  unsigned long value;
  this->CheckError(olDaGetExternalClockDivider(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::SetCTMode(unsigned int ctMode) {
  this->CheckError(olDaSetCTMode(this->board.hdass,ctMode));
}

unsigned int vtkDataBox::GetCTMode() {
  unsigned int value;
  this->CheckError(olDaGetCTMode(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::SetCascadeMode(unsigned int cascadeMode) {
  this->CheckError(olDaSetCascadeMode(this->board.hdass,cascadeMode));
}

unsigned int vtkDataBox::GetCascadeMode() {
  unsigned int value;
  this->CheckError(olDaGetCascadeMode(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::SetGateType(unsigned int gate) {
  this->CheckError(olDaSetGateType(this->board.hdass,gate));
}

unsigned int vtkDataBox::GetGateType() {
  unsigned int value;
  this->CheckError(olDaGetGateType(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::SetPulseType(unsigned int pulse) {
  this->CheckError(olDaSetPulseType(this->board.hdass,pulse));
}

unsigned int vtkDataBox::GetPulseType() {
  unsigned int value;
  this->CheckError(olDaGetPulseType(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::SetPulseWidth(double pulseWidthPercent) {
  this->CheckError(olDaSetPulseWidth(this->board.hdass, pulseWidthPercent));
}

double vtkDataBox::GetPulseWidth() {
  double value;
  this->CheckError(olDaGetPulseWidth(this->board.hdass, &value));
  return value;  
}

void vtkDataBox::PutSingleValue2(long value, unsigned int channel, double gain) {
  this->CheckError(olDaPutSingleValue(this->board.hdass, value, channel, gain));
}

long vtkDataBox::GetSingleValue2(unsigned int channel, double gain) {
  if (this->deviceStarted ==0  && this->deviceSingle ==1) {
    this->ConfigOperations();
    this->StartOperations();
    this->deviceStarted=1;
  }
  long value;
  this->CheckError(olDaGetSingleValue(this->board.hdass, &value, channel, gain));
  return value;  
}

void vtkDataBox::GetSingleValueEx(unsigned int channel, int autoRange, double gain, long valueCounts, double valueVolts) {
  this->CheckError(olDaGetSingleValueEx(this->board.hdass, channel, autoRange, &gain, &valueCounts, &valueVolts));
}

void vtkDataBox::ConfigOperations() {
  this->CheckError(olDaConfig(this->board.hdass));
}

void vtkDataBox::StartOperations() {
  this->CheckError(olDaStart(this->board.hdass));
}

void vtkDataBox::PauseOperations() {
  this->CheckError(olDaPause(this->board.hdass));
}

void vtkDataBox::ContinueOperations() {
  this->CheckError(olDaContinue(this->board.hdass));
}

void vtkDataBox::StopOperations() {
  this->CheckError(olDaStop(this->board.hdass));
}

void vtkDataBox::AbortOperations() {
  this->CheckError(olDaAbort(this->board.hdass));
}

void vtkDataBox::ResetOperations() {
  this->CheckError(olDaReset(this->board.hdass));
}

void vtkDataBox::FlushBuffers() {
   this->CheckError(olDaFlushBuffers(this->board.hdass));
}

unsigned long vtkDataBox::ReadEvents() {
  unsigned long value;
    this->CheckError(olDaReadEvents(this->board.hdass, &value));
  return value;
}

//ECODE WINAPI olDaFlushFromBufferInprocess (HDASS hDass, HBUF hBuf, ULNG ulNumSamples);
//ECODE WINAPI olDaGetSSList (PHSSLIST phSSList);

void vtkDataBox::CreateBuffers(unsigned long size, unsigned int samplesize ) {
   HBUF  buffer = NULL;
   HBUF  getBuffer = NULL;
   for (int i=0;i<NUM_BUFFERS;i++)
   {
      this->CheckError(olDmCallocBuffer(0,0, size,samplesize,&buffer));
    buffers[i]=buffer;
      this->CheckError(olDaPutBuffer(this->board.hdass, buffer));
    this->CheckError(olDaGetBuffer(this->board.hdass, &getBuffer));
   }
}

void vtkDataBox::GetBuffers(HBUF * buffer){
    this->CheckError(olDaGetBuffer(this->board.hdass, buffer));
}

void vtkDataBox::PutBuffers(HBUF buffer){
    this->CheckError(olDaPutBuffer(this->board.hdass, buffer));
}

//protected
void vtkDataBox::FreeBuffers() {
   this->FlushBuffers();
   for (int i=0;i<NUM_BUFFERS;i++)
   {
      olDmFreeBuffer(buffers[i]);
   }
   
}

bool vtkDataBox::CheckCapabilitiesSingleValue() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SINGLEVALUE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesContinuousValue() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CONTINUOUS,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesContinuousPretriggerValue() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CONTINUOUS_PRETRIG,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesContinuousPrePosttriggerValue() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CONTINUOUS_ABOUTTRIG,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesSimultaneousOperations() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SIMULTANEOUS_START,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesPausingOperations() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_PAUSE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesAsynchronousOperations() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_POSTMESSAGE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesBufferingOperations() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_BUFFERING,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesSingleBufferWrapping() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_WRPSINGLE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesMultiBufferWrapping() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_WRPMULTIPLE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesInProcessFlush() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_INPROCESSFLUSH,&value));
  if (value == 0) { return true; } else {  return false; }
}

unsigned int vtkDataBox::CheckNumberOfDMAChannels() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_NUMDMACHANS,&value));
  return value;
}

bool vtkDataBox::CheckCapabilitiesGapFreeNoDMA() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GAPFREE_NODMA,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesGapFreeSingleDMA() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GAPFREE_SINGLEDMA,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesGapFreeDualDMA() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GAPFREE_DUALDMA,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesTriggeredScans() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_TRIGSCAN,&value));
  if (value == 0) { return true; } else {  return false; }
}

unsigned int vtkDataBox::CheckMaximumMultiScans() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_MAXMULTISCAN,&value));
  return value;
}

bool vtkDataBox::CheckCapabilitiesScanPerTrigger() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSS_SUP_RETRIGGER_SCAN_PER_TRIGGER,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesInternalTrigger() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSS_SUP_RETRIGGER_SCAN_PER_TRIGGER,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesExtraTrigger() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_RETRIGGER_EXTRA,&value));
  if (value == 0) { return true; } else {  return false; }
}

unsigned int vtkDataBox::CheckNumberOfChannelGainEntries() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_CGLDEPTH,&value));
  return value;
}

bool vtkDataBox::CheckCapabilitiesRandomChannelGainListSetup() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_RANDOM_CGL,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesSequentialChannelGainListSetup() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SEQUENTIAL_CGL,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesSequentialChannelGainListSetupOnZero() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_ZEROSEQUENTIAL_CGL,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesSimultaneousSampleAndHold() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SIMULTANEOUS_SH,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesChannelGainListInhibition() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CHANNELLIST_INHIBIT,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesProgrammableGain() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_PROGRAMGAIN,&value));
  if (value == 0) { return true; } else {  return false; }
}

unsigned int vtkDataBox::CheckNumberOfGainSelections() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_NUMGAINS,&value));
  return value; 
}

bool vtkDataBox::CheckCapabilitiesAutoranging() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SINGLEVALUE_AUTORANGE,&value));
  if (value == 0) { return true; } else {  return false; }
}


bool vtkDataBox::CheckCapabilitiesSynchronousDigitalOutput() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SYNCHRONOUS_DIGITALIO,&value));
  if (value == 0) { return true; } else {  return false; }
}

unsigned int vtkDataBox::CheckMaximumDigialChannelList() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_MAX_DIGITALIOLIST_VALUE,&value));
  return value; 
}

unsigned int vtkDataBox::NumberOfChannels() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_NUMCHANNELS,&value));
  return value; 
}

bool vtkDataBox::CheckCapabilitiesChannellExpansionWithDT2896() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_EXP2896,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesChannellExpansionWithDT727() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_EXP727,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesSingleEndedInputs() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SINGLEENDED,&value));
  if (value == 0) { return true; } else {  return false; }
}

unsigned int vtkDataBox::NumberOfSingleEndedChannels() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_MAXSECHANS,&value));
  return value; 
}

bool vtkDataBox::CheckCapabilitiesDifferentialInputs() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_DIFFERENTIAL,&value));
  if (value == 0) { return true; } else {  return false; }
}

unsigned int vtkDataBox::NumberOfDifferentialChannels() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_MAXDICHANS,&value));
  return value; 
}

bool vtkDataBox::CheckCapabilitiesFilteringPerChannel() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_FILTERPERCHAN,&value));
  if (value == 0) { return true; } else {  return false; }
}

unsigned int vtkDataBox::NumberOfFilterSelections() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_NUMFILTERS,&value));
  return value; 
}

bool vtkDataBox::CheckCapabilitiesRangePerChannel() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_RANGEPERCHANNEL,&value));
  if (value == 0) { return true; } else {  return false; }
}

unsigned int vtkDataBox::NumberOfRangeSelections() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_NUMRANGES,&value));
  return value; 
}

bool vtkDataBox::CheckCapabilitiesProgrammableResolution() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SWRESOLUTION,&value));
  if (value == 0) { return true; } else {  return false; }
}

unsigned int vtkDataBox::NumberOfProgrammableResolutions() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_NUMRESOLUTIONS,&value));
  return value; 
}

bool vtkDataBox::CheckCapabilitiesBinaryEncoding() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_BINARY,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesTwosComplementEncoding() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_2SCOMP,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesInternalSoftwareTrigger() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SOFTTRIG,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesExternalDigitalTrigger() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_EXTERNTRIG,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesPositiveAnalogTrigger() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_THRESHTRIGPOS, &value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesNegativeAnalogTrigger() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_THRESHTRIGNEG,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesAnalogEventTrigger() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_ANALOGEVENTTRIG,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesDigitalEventTrigger() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_DIGITALEVENTTRIG,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesTimerEventTrigger() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_TIMEREVENTTRIG,&value));
  if (value == 0) { return true; } else {  return false; }
}

unsigned int vtkDataBox::NumberOfExtraTriggerSources() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_NUMEXTRATRIGGERS,&value));
  return value; 
}

bool vtkDataBox::CheckCapabilitiesInternalClock() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_INTCLOCK,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesExternalClock() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_EXTCLOCK,&value));
  if (value == 0) { return true; } else {  return false; }
}

unsigned int vtkDataBox::NumberOfExtraClockSources() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_NUMEXTRACLOCKS,&value));
  return value; 
}


/////////////////////////////////////////////////////////////////

bool vtkDataBox::CheckCapabilitiesCascading() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CASCADING,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesEventCounting() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CTMODE_COUNT,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesRateGeneration() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CTMODE_RATE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesOneShotMode() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CTMODE_ONESHOT,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesRepetitiveOneShotMode() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CTMODE_ONESHOT_RPT,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesHigh2LowPulse() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_PLS_HIGH2LOW,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesLow2HighPulse() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_PLS_LOW2HIGH,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesInternalGate() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_NONE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesHighLevelGate() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_HIGH_LEVEL,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesLowLevelGate() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_LOW_LEVEL,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesHighEdgeGate() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_HIGH_EDGE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesLowEdgeGate() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_LOW_EDGE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesLevelChangeGate() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_LEVEL,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesHighLevelGateWithDebounce() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_HIGH_LEVEL_DEBOUNCE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesLowLevelGateWithDebounce() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_LOW_LEVEL_DEBOUNCE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesHighEdgeGateWithDebounce() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_HIGH_EDGE_DEBOUNCE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesLowEdgeGateWithDebounce() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_LOW_EDGE_DEBOUNCE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesLevelChangeGateWithDebounce() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_LEVEL_DEBOUNCE,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesIOInterrupt() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_INTERRUPT,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesFIFO() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_FIFO,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesProcessorOnBoard() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_PROCESSOR,&value));
  if (value == 0) { return true; } else {  return false; }
}

bool vtkDataBox::CheckCapabilitiesSoftwareCalibration() {
  unsigned int value;
  this->CheckError(olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SWCAL,&value));
  if (value == 0) { return true; } else {  return false; }
}

void vtkDataBox::GetInternalTimeStamp(HBUF buffer, LPTDS timestamp) {
  this->CheckError(olDmGetTimeDateStamp(buffer, timestamp));
}


// buffer: timing accuracy only 1 second: cannot be used for our fast application  
unsigned long vtkDataBox::GetInternalTimeStamp(HBUF buffer) {
  struct _timeb timeptr;
  this->internalTime = 0;
  
  if (this->CheckError(olDmGetTimeDateStamp(buffer, this->internalTimestamp)))
  {
    _ftime(&timeptr);
    //Sleep(1);
    this->internalTime = (  timeptr.millitm+
                this->internalTimestamp->wSec *1000+
                this->internalTimestamp->wMin *100000+
                this->internalTimestamp->wHour*10000000);

    return this->internalTime;
  }
  else return 0;
}
  
double vtkDataBox::GetChannelsCurrentValue(int channel){
  return 0;
}


/*
ECODE WINAPI olDmGetBufferPtr (HBUF, LPVOID FAR*);
ECODE WINAPI olDmGetBufferECode (HBUF, LPECODE);
ECODE WINAPI olDmGetBufferSize (HBUF, DWORD FAR*);
ECODE WINAPI olDmGetVersion (LPSTR);
ECODE WINAPI olDmCopyBuffer (HBUF, LPVOID);
ECODE WINAPI olDmCopyToBuffer (HBUF hBuf, LPVOID lpAppBuffer, ULNG ulNumSamples);

ECODE WINAPI olDmLockBuffer (HBUF);
ECODE WINAPI olDmUnlockBuffer (HBUF);
ECODE WINAPI olDmReCallocBuffer (UINT, UINT, DWORD, UINT, LPHBUF);
ECODE WINAPI olDmReMallocBuffer (UINT, UINT, DWORD, LPHBUF);
ECODE WINAPI olDmGetDataBits (HBUF, UINT FAR*);
ECODE WINAPI olDmSetDataWidth (HBUF, UINT);
ECODE WINAPI olDmGetDataWidth (HBUF, UINT FAR*);
ECODE WINAPI olDmGetMaxSamples (HBUF, DWORD FAR*);
ECODE WINAPI olDmSetValidSamples (HBUF, DWORD);
ECODE WINAPI olDmGetValidSamples (HBUF, DWORD FAR*);
ECODE WINAPI olDmCopyFromBuffer(HBUF hBuf, LPVOID lpAppBuffer, ULNG ulMaxSamples);
   
ECODE WINAPI olDmGetExtraBytes (HBUF hBuf, ULNG FAR *lpulExtra1,ULNG FAR *lpulExtra2);
ECODE WINAPI olDmSetExtraBytes (HBUF hBuf, ULNG ulExtra1, ULNG ulExtra2);

ECODE WINAPI olDmLockBufferEx (HBUF hBuf, BOOL fEnableScatter);
ECODE WINAPI olDmUnlockBufferEx (HBUF hBuf, BOOL fEnableScatter);

ECODE WINAPI drvDmSetCurrentTDS (HBUF hBuf);
ECODE WINAPI drvDmSetDataBits (HBUF hBuf, UINT Bits);
*/

void  vtkDataBox::GetSingleValue(long * value, unsigned int channel, double gain=0) {
  if (this->deviceStarted ==0  && this->deviceSingle ==1) {
    this->ConfigOperations();
    this->StartOperations();
    this->deviceStarted=1;
  }
  if (gain == 0) {
    gain = this->Gain;
  }
  this->CheckError(olDaGetSingleValue(this->board.hdass, value, channel, gain));
}

long  vtkDataBox::GetSingleValue(unsigned int channel, double gain=0) {
  if (this->deviceStarted ==0  && this->deviceSingle ==1) {
    this->ConfigOperations();
    this->StartOperations();
    this->deviceStarted=1;
  }
  if (gain == 0) {
    gain = this->Gain;
  }
  long value;
  this->CheckError(olDaGetSingleValue(this->board.hdass, &value, channel, gain));
  return value;
}

void  vtkDataBox::PutSingleValue(long value, unsigned int channel, double gain) {
    if (gain == 0) {
    gain = this->Gain;
  }
  this->CheckError(olDaPutSingleValue(this->board.hdass, value, channel, gain));
}

int vtkDataBox::Probe() {
  if (deviceOpen==0) {
    CheckError(olDaEnumBoards(GetDriver,(LPARAM)(LPBOARD)&this->board));
    return TRUE;
  }
  else {
    cout << "Device already open.  No need to probe it" << endl;
    return FALSE;
  }
}


void vtkDataBox::GetDeviceHandle(unsigned int numberADs){
  this->CheckError(olDaGetDevCaps(this->board.hdrvr,OLDC_ADELEMENTS,&numberADs));
}

int vtkDataBox::AllocateSubSystem(unsigned int currentAD) {
  return olDaGetDASS(this->board.hdrvr,OLSS_AD,currentAD,&this->board.hdass);
}

int vtkDataBox::OpenBox() 
{
   unsigned long size;
   unsigned int gainSupported,samplesize;
   int i; //?
   int eCode; //?

   if (deviceOpen==1) {
     cout << "Board Already Open" << endl;
     return 0;
   }
    
   unsigned int channel = 0;
   unsigned int numberADs = 0;
   unsigned int currentAD = 0;
     
   ECODE ec=OLNOERROR;
   this->board.hdrvr = NULL;

   /* Get first available Open Layers board */
   this->Probe();
   this->CheckError(this->board.status);

   /* check for NULL driver handle - means no boards */
   if (this->board.hdrvr == NULL){
    cout << " No Open Layer boards!!!" << endl;
    return 0;
   }

   /* get handle to first available ADC sub system */
   this->GetDeviceHandle(numberADs);
   while(1)    // Enumerate through all the A/D subsystems and try and select the first available one...
   {
        ec=this->AllocateSubSystem(currentAD);
        if (ec!=OLNOERROR){   // busy subsys etc...
            currentAD++;
            if (currentAD>numberADs){
               cout << "No Available A/D Subsystems!" << endl;
               return 0;
            }
        }
        else
           break;
   }

   // Setup Flow (step 3)
   if (this->deviceSingle)
     this->SetFlowToSingleValue();
   else
     this->SetFlowToContinuous();

   // Setup DMA (step 4)
   this->CheckError(olDaGetSSCapsEx(this->board.hdass,OLSSCE_MAXTHROUGHPUT,&this->Freq));
   this->DMA = this->CheckNumberOfDMAChannels();  // no DMA for DT9804ECI
   gainSupported = this->CheckCapabilitiesProgrammableGain();

   this->DMA  = min (1, DMA);            // try for one dma channel 
   this->SetNumberOfDMAChannels(this->DMA);

   // Setup Subsystem Parameters (page 131) (step 5)

   // setup channel Type (5.1)
   //===========================================================================
   this->SetChannelTypeToSingle();
   //this->SetChannelTypeToDifferential();
   //===========================================================================

   // setup resolution (5.2)
   samplesize = this->GetResolution();
   if (samplesize > 16)
       samplesize=4; //4 bytes...// e.g. 24 bits = 4 btyes
   else
       samplesize=2;             // e.g. 16 or 12 bits = 2 bytes

  // setup encoding (5.3) (optional)
  this->SetEncodingToBinary();

    // setup range (5.4) (optional)
  double max=0, min=0;
  this->GetRange(&max,&min);

  // setup channel filter (5.5) (optional)

  // setup channel List (page 132) (step 6)


   // Setup Channel List Size (6.1)
  this->SetChannelListSize(this->numCurrentChannels);
  if (this->numCurrentChannels > MAXCHANNELS) {
       this->numCurrentChannels = MAXCHANNELS;
  }

   // Setup channel List Entries
   for (i=0; i <this->numCurrentChannels;i++)
     this->SetChannelListEntry(i,i);


   // Adjustable parameters
   //===========================================================================
   // try for 1000hz throughput
   size = (unsigned long)this->numCurrentChannels*this->bufferSize;
   //===========================================================================

   // Setup Gain List Entries (6.3) (Optional)
   for (i=0; i <this->numCurrentChannels;i++)
      this->CheckError( olDaSetGainListEntry (this->board.hdass, i, this->Gain) );
   // Setup Channel List Entry Inhibit (6.4) (Optional)

   // Setup Synchronous Digitial IO Usage (6.5) (Optional)

   // Setup Digital IO List Entry (6.6) (Optional)

   // setup clocks, triggers and pre-triggers
   this->SetClockSource(OL_CLK_INTERNAL); //***
   this->SetClockFrequency(this->Freq);

   this->CheckError( olDaSetTriggeredScanUsage( this->board.hdass, true ));
   this->CheckError( olDaSetMultiscanCount( this->board.hdass, 1 ) );
   this->CheckError( olDaSetRetriggerMode( this->board.hdass, OL_RETRIGGER_INTERNAL ) );
   this->CheckError( olDaSetRetriggerFrequency( this->board.hdass, retriggerFrequency ) );

   
   for (i=0; i<1; i++){
      this->CheckError(olDaSetChannelFilter(this->board.hdass,i,this->CutOffFreqFilter));
   }

   // set up buffering
   this->SetBufferWrapModeToMultiple();
   this->CreateBuffers(size, samplesize);
   this->ConfigureNotificationProcedure();

   // configure system
   this->ConfigOperations();

   // Get current configured parameters
   double dCutOffFreqFilterGet=0;
   for (i=0; i<1; i++){
      eCode = olDaGetChannelFilter(this->board.hdass,i,&dCutOffFreqFilterGet);
    this->CheckError(eCode);
   }

   deviceOpen=1;

   return 1;
}

vtkDataBoxBuffer * vtkDataBox::GetBuffer() {
  return this->data;
}

int vtkDataBox::OpenBoxOrg() 
{

   unsigned long size;
   unsigned int gainSupported,samplesize;
   int i;

   if (deviceOpen==1) {
     cout << "Board Already Open" << endl;
     return 0;
   }
   
   unsigned int channel = 0;
   unsigned int numberADs = 0;
   unsigned int currentAD = 0;
     
   ECODE ec=OLNOERROR;

   this->board.hdrvr = NULL;

   /* Get first available Open Layers board */
   this->Probe();
   this->CheckError(this->board.status);

   /* check for NULL driver handle - means no boards */
   if (this->board.hdrvr == NULL){
    cout << " No Open Layer boards!!!" << endl;
      return 0;
   }

   /* get handle to first available ADC sub system */
   this->GetDeviceHandle(numberADs);
   while(1)    // Enumerate through all the A/D subsystems and try and select the first available one...
   {
        ec=this->AllocateSubSystem(currentAD);
        if (ec!=OLNOERROR)
        {
            // busy subsys etc...
            currentAD++;
            if (currentAD>numberADs)
      {
         cout << "No Available A/D Subsystems!" << endl;
               return 0;
      }
        }
        else
           break;

   }

   // Setup Flow (step 3)
   if (this->deviceSingle)
     this->SetFlowToSingleValue();
   else
     this->SetFlowToContinuous();

   // Setup DMA (step 4)

   this->CheckError(olDaGetSSCapsEx(this->board.hdass,OLSSCE_MAXTHROUGHPUT,&this->Freq));
   this->DMA = this->CheckNumberOfDMAChannels();
   gainSupported = this->CheckCapabilitiesProgrammableGain();
   
   this->DMA  = min (1, DMA);            // try for one dma channel 
   //this->Freq = min (96000.0, this->Freq);      // try for 1000hz throughput

   this->SetNumberOfDMAChannels(this->DMA);
   this->SetClockFrequency(this->Freq);
   // Setup Subsystem Parameters (page 131) (step 5)

   // setup channel Type (5.1)
   this->SetChannelTypeToSingle();
  //this->SetChannelTypeToDifferential();

   // setup resolution (5.2)
   samplesize = this->GetResolution();
   if (samplesize > 16)
       samplesize=4; //4 bytes...// e.g. 24 bits = 4 btyes
   else
       samplesize=2;             // e.g. 16 or 12 bits = 2 bytes

  // setup encoding (5.3) (optional)
  this->SetEncodingToBinary();

    // setup range (5.4) (optional)
  double max=0, min=0;
  this->GetRange(&max,&min);
  cout << "MAX: "<< max << "\tMIN: " << min << endl;

  // setup channel filter (5.5) (optional)


   // setup channel List (page 132) (step 6)

   // Setup Channel List Size (6.1)
   this->SetChannelListSize((int)this->NumberOfSingleEndedChannels());
   int numChannels = (int)this->NumberOfSingleEndedChannels();
   if (numChannels > MAXCHANNELS) {
       numChannels = MAXCHANNELS;
   }

   // Setup channel List Entries
   for (i=0; i <numChannels;i++)
     this->SetChannelListEntry(i,i);
   size = (unsigned long)numChannels*3;     /* 1 frame per value */

   // Setup Gain List Entries (6.3) (Optional)
   for (i=0; i <numChannels;i++)
     this->SetGainListEntry(i,i);

   // Setup Channel List Entry Inhibit (6.4) (Optional)

   // Setup Synchronous Digitial IO Usage (6.5) (Optional)

   // Setup Digital IO List Entry (6.6) (Optional)

   // setup clocks, triggers and pre-triggers

   // set up buffering
   this->SetBufferWrapModeToMultiple();
   this->CreateBuffers(size, samplesize);
   this->ConfigureNotificationProcedure();

   // configure system

   this->ConfigOperations();

   deviceOpen=1;
   return 1;
}
 
void vtkDataBox::StopBox() {
  if (deviceStarted ==1) {
    deviceStarted=0;
    this->StopOperations();
  }
}


int vtkDataBox::CloseBox() {
   this->StopBox();
   this->FreeBuffers();
   
   /* release the subsystem and the board */

   this->CheckError(olDaReleaseDASS(this->board.hdass));
   this->CheckError(olDaTerminate(this->board.hdrvr));

   /* all done - return */
   deviceOpen=0;
   return 1;

}

void vtkDataBox::SetGain(double gain) {
   this->Gain = gain;
}

double vtkDataBox::GetGain() {
  return this->Gain;
}

void vtkDataBox::SetDMA(unsigned int dma){
  if (dma <= this->CheckNumberOfDMAChannels()) {
    this->DMA = dma;
  }
  else {
    cout << "Invalid DMA size.  Must be less than" <<  this->CheckNumberOfDMAChannels() << endl;
  }
}
unsigned int vtkDataBox::GetDMA() {
  return this->DMA;
}

void vtkDataBox::SetFrequency(double freq) {
  this->Freq = freq;
}

double vtkDataBox::GetFrequency() {
  return this->Freq;
}

void vtkDataBox::SetBufferLength(double bufferLength) {
  this->BufferLength = bufferLength;
}

double vtkDataBox::GetBufferLength() {
  return this->BufferLength;
}

void vtkDataBox::StartBox() {
  if (this->deviceStarted ==0) {
    this->ConfigOperations();
    this->StartOperations();
    if (this->board.status ==NOERROR) {
       this->deviceStarted=1;
    }
  }
}

void vtkDataBox::CodeToVolts(unsigned int resolution, unsigned int encoding, long value, double * voltage) {
  double max,min;
  double gain;
  this->GetRange(&max,&min);
  gain = this->GetGain();
  this->CheckError(olDaCodeToVolts(min,max, gain, resolution, encoding, value, voltage));
}

/*
void vtkDataBox::StartBoxBuffer() {
  if (bufferStarted ==0) {
    bufferStarted=1;
  }
}

void vtkDataBox::StopBoxBuffer() {
  if (StartBoxBuffer ==1) {
    bufferStarted=0;
  }
}
*/

int vtkDataBox::PrintCapabilitiesAll() 
{
  unsigned int value, result;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSS_SUP_RETRIGGER_INTERNAL,&value) );
  cout << "OLSS_SUP_RETRIGGER_INTERNAL: " << value << ", result: " << result << endl;
  cout << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_MAXSECHANS,&value) );
  cout << "OLSSC_MAXSECHANS: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_MAXDICHANS,&value) );
  cout << "OLSSC_MAXDICHANS: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_CGLDEPTH,&value) );
  cout << "OLSSC_CGLDEPTH: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_NUMFILTERS,&value) );
  cout << "OLSSC_NUMFILTERS: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_NUMGAINS,&value) );
  cout << "OLSSC_NUMGAINS: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_NUMRANGES,&value) );
  cout << "OLSSC_NUMRANGES: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_NUMDMACHANS,&value) );
  cout << "OLSSC_NUMDMACHANS: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_NUMCHANNELS,&value) );
  cout << "OLSSC_NUMCHANNELS: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_NUMEXTRACLOCKS,&value) );
  cout << "OLSSC_NUMEXTRACLOCKS: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_NUMEXTRATRIGGERS,&value) );
  cout << "OLSSC_NUMEXTRATRIGGERS: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_NUMRESOLUTIONS,&value) );
  cout << "OLSSC_NUMRESOLUTIONS: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_INTERRUPT,&value) );
  cout << "OLSSC_SUP_INTERRUPT: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SINGLEENDED,&value) );
  cout << "OLSSC_SUP_SINGLEENDED: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_DIFFERENTIAL,&value) );
  cout << "OLSSC_SUP_DIFFERENTIAL: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_BINARY,&value) );
  cout << "OLSSC_SUP_BINARY: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_2SCOMP,&value) );
  cout << "OLSSC_SUP_2SCOMP: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SOFTTRIG,&value) );
  cout << "OLSSC_SUP_SOFTTRIG: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_EXTERNTRIG,&value) );
  cout << "OLSSC_SUP_EXTERNTRIG: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_THRESHTRIGPOS,&value) );
  cout << "OLSSC_SUP_THRESHTRIGPOS: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_THRESHTRIGNEG,&value) );
  cout << "OLSSC_SUP_THRESHTRIGNEG: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_ANALOGEVENTTRIG,&value) );
  cout << "OLSSC_SUP_ANALOGEVENTTRIG: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_DIGITALEVENTTRIG,&value) );
  cout << "OLSSC_SUP_DIGITALEVENTTRIG: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_TIMEREVENTTRIG,&value) );
  cout << "OLSSC_SUP_TIMEREVENTTRIG: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_TRIGSCAN,&value) );
  cout << "OLSSC_SUP_TRIGSCAN: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_INTCLOCK,&value) );
  cout << "OLSSC_SUP_INTCLOCK: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_EXTCLOCK,&value) );
  cout << "OLSSC_SUP_EXTCLOCK: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SWCAL,&value) );
  cout << "OLSSC_SUP_SWCAL: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_EXP2896,&value) );
  cout << "OLSSC_SUP_EXP2896: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_EXP727,&value) );
  cout << "OLSSC_SUP_EXP727: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_FILTERPERCHAN,&value) );
  cout << "OLSSC_SUP_FILTERPERCHAN: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_DTCONNECT,&value) );
  cout << "OLSSC_SUP_DTCONNECT: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_FIFO,&value) );
  cout << "OLSSC_SUP_FIFO: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_PROGRAMGAIN,&value) );
  cout << "OLSSC_SUP_PROGRAMGAIN: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_PROCESSOR,&value) );
  cout << "OLSSC_SUP_PROCESSOR: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SWRESOLUTION,&value) );
  cout << "OLSSC_SUP_SWRESOLUTION: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CONTINUOUS,&value) );
  cout << "OLSSC_SUP_CONTINUOUS: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SINGLEVALUE,&value) );
  cout << "OLSSC_SUP_SINGLEVALUE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_PAUSE,&value) );
  cout << "OLSSC_SUP_PAUSE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_WRPMULTIPLE,&value) );
  cout << "OLSSC_SUP_WRPMULTIPLE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_WRPSINGLE,&value) );
  cout << "OLSSC_SUP_WRPSINGLE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_POSTMESSAGE,&value) );
  cout << "OLSSC_SUP_POSTMESSAGE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CASCADING,&value) );
  cout << "OLSSC_SUP_CASCADING: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CTMODE_COUNT,&value) );
  cout << "OLSSC_SUP_CTMODE_COUNT: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CTMODE_RATE,&value) );
  cout << "OLSSC_SUP_CTMODE_RATE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CTMODE_ONESHOT,&value) );
  cout << "OLSSC_SUP_CTMODE_ONESHOT: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CTMODE_ONESHOT_RPT,&value) );
  cout << "OLSSC_SUP_CTMODE_ONESHOT_RPT: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_MAX_DIGITALIOLIST_VALUE,&value) );
  cout << "OLSSC_MAX_DIGITALIOLIST_VALUE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_DTCONNECT_CONTINUOUS,&value) );
  cout << "OLSSC_SUP_DTCONNECT_CONTINUOUS: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_DTCONNECT_BURST,&value) );
  cout << "OLSSC_SUP_DTCONNECT_BURST: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CHANNELLIST_INHIBIT,&value) );
  cout << "OLSSC_SUP_CHANNELLIST_INHIBIT: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SYNCHRONOUS_DIGITALIO,&value) );
  cout << "OLSSC_SUP_SYNCHRONOUS_DIGITALIO: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SIMULTANEOUS_START,&value) );
  cout << "OLSSC_SUP_SIMULTANEOUS_START: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_INPROCESSFLUSH,&value) );
  cout << "OLSSC_SUP_INPROCESSFLUSH: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_RANGEPERCHANNEL,&value) );
  cout << "OLSSC_SUP_RANGEPERCHANNEL: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SIMULTANEOUS_SH,&value) );
  cout << "OLSSC_SUP_SIMULTANEOUS_SH: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_RANDOM_CGL,&value) );
  cout << "OLSSC_SUP_RANDOM_CGL: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_SEQUENTIAL_CGL,&value) );
  cout << "OLSSC_SUP_SEQUENTIAL_CGL: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_ZEROSEQUENTIAL_CGL,&value) );
  cout << "OLSSC_SUP_ZEROSEQUENTIAL_CGL: " << value << ", result: " << result << endl;
  
  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GAPFREE_NODMA,&value) );
  cout << "OLSSC_SUP_GAPFREE_NODMA: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GAPFREE_SINGLEDMA,&value) );
  cout << "OLSSC_SUP_GAPFREE_SINGLEDMA: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GAPFREE_DUALDMA,&value) );
  cout << "OLSSC_SUP_GAPFREE_DUALDMA: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSCE_MAXTHROUGHPUT,&value) );
  cout << "OLSSCE_MAXTHROUGHPUT: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSCE_MINTHROUGHPUT,&value) );
  cout << "OLSSCE_MINTHROUGHPUT: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSCE_MAXRETRIGGER,&value) );
  cout << "OLSSCE_MAXRETRIGGER: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSCE_MINRETRIGGER,&value) );
  cout << "OLSSCE_MINRETRIGGER: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSCE_MAXCLOCKDIVIDER,&value) );
  cout << "OLSSCE_MAXCLOCKDIVIDER: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSCE_MINCLOCKDIVIDER,&value) );
  cout << "OLSSCE_MINCLOCKDIVIDER: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSCE_BASECLOCK,&value) );
  cout << "OLSSCE_BASECLOCK: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSCE_RANGELOW,&value) );
  cout << "OLSSCE_RANGELOW: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSCE_RANGEHIGH,&value) );
  cout << "OLSSCE_RANGEHIGH: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSCE_FILTER,&value) );
  cout << "OLSSCE_FILTER: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSCE_GAIN,&value) );
  cout << "OLSSCE_GAIN: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSCE_RESOLUTION,&value) );
  cout << "OLSSCE_RESOLUTION: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_PLS_HIGH2LOW,&value) );
  cout << "OLSSC_SUP_PLS_HIGH2LOW: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_PLS_LOW2HIGH,&value) );
  cout << "OLSSC_SUP_PLS_LOW2HIGH: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_NONE,&value) );
  cout << "OLSSC_SUP_GATE_NONE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_HIGH_LEVEL,&value) );
  cout << "OLSSC_SUP_GATE_HIGH_LEVEL: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_LOW_LEVEL,&value) );
  cout << "OLSSC_SUP_GATE_LOW_LEVEL: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_HIGH_EDGE,&value) );
  cout << "OLSSC_SUP_GATE_HIGH_EDGE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_LOW_EDGE,&value) );
  cout << "OLSSC_SUP_GATE_LOW_EDGE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_LEVEL,&value) );
  cout << "OLSSC_SUP_GATE_LEVEL: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_HIGH_LEVEL_DEBOUNCE,&value) );
  cout << "OLSSC_SUP_GATE_HIGH_LEVEL_DEBOUNCE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_LOW_LEVEL_DEBOUNCE,&value) );
  cout << "OLSSC_SUP_GATE_LOW_LEVEL_DEBOUNCE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_HIGH_EDGE_DEBOUNCE,&value) );
  cout << "OLSSC_SUP_GATE_HIGH_EDGE_DEBOUNCE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_LOW_EDGE_DEBOUNCE,&value) );
  cout << "OLSSC_SUP_GATE_LOW_EDGE_DEBOUNCE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_GATE_LEVEL_DEBOUNCE,&value) );
  cout << "OLSSC_SUP_GATE_LEVEL_DEBOUNCE: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSS_SUP_RETRIGGER_INTERNAL,&value) );
  cout << "OLSS_SUP_RETRIGGER_INTERNAL: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSS_SUP_RETRIGGER_SCAN_PER_TRIGGER,&value) );
  cout << "OLSS_SUP_RETRIGGER_SCAN_PER_TRIGGER: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_MAXMULTISCAN,&value) );
  cout << "OLSSC_MAXMULTISCAN: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CONTINUOUS_PRETRIG,&value) );
  cout << "OLSSC_SUP_CONTINUOUS_PRETRIG: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_CONTINUOUS_ABOUTTRIG,&value) );
  cout << "OLSSC_SUP_CONTINUOUS_ABOUTTRIG: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_BUFFERING,&value) );
  cout << "OLSSC_SUP_BUFFERING: " << value << ", result: " << result << endl;

  result = this->CheckError( olDaGetSSCaps(this->board.hdass,OLSSC_SUP_RETRIGGER_EXTRA,&value) );
  cout << "OLSSC_SUP_RETRIGGER_EXTRA: " << value << ", result: " << result << endl;
/*
  long valueL = -10;
  result = this->CheckError( olDaEnumSSCaps(this->board.hdass, OL_ENUM_FILTERS, AddFilterList, (LPARAM)valueL) );
  cout << "OL_ENUM_FILTERS: " << valueL << ", result: " << result << endl;

  valueL = -2;
  result = this->CheckError( olDaEnumSSCaps(this->board.hdass, OL_ENUM_GAINS, AddGainList, (LPARAM)valueL) );
  cout << "OL_ENUM_GAINS: " << valueL << ", result: " << result << endl;
  //======================================================================================
*/
  return 0;
}