/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkDirectShowVideoSource.cxx

  Copyright (c) John Baxter, Robarts Research Institute
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkDirectShowVideoSource.h"
#include "vtkDirectShowHelperMethods.h"

#include "vtkObjectFactory.h"
#include "vtkDataArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkCriticalSection.h"
#include "vtkTimerLog.h"
#include "vtkMultiThreader.h"

#include "windows.h"
#include "dshow.h"
#include "strsafe.h"
#include <iostream>

//TODO: Find some way to remove the absolute file path
#pragma include_alias( "dxtrans.h", "qedit.h" )
#define __IDxtCompositor_INTERFACE_DEFINED__
#define __IDxtAlphaSetter_INTERFACE_DEFINED__
#define __IDxtJpeg_INTERFACE_DEFINED__
#define __IDxtKey_INTERFACE_DEFINED__
#include "qedit.h"

#include <vtkstd/string> 
#include <ctype.h>

// because of warnings in windows header push and pop the warning level
#ifdef _MSC_VER
#pragma warning (push, 3)
#endif

#include "vtkWindows.h"
#include <winuser.h>
#include <vfw.h>

#ifdef _MSC_VER
#pragma warning (pop)
#endif

class vtkDirectShowVideoSourceInternal : public ISampleGrabberCB
{
public:
	vtkDirectShowVideoSourceInternal(vtkDirectShowVideoSource* s) {
		parent = s;

		//prepare the internal classes
		this->pGraph = 0;
		this->pBuilder = 0;
		this->pSourceFilter = 0;
		this->pEnum = 0;
	
		//prepare the media control and event classes
		this->pMediaControl = 0;
		this->pMediaEvent = 0;

		//prepare the buffer grabbing classes
		this->pNullRenderer = 0;
		this->pGrabber = 0;
		this->pGrabberFilter = 0;

		//prepare the media type to a capturable default
		ZeroMemory( &(this->mediaType), sizeof(this->mediaType) );
		this->mediaType.majortype = MEDIATYPE_Video;
		this->mediaType.subtype = MEDIASUBTYPE_RGB24;
		this->majorType = MEDIATYPE_Video;
		
		lastTime = -100;
	}

	vtkDirectShowVideoSource*	parent;

	IGraphBuilder*			pGraph;
	ICaptureGraphBuilder2*	pBuilder;
	IBaseFilter*			pSourceFilter;
	IEnumPins*				pEnum;

	IMediaControl*			pMediaControl;
	IMediaEventEx*			pMediaEvent;

	IBaseFilter*			pNullRenderer;
	ISampleGrabber*			pGrabber;
	IBaseFilter*			pGrabberFilter;
	
	AM_MEDIA_TYPE			mediaType;
	GUID					majorType;

	double					lastTime;

	std::vector<IBaseFilter*>	deviceArray;

	// Fake referance counting methods.
	STDMETHODIMP_(ULONG) AddRef() { return 1; }
	STDMETHODIMP_(ULONG) Release() { return 2; }

	//no clue what this method does...
    STDMETHODIMP QueryInterface(REFIID riid, void **ppvObject)
    {
        if (NULL == ppvObject) return E_POINTER;
        if (riid == __uuidof(IUnknown))
        {
            *ppvObject = static_cast<IUnknown*>(this);
             return S_OK;
        }
        if (riid == __uuidof(ISampleGrabberCB))
        {
            *ppvObject = static_cast<ISampleGrabberCB*>(this);
             return S_OK;
        }
        return E_NOTIMPL;
    }

	//
    STDMETHODIMP SampleCB(double Time, IMediaSample *pSample)
    {
        return E_NOTIMPL;
    }

	//copies the buffer into the output image and raises the modified flag
    STDMETHODIMP BufferCB(double Time, BYTE *pBuffer, long BufferLen)
    {

		if( (Time - lastTime) * this->parent->FrameRate < 0.95 ) return S_OK;
		if( !this->parent->Recording ) return S_OK;
		lastTime = Time;

		//switch from BGR to RGB
		char* switchPtr = (char*) pBuffer;
		for( long i = 0; i < BufferLen; i+=3 ){
			char red = *switchPtr;
			*(switchPtr) = *(switchPtr+2);
			*(switchPtr+2) = red;
			switchPtr += 3;
		}

		//copy over buffers and mark the video source as modified
		this->parent->medialBufferMutex->Lock();
		memcpy((void*) this->parent->output, (void*) pBuffer, (size_t) BufferLen);
		this->parent->medialBufferMutex->Unlock();

		this->parent->Modified();

        return S_OK;    
    }


};

vtkStandardNewMacro(vtkDirectShowVideoSource);

//----------------------------------------------------------------------------
vtkDirectShowVideoSource::vtkDirectShowVideoSource()
{
	
	//set up the internal class (handles pretty much everything)
	this->Internal = new vtkDirectShowVideoSourceInternal(this);

	//set some reasonable default values
	this->vtkVideoSource::SetFrameRate( 5.0f );
	this->videoSourceNumber = 0;
	this->FrameBufferBitsPerPixel = 24;
	this->vtkVideoSource::SetOutputFormat(VTK_RGB);
	this->vtkVideoSource::SetFrameBufferSize( 100 );

	//set the output image to a null buffer
	this->output = 0;
	this->medialBufferMutex = vtkMutexLock::New();


	this->Initialized = 0;
	
	this->EnumerateVideoSources();
}

//----------------------------------------------------------------------------
vtkDirectShowVideoSource::~vtkDirectShowVideoSource()
{
  this->vtkDirectShowVideoSource::ReleaseSystemResources();
  delete this->Internal;
}  

//----------------------------------------------------------------------------
void vtkDirectShowVideoSource::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
void vtkDirectShowVideoSource::SetVideoSourceNumber(unsigned int n){
	
	if( n >= this->devices.size() ){
		vtkErrorMacro(<<"Cannot use that video source number. Must be in the array.");
		return;
	}

	//if we haven't already initialized, we will use this dialog anyway
	if( n != this->videoSourceNumber && this->Initialized ){
		this->videoSourceNumber = n;
		this->ReleaseSystemResources();
		this->Initialize();
	}
	this->videoSourceNumber = n;

}

unsigned int vtkDirectShowVideoSource::GetVideoSourceNumber(){
	return this->videoSourceNumber;
}

//----------------------------------------------------------------------------
void vtkDirectShowVideoSource::SetFrameRate(float rate){
	this->vtkVideoSource::SetFrameRate(rate);
	//vtkErrorMacro(<<"Use the VideoFormatDialog() method to set the frame rate.");
}

//----------------------------------------------------------------------------
void vtkDirectShowVideoSource::SetFrameSize(int x, int y, int z) {
	vtkErrorMacro(<<"Size is determined by device and cannot be set.");
}

void vtkDirectShowVideoSource::SetOutputFormat(int format) {
	vtkErrorMacro(<<"Use the VideoFormatDialog() method to set the output format.");
}

//----------------------------------------------------------------------------
void vtkDirectShowVideoSource::VideoFormatDialog(){

	//TODO if warranted

}

//----------------------------------------------------------------------------
void vtkDirectShowVideoSource::Initialize()
{
	// if we are already initialized, do not initialize again
	if (this->Initialized){
		return;
	}

	HRESULT hr = S_OK;
	HRESULT hrAdd = S_OK;

	//initializes the COM library which DirectShow runs partially on top of
	hr = CoInitialize(NULL);
	if (FAILED(hr)){
		vtkErrorMacro(<<"The Microsoft COM library could not be initialized.");
		return;
	}

	//create the GraphBuilder (other name is Filter Graph Manager)
	//this will oversee the creation and management of the filters making up the DShow pipeline
		// CLSID_FilterGraph is the class identifier (tells what class the void** is)
		// CLSCTX_INPROC_SERVER is the communication type, which is just interprocess (as opposed to network)
	hr = CoCreateInstance(CLSID_FilterGraph, NULL, CLSCTX_INPROC_SERVER, IID_IGraphBuilder, (void **) &(this->Internal->pGraph));
	if (FAILED(hr)){
		vtkErrorMacro(<<"Could not create filter graph manager.");
		this->ReleaseSystemResources();
		return;
	}

	//query the graph manager for interfaces enabling video capture events
	hr = this->Internal->pGraph->QueryInterface( IID_PPV_ARGS(&(this->Internal->pMediaControl)) );
	if (FAILED(hr)){
		vtkErrorMacro(<<"Unsucessful query for media control interfaces.");
		this->ReleaseSystemResources();
		return;
	}
	hr = this->Internal->pGraph->QueryInterface( IID_PPV_ARGS(&this->Internal->pMediaEvent) );
	if (FAILED(hr)){
		vtkErrorMacro(<<"Unsucessful query for media event interfaces.");
		this->ReleaseSystemResources();
		return;
	}

	//create the Capture Graph Builder which manages the pipeline for actual video capture
	hr = CoCreateInstance(CLSID_CaptureGraphBuilder2, NULL, CLSCTX_INPROC_SERVER, IID_ICaptureGraphBuilder2, (void **) &(this->Internal->pBuilder));
	if (FAILED(hr)){
		vtkErrorMacro(<<"Could not create capture graph manager.");
		this->ReleaseSystemResources();
		return;
	}
	
	//Set the filter graph used by the capturing mechanism
	hr = this->Internal->pBuilder->SetFiltergraph( this->Internal->pGraph );
	if (FAILED(hr)){
		vtkErrorMacro(<<"Could not link graph to capture builder.");
		this->ReleaseSystemResources();
		return;
	}

	//create the sample grabber and it's filter
	hr = CoCreateInstance(CLSID_SampleGrabber, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS( &(this->Internal->pGrabberFilter) ));
    if (FAILED(hr)){
		vtkErrorMacro(<<"Could not create sample grabber filter.");
		this->ReleaseSystemResources();
		return;
    }
	hr = this->Internal->pGraph->AddFilter(this->Internal->pGrabberFilter, L"Sample Grabber");
    if (FAILED(hr)){
		vtkErrorMacro(<<"Could not link sample grabber filter to graph.");
		this->ReleaseSystemResources();
		return;
    }
	hr = this->Internal->pGrabberFilter->QueryInterface(IID_PPV_ARGS(&(this->Internal->pGrabber)));
	if (FAILED(hr)){
		vtkErrorMacro(<<"Could not link sample grabber to its filter.");
		this->ReleaseSystemResources();
		return;
    }

	//set the media type on the grabber
	VIDEOINFOHEADER * vih = new VIDEOINFOHEADER();
	vih->AvgTimePerFrame = 10000000.0 / this->FrameRate;
	this->Internal->mediaType.pbFormat = (BYTE*) vih;
	hr = this->Internal->pGrabber->SetMediaType(&(this->Internal->mediaType));
	if (FAILED(hr)){
		vtkErrorMacro(<<"Could not set sample grabber to RGB24 media type.");
		this->ReleaseSystemResources();
		return;
    }

	//create the source filter and set it to the graph
	this->Internal->pSourceFilter = this->Internal->deviceArray.at(this->videoSourceNumber);
	hrAdd = this->Internal->pGraph->AddFilter(this->Internal->pSourceFilter, L"Video Capture");
	if (FAILED(hr)||FAILED(hrAdd)){
		vtkErrorMacro(<<"Could not create source filter or add it to the graph.");
		this->ReleaseSystemResources();
		return;
	}
	hr = this->Internal->pSourceFilter->EnumPins(&(this->Internal->pEnum));
    if (FAILED(hr)) {
		vtkErrorMacro(<<"Could not enumerate over source pins.");
		this->ReleaseSystemResources();
		return;
    }
	IPin* pPin = 0;
	unsigned int count = 0;
	while ( this->Internal->pEnum->Next(1, &pPin, NULL) == S_OK ){
		hr = ConnectFilters(this->Internal->pGraph, pPin, this->Internal->pGrabberFilter);
		SAFE_RELEASE(pPin);
		if (SUCCEEDED(hr))  break;
	}
	if(FAILED(hr)){
		vtkErrorMacro(<<"Could not link source to the grabber.");
		this->ReleaseSystemResources();
		return;
	}

	//grab the frame rate and frame size from the settings page
	AM_MEDIA_TYPE* mt = &(this->Internal->mediaType);
	hr = this->Internal->pGrabber->GetConnectedMediaType( mt );
	vih = (VIDEOINFOHEADER*) mt->pbFormat;
	this->FrameSize[0] = vih->bmiHeader.biWidth;
	this->FrameSize[1] = vih->bmiHeader.biHeight;
	this->ImageSize = 3 * this->FrameSize[0] * this->FrameSize[1];
	this->FrameSize[2] = 1;

	//create the VTK output buffer
	this->medialBufferMutex->Lock();
	if(this->output) delete[] this->output;
	this->output = new char[this->ImageSize];
	this->medialBufferMutex->Unlock();

	//create the null renderer
	hr = CoCreateInstance (CLSID_NullRenderer, NULL, CLSCTX_INPROC_SERVER, IID_IBaseFilter, (void **) &(this->Internal->pNullRenderer));
	hrAdd = this->Internal->pGraph->AddFilter ( this->Internal->pNullRenderer, L"Null Renderer");
	if (FAILED(hr) || FAILED(hrAdd) ){
		vtkErrorMacro(<<"Could not create and add null renderer.");
		this->ReleaseSystemResources();
		return;
	}
	hr = ConnectFilters(this->Internal->pGraph, this->Internal->pGrabberFilter, this->Internal->pNullRenderer);
    if (FAILED(hr)){
		vtkErrorMacro(<<"Could not link null renderer to graph.");
		this->ReleaseSystemResources();
		return;
    }

	//set the call-back mechanism (note that 2nd param = 1 sends the buffer with the callback)
	hr = this->Internal->pGrabber->SetCallback(this->Internal, 1);
	if( FAILED(hr) ){
		vtkErrorMacro(<<"Could not set the grabber callback method.");
		this->ReleaseSystemResources();
		return;
	}

	//start the graph previewing (TODO: Determine if this should be in a separate play method?)
	hr = this->Internal->pMediaControl->Run();
	if (FAILED(hr)){
		vtkErrorMacro(<<"Could not run graph.");
		this->ReleaseSystemResources();
		return;
	}

	// Initialization worked
	this->Initialized = 1;

	// Update frame buffer  to reflect any changes
	this->UpdateFrameBuffer();
}  

//----------------------------------------------------------------------------
void vtkDirectShowVideoSource::ReleaseSystemResources()
{

	//cleans up the sample grabber and the null renderer
	SAFE_RELEASE( this->Internal->pNullRenderer );
	SAFE_RELEASE( this->Internal->pGrabberFilter );
	SAFE_RELEASE( this->Internal->pGrabber );

	//cleans up the media event and control interfaces
	SAFE_RELEASE( this->Internal->pMediaEvent );
	SAFE_RELEASE( this->Internal->pMediaControl );

	//clean up the filter that grabs the source
	SAFE_RELEASE( this->Internal->pEnum );
	SAFE_RELEASE( this->Internal->pSourceFilter );

	//cleans up the capture graph builder
	SAFE_RELEASE( this->Internal->pBuilder );

	//cleans up the graph builder (should be called after rest of pipeline is clear)
	SAFE_RELEASE( this->Internal->pGraph );

	//cleans up the COM library, which is the final step is releasing the
	//system resources
	CoUninitialize();
	
	//clean up the median buffer
	delete[] this->output;
	this->output = 0;

	this->Initialized = 0;
}

//----------------------------------------------------------------------------
void vtkDirectShowVideoSource::UpdateFrameBuffer()
{
  int i, oldExt;
  int ext[3];
  vtkDataArray *buffer;

  // clip the ClipRegion with the FrameSize
  for (i = 0; i < 3; i++)
    {
    oldExt = this->FrameBufferExtent[2*i+1] - this->FrameBufferExtent[2*i] + 1;
    this->FrameBufferExtent[2*i] = ((this->ClipRegion[2*i] > 0) 
                             ? this->ClipRegion[2*i] : 0);  
    this->FrameBufferExtent[2*i+1] = ((this->ClipRegion[2*i+1] < 
                                       this->FrameSize[i]-1) 
                             ? this->ClipRegion[2*i+1] : this->FrameSize[i]-1);

    ext[i] = this->FrameBufferExtent[2*i+1] - this->FrameBufferExtent[2*i] + 1;
    if (ext[i] < 0)
      {
      this->FrameBufferExtent[2*i] = 0;
      this->FrameBufferExtent[2*i+1] = -1;
      ext[i] = 0;
      }

    if (oldExt > ext[i])
      { // dimensions of framebuffer changed
      this->OutputNeedsInitialization = 1;
      }
    }

  // total number of bytes required for the framebuffer
  int bytesPerRow = ext[0]*(this->FrameBufferBitsPerPixel/8);
  bytesPerRow = ((bytesPerRow + this->FrameBufferRowAlignment - 1) /
                 this->FrameBufferRowAlignment)*this->FrameBufferRowAlignment;
  int totalSize = bytesPerRow * ext[1] * ext[2];
  i = this->FrameBufferSize;

  while (--i >= 0)
    {
    buffer = reinterpret_cast<vtkDataArray *>(this->FrameBuffer[i]);
    if (buffer->GetDataType() != VTK_UNSIGNED_CHAR ||
        buffer->GetNumberOfComponents() != 3 ||
        buffer->GetNumberOfTuples() != totalSize)
      {
      buffer->Delete();
      buffer = vtkUnsignedCharArray::New();
      this->FrameBuffer[i] = buffer;
      buffer->SetNumberOfComponents(3);
      buffer->SetNumberOfTuples(ext[0] * ext[1] * ext[2]);

      }
    }
}


void vtkDirectShowVideoSource::InternalGrab()
{

  // get a thread lock on the frame buffer
  this->FrameBufferMutex->Lock();

  if (this->AutoAdvance)
    {
    this->AdvanceFrameBuffer(1);
    if (this->FrameIndex + 1 < this->FrameBufferSize)
      {
      this->FrameIndex++;
      }
    }

  int index = this->FrameBufferIndex % this->FrameBufferSize;
  while (index < 0)
    {
    index += this->FrameBufferSize;
    }
  
  // Get a pointer to the location of the frame buffer
  char *ptr = (char *) reinterpret_cast<vtkUnsignedCharArray *>(this->FrameBuffer[index])->GetPointer(0);

  // Copy image into frame buffer
  this->medialBufferMutex->Lock();
	  for( int i = 0; i <= this->FrameBufferExtent[3] - this->FrameBufferExtent[2]; i++ ){
		  size_t copySize = 3 * sizeof(char) * (this->FrameBufferExtent[1] - this->FrameBufferExtent[0] + 1);
		  char* startPtr = this->output + 3 * (i * this->FrameSize[0] + this->FrameBufferExtent[0]);
		  char* destPtr = ptr + 3 * i * (this->FrameBufferExtent[1] - this->FrameBufferExtent[0] + 1);
		  memcpy(destPtr, startPtr, copySize);
	  }
  this->medialBufferMutex->Unlock();

  this->FrameBufferTimeStamps[index] = vtkTimerLog::GetUniversalTime();

  if (this->FrameCount++ == 0)
    {
    this->StartTimeStamp = this->FrameBufferTimeStamps[index];
    }

  this->Modified();

  this->FrameBufferMutex->Unlock();
}

#include "WTypes.h"
#include "comutil.h"


const char* vtkDirectShowVideoSource::GetDeviceName(int d){
	if( d < this->devices.size() )
		return this->devices.at(d).c_str();
	return 0;
}

void vtkDirectShowVideoSource::EnumerateVideoSources(){

	// Create the System Device Enumerator.
	HRESULT hr;
	hr = CoInitialize(NULL);
	ICreateDevEnum *pSysDevEnum = NULL;
	IBaseFilter *pSrc = NULL;
	hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER,
		IID_ICreateDevEnum, (void **)&pSysDevEnum);
	if (FAILED(hr))
	{
		return;
	}

	// Obtain a class enumerator for the video compressor category.
	IEnumMoniker *pEnumCat = NULL;
	hr = pSysDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &pEnumCat, 0);

	if (hr == S_OK) 
	{
		// Enumerate the monikers.
		IMoniker *pMoniker = NULL;
		ULONG cFetched;
		while(pEnumCat->Next(1, &pMoniker, &cFetched) == S_OK)
		{
			IPropertyBag *pPropBag;
			hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag, 
				(void **)&pPropBag);
			if (SUCCEEDED(hr))
			{
				// To retrieve the filter's friendly name, do the following:
				VARIANT varName;
				VariantInit(&varName);
				hr = pPropBag->Read(L"FriendlyName", &varName, 0);
				if (SUCCEEDED(hr))
				{
					std::string videoName = _com_util::ConvertBSTRToString(varName.bstrVal);
					this->devices.push_back(videoName);
					hr = pMoniker->BindToObject(0,0,IID_IBaseFilter, (void**)&pSrc);
					this->Internal->deviceArray.push_back(pSrc);
				}
				VariantClear(&varName);
			}
			pMoniker->Release();
		}
		pEnumCat->Release();
	}
	pSysDevEnum->Release();

}