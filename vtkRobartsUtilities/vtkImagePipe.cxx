/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkImagePipe.cxx

  Copyright (c) John Baxter, Robarts Research Institute
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkImagePipe.h"

#include "vtkObjectFactory.h"
#include "vtkDataArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkCriticalSection.h"
#include "vtkTimerLog.h"

#include "vtkPointData.h"

#include <iostream>

#include <unistd.h>

vtkStandardNewMacro(vtkImagePipe);

extern "C"
struct vtkImagePipeInitData {
	int extent[6];
	double origin[3];
	double spacing[3];
	int scalarType;
	int scalarSize;
	int numComponents;
	int imageSize; //can also be used to weakly confirm data integrity
};


//----------------------------------------------------------------------------
vtkImagePipe::vtkImagePipe()
{
	//set some reasonable default values
	this->Initialized = 0;

	this->controller = vtkSocketController::New();
	this->controller->Initialize();
	threader = vtkMultiThreader::New();

	//initialize to neither client nor server, but unset
	this->isServer = false;
	this->serverSet = false;
	this->portNumber = -1;
	this->IPAddress = 0;
	this->serverSocket = 0;
	this->clientSocket = 0;
	this->buffer = 0;
	this->ImageSize = 0;

	//initialize the mutex locks
	this->newThreadLock = vtkMutexLock::New();
	this->rwBufferLock = vtkReadWriteLock::New();

}

//----------------------------------------------------------------------------
vtkImagePipe::~vtkImagePipe()
{
  this->ReleaseSystemResources();
  this->newThreadLock->Delete();
  this->rwBufferLock->Delete();
  this->threader->Delete();
}  

//----------------------------------------------------------------------------
void vtkImagePipe::ReleaseSystemResources()
{
	if( !this->Initialized ) return;

	if( !this->isServer ){
		int request = -1;
		this->clientSocket->Send( (void*) &request, sizeof(request) );
		this->clientSocket->CloseSocket();
	}else{
		this->threader->TerminateThread( this->mainServerThread );
	}

	if( this->serverSocket ){
		this->serverSocket->CloseSocket();
		this->serverSocket->Delete();
		this->serverSocket = 0;
	}
	if( this->clientSocket ){
		this->clientSocket->CloseSocket();
		this->clientSocket->Delete();
		this->clientSocket = 0;
	}
	this->portNumber = -1;
	this->IPAddress = 0;
	this->isServer = false;
	this->serverSet = false;
	this->Initialized = 0;
}

//----------------------------------------------------------------------------
void vtkImagePipe::SetInput( vtkImageData* in ){
	if( !this->serverSet || !this->isServer ){
		vtkErrorMacro(<<"Must be in server mode.");
		return;
	}

	this->buffer = in;
}

//----------------------------------------------------------------------------
vtkImageData* vtkImagePipe::GetOutput( ){
	if( !this->serverSet || this->isServer ){
		vtkErrorMacro(<<"Must be in client mode.");
		return 0 ;
	}
	if( !this->Initialized ){
		vtkErrorMacro(<<"Must be initialized first.");
		return 0 ;
	}

	return this->buffer;
}

//----------------------------------------------------------------------------
void vtkImagePipe::SetAsServer( bool isServer ){

	// if we are already initialized, you cannot set the server status
	if (this->Initialized){
		vtkErrorMacro(<<"Must uninitialize before changing parameters.");
		return;
	}

	this->isServer = isServer;
	if( this->serverSocket ){
		this->serverSocket->CloseSocket();
		this->serverSocket->Delete();
		this->serverSocket = 0;
	}
	if( this->clientSocket ){
		this->clientSocket->CloseSocket();
		this->clientSocket->Delete();
		this->clientSocket = 0;
	}
	if( isServer ) this->serverSocket = vtkServerSocket::New();
	else this->clientSocket = vtkClientSocket::New();
	this->serverSet = true;
}

void vtkImagePipe::SetSourceAddress( char* ipAddress, int portNumber ){
	
	// if we are already initialized, you cannot set the server status
	if (this->Initialized){
		vtkErrorMacro(<<"Must uninitialize before changing parameters.");
		return;
	}

	if( portNumber < 0 ) {
		vtkErrorMacro(<<"Invalid port number.");
		return;
	}

	if( !this->serverSet ) {
		vtkErrorMacro(<<"Must first specify whether client or server using SetAsServer().");
		return;
	}

	if( this->isServer ){
		if( this->serverSocket->CreateServer( portNumber ) ){
			vtkErrorMacro(<<"Could not connect to port.");
			return;
		}else{
			this->portNumber = portNumber;
		}
	}else{
		this->IPAddress = ipAddress;
		this->portNumber = portNumber;
	}
}

//----------------------------------------------------------------------------
void vtkImagePipe::PrintSelf(ostream& os, vtkIndent indent)
{

}

//----------------------------------------------------------------------------
void vtkImagePipe::Initialize()
{
	// if we are already initialized, do not initialize again
	if (this->Initialized){
		return;
	}

	//if we haven't set the client-server status, 
	if ( !this->serverSet || this->portNumber == -1 ){
		vtkErrorMacro(<<"Set the client/server settings before initialization.");
		return;
	}

	//if we are the server, make sure we have input set
	if( this->isServer ){
		if( !this->buffer ){
			vtkErrorMacro(<<"Need to set the input.");
			return;
		}
		this->mainServerThread = this->threader->SpawnThread( (vtkThreadFunctionType) &FirstServerSideUpdate, (void*) this );
	}

	//create the connection if a client
	if( !this->isServer ){
		int connectedStatus = this->clientSocket->ConnectToServer( this->IPAddress, this->portNumber );
		if( connectedStatus ){
			vtkErrorMacro(<<"Could not connect to server side socket.");
			return;
		}
	}

	//create output buffer if client
	if( !this->isServer ){
		buffer = vtkImageData::New();
	}

	// Initialization worked
	this->Initialized = 1;

}  

void* vtkImagePipe::FirstServerSideUpdate(vtkMultiThreader::ThreadInfo *data){
	
	vtkImagePipe *self = (vtkImagePipe *)(data->UserData);
	
	//enter the infinite loop
	while(true){
		//check for any new clients
		vtkClientSocket* newClient = self->serverSocket->WaitForConnection(1);
		if( newClient ){
			self->newThreadLock->Lock();
			self->clientSocket = newClient;
			self->threader->SpawnThread( (vtkThreadFunctionType) &ServerSideUpdate, (void*) self );
		}else{
			sleep( 100 );
		}
	}
	return 0;

}

void* vtkImagePipe::ServerSideUpdate(vtkMultiThreader::ThreadInfo *data){
	
	//collect server and client information
	vtkImagePipe *self = (vtkImagePipe *)(data->UserData);
	vtkClientSocket* client = self->clientSocket;
	self->newThreadLock->Unlock();

	//enter the infinite loop
	while(true){

		//if we have a request, push data onto the pipe
		int request = 0;
		int amount = client->Receive( &request, sizeof(request), 1 );
		if( amount == 0 ) break;
		else if( amount < sizeof(request) ) continue;
		else if( request != 1 ) break;
		
		//read lock the buffer
		self->rwBufferLock->ReaderLock();

		//create the info structure
		vtkImagePipeInitData initData;
		self->buffer->GetExtent( initData.extent );
		self->buffer->GetSpacing( initData.spacing );
		self->buffer->GetOrigin( initData.origin );
		initData.scalarType = self->buffer->GetScalarType();
		initData.numComponents = self->buffer->GetNumberOfScalarComponents();
		initData.scalarSize = self->buffer->GetScalarSize();
		initData.imageSize = (initData.extent[1] - initData.extent[0] + 1) *
								(initData.extent[3] - initData.extent[2] + 1) *
								(initData.extent[5] - initData.extent[4] + 1) *
								initData.numComponents * initData.scalarSize;
		self->ImageSize = initData.imageSize;

		//send over the data
		client->Send( (void*) &initData, sizeof(initData) );
		client->Send( self->buffer->GetScalarPointer(), self->ImageSize );
		
		//read unlock the buffer
		self->rwBufferLock->ReaderUnlock();

	}

	client->CloseSocket();
	return 0;

}

void vtkImagePipe::Update(){
	if( !this->Initialized ) return;
	if( this->isServer ){
		//protect buffer updating with read/write lock
		this->rwBufferLock->WriterLock();
		this->buffer->Update();
		this->rwBufferLock->WriterUnlock();
	}else{
		ClientSideUpdate();
	}
}

void vtkImagePipe::ClientSideUpdate(){

	//send input request
	int request = 1;
	int serverThere = this->clientSocket->Send( &request, sizeof(request) );
	if( !serverThere ){
		vtkErrorMacro(<<"Server unavailable.");
		return;
	}

	//collect input parameters and change the output buffer if needed
	vtkImagePipeInitData initData;
	serverThere = clientSocket->Receive( (void*) &initData, sizeof(initData), 1 );
	if( !serverThere ){
		vtkErrorMacro(<<"Server unavailable.");
		return;
	}
	this->buffer->SetSpacing( initData.spacing );
	this->buffer->SetOrigin( initData.origin );
	this->buffer->SetExtent( initData.extent );
	this->buffer->SetNumberOfScalarComponents( initData.numComponents );
	this->buffer->SetScalarType( initData.scalarType );
	int calcImageSize = (initData.extent[1] - initData.extent[0] + 1) *
						(initData.extent[3] - initData.extent[2] + 1) *
						(initData.extent[5] - initData.extent[4] + 1) *
						initData.numComponents * initData.scalarSize;
	this->ImageSize = initData.imageSize;
	if( this->ImageSize != calcImageSize ){
		vtkErrorMacro(<<"Image information packet does not conform to the image size error check.");
		return;
	}
	this->buffer->AllocateScalars();

	//grab the data from the socket
	serverThere = this->clientSocket->Receive( this->buffer->GetScalarPointer(), this->ImageSize, 1 );
	if( !serverThere ){
		vtkErrorMacro(<<"Server unavailable.");
		return;
	}

}
