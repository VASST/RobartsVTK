#include "vtkBarrierLock.h"
#include "vtkObjectFactory.h"

vtkStandardNewMacro(vtkBarrierLock);

vtkBarrierLock::vtkBarrierLock(){
	this->NumEntered = 0;
	this->NumEnteredMax = 0;
	this->Repeatable = false;
	this->BarrierUsed = false;

	this->Condition = vtkConditionVariable::New();
	this->EntryLock = vtkMutexLock::New();
}

vtkBarrierLock::~vtkBarrierLock(){
	this->Condition->Delete();
	this->EntryLock->Delete();
}

void vtkBarrierLock::Initialize(int numThreads){
	this->EntryLock->Lock();
	BarrierUsed = false;
	NumEnteredMax = numThreads;
	if(NumEntered >= NumEnteredMax)
		Condition->Broadcast();
	BarrierUsed = false;
	this->EntryLock->Unlock();
}

void vtkBarrierLock::DeInitialize(){
	this->Condition->Broadcast();
}

bool vtkBarrierLock::Query(){
	this->EntryLock->Lock();
	bool retVal = false;
	retVal = retVal || (this->BarrierUsed && this->Repeatable);
	retVal = retVal || (this->NumEntered == this->NumEnteredMax - 1);
	this->EntryLock->Unlock();
	return retVal;
}

void vtkBarrierLock::Enter(){
	this->EntryLock->Lock();
	this->NumEntered++;

	if(this->NumEntered < this->NumEnteredMax && !(BarrierUsed && !Repeatable) )
		Condition->Wait(this->EntryLock);
	else{
		Condition->Broadcast();
		BarrierUsed = true;
	}
	this->NumEntered--;

	this->EntryLock->Unlock();
}