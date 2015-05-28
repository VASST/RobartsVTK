#include "vtkReadWriteLock.h"
#include "vtkObjectFactory.h"

vtkStandardNewMacro(vtkReadWriteLock);

vtkReadWriteLock::vtkReadWriteLock(){
  this->noWriters = vtkMutexLock::New();
  this->noReaders = vtkMutexLock::New();
  this->counter = vtkMutexLock::New();

  this->readerCount = 0;
}

vtkReadWriteLock::~vtkReadWriteLock(){
  this->noWriters->Delete();
  this->noReaders->Delete();
  this->counter->Delete();
}

void vtkReadWriteLock::ReaderLock(){
  this->noWriters->Lock();

  //collect counting information
  this->counter->Lock();
  unsigned int prevReaderCount = this->readerCount;
  this->readerCount++;
  this->counter->Unlock();

  //if we are the first one in, apply the lock to keep writers out
  if( prevReaderCount == 0 ) this->noReaders->Lock();

  this->noWriters->Unlock();
}

void vtkReadWriteLock::ReaderUnlock(){
  //get the number of readers still in the critical section
  this->counter->Lock();
  this->readerCount--;
  unsigned int currReaderCount = this->readerCount;
  this->counter->Unlock();

  //if we are the last one out, unlock the writers area
  if( currReaderCount == 0 ) this->noReaders->Unlock();
}

void vtkReadWriteLock::WriterLock(){
  this->noWriters->Lock();
  this->noReaders->Lock();
  this->noWriters->Unlock();
}

void vtkReadWriteLock::WriterUnlock(){
  this->noReaders->Unlock();
}
