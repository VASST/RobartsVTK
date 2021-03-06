/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef TRANSFERFUNCTIONWINDOWWIDGETINTERFACE
#define TRANSFERFUNCTIONWINDOWWIDGETINTERFACE

#include "TFUICommonExport.h"

#include <QWidget>
#include <vector>

class CudaObject;

class TFUICommonExport qTransferFunctionWindowWidgetInterface : public QWidget
{

public:
  qTransferFunctionWindowWidgetInterface(QWidget *parent = 0);

  //keyboard options
  virtual void keyPressEvent(QKeyEvent* e) = 0;
  virtual void keyReleaseEvent(QKeyEvent* e) = 0;

  virtual void LoadedImageData() = 0;
  virtual void UpdateScreen() = 0;

  void AddCapableObject( CudaObject* newObject );
  void RemoveCapableObject( CudaObject* remObject );
  size_t GetNumberOfCapableObjects();
  CudaObject* GetObject(unsigned int index);

  virtual int GetRComponent();
  virtual int GetGComponent();
  virtual int GetBComponent();

  virtual double GetRMax();
  virtual double GetGMax();
  virtual double GetBMax();
  virtual double GetRMin();
  virtual double GetGMin();
  virtual double GetBMin();

private:
  std::vector<CudaObject*> capableObjects;
};

#endif