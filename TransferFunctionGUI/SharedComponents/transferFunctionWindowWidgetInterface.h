#ifndef TRANSFERFUNCTIONWINDOWWIDGETINTERFACE
#define TRANSFERFUNCTIONWINDOWWIDGETINTERFACE

#include "qwidget.h"
#include "vtkCudaObject.h"
#include <vector>

class transferFunctionWindowWidgetInterface : public QWidget
{

public:
  transferFunctionWindowWidgetInterface(QWidget *parent = 0) : QWidget(parent) {};

  //keyboard options
  virtual void keyPressEvent(QKeyEvent* e) = 0;
  virtual void keyReleaseEvent(QKeyEvent* e) = 0;
  
  virtual void LoadedImageData() = 0;
  virtual void UpdateScreen() = 0;
  
  void AddCapableObject( vtkCudaObject* newObject );
  void RemoveCapableObject( vtkCudaObject* remObject );
  int GetNumberOfCapableObjects();
  vtkCudaObject* GetObject(int index);
  
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
  
  std::vector<vtkCudaObject*> capableObjects;

};

#endif