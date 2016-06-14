/*=========================================================================

Program:   Camera Calibration
Module:    $RCSfile: QComputeThread.h,v $
Creator:   Adam Rankin <arankin@robarts.ca>
Language:  C++
Author:    $Author: Adam Rankin $

==========================================================================

Copyright (c) Adam Rankin, arankin@robarts.ca

Use, modification and redistribution of the software, in source or
binary forms, are permitted provided that the following terms and
conditions are met:

1) Redistribution of the source code, in verbatim or modified
form, must retain the above copyright notice, this license,
the following disclaimer, and any notices that refer to this
license and/or the following disclaimer.

2) Redistribution in binary form must include the above copyright
notice, a copy of this license and the following disclaimer
in the documentation or with other materials provided with the
distribution.

3) Modified copies of the source code must be clearly marked as such,
and must not be misrepresented as verbatim copies of the source code.

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

#ifndef __QCOMPUTETHREAD_H__
#define __QCOMPUTETHREAD_H__

#include <QThread>
#include <opencv2/core.hpp>

class QComputeThread : public QThread
{
  enum ComputationType
  {
    COMPUTATION_NONE,
    COMPUTATION_MONO_CALIBRATE,
    COMPUTATION_STEREO_CALIBRATE
  };

  Q_OBJECT

public:
  QComputeThread(QObject *parent = 0);
  ~QComputeThread();

  bool CalibrateCamera();
  bool StereoCalibrate();

signals:
  void monoCalibrationComplete();
  void stereoCalibrationComplete();

protected:
  void run() Q_DECL_OVERRIDE;

  ComputationType Computation;
  QMutex          Mutex;
};

#endif
