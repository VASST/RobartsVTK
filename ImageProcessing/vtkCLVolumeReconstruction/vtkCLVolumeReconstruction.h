/*==========================================================================

  Copyright (c) 2016 Uditha L. Jayarathne, ujayarat@robarts.ca

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

#ifndef _vtkCLVolumeReconstruction_h_
#define _vtkCLVolumeReconstruction_h_

#include "vtkCLVolumeReconstructionExport.h"

// OpenCL includes
#include <CL/cl.h>

// STL includes
#include <algorithm>
#include <array>
#include <mutex>
#include <queue>

// CUDA includes
#include <vector_functions.h>
#include <vector_types.h>

// VTK includes
#include <vtkImageAlgorithm.h>
#include <vtkSmartPointer.h>

class vtkImageData;
class vtkMatrix4x4;
class vtkTransform;
class vtkDataSet;

//# define KERNEL_DEBUG
//# define VCLVR_DEBUG

#define max3(a,b,c) std::max(a, std::max(b, c))
#define distance(v, plane) (plane.x*v.x + plane.y*v.y + plane.z*v.z + plane.w)/sqrt(plane.x*plane.x + plane.y*plane.y + plane.z*plane.z)
#define inrange(x,a,b) ((x) >= (a) && (x) < (b))

#define pos_matrix_a(x,y) (pos_matrix[(y)*4 + (x)])

class vtkCLVolumeReconstructionExport vtkCLVolumeReconstruction : public vtkImageAlgorithm
{
public:
  static vtkCLVolumeReconstruction* New();
  vtkTypeMacro(vtkCLVolumeReconstruction, vtkAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  /* Initializes devices */
  void Initialize();

  /* Print device information */
  void PrintInfo();

  /* Set CUDA device index */
  void SetDevice(int);

  /* Releases resources */
  void ReleaseDevices();

  /* Set program_src */
  void SetProgramSourcePath(const std::string& fileName);

  /* Set BScan width and Height */
  void SetBScanSize(int, int);

  /* Set BScan spacing */
  void SetBScanSpacing(double, double);

  /* Set output origin */
  void SetOutputOrigin(double, double, double);

  /* Set US Calibration matrix. Default is identity matrix */
  void SetCalMatrix(float*);

  /* Set Reconstruction axis. 0 - X axis, 1 - Y axis, 2 - Z axis */
  void SetReconstructionAxis(int);

  /* Set input data */
  void SetInputImageData(double, vtkImageData*);

  /* Set input position data */
  void SetInputPoseData(double, vtkMatrix4x4*);
  void SetImagePoseTransform(vtkTransform*);

  /* Clear buffers */
  void ClearReconstruction();

  /* Set output Extent. This needs to be set. Otherwise the volume will be zero. */
  void SetOutputExtent(int, int, int, int, int, int);

  /* Get output extent */
  void GetOutputExtent(int*);

  /* Set output spacing */
  void SetOutputSpacing(double);

  /* Start doing freehand reconstruction */
  void StartReconstruction();

  /* Starts realtime reconstruction */
  void UpdateReconstruction();

  /* Get Output Volume */
  void GetOutputVolume(vtkImageData*);

  /*Get volume origin */
  void GetOrigin(double*) const;

  /* Get Spacing */
  void GetSpacing(double spacing[3]) const;

protected:
  vtkCLVolumeReconstruction();
  ~vtkCLVolumeReconstruction();

  virtual int RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector);

protected:
  /* Utility functions */
  std::string FileToString(const std::string& fileName, const std::string& header = "");

  /* creates the OpenCL kernel given the device */
  cl_kernel OpenCLKernelBuild(cl_program, cl_device_id, char*);

  /* Initialize buffers */
  void InitializeBuffers();

  /*  create CL buffers */
  cl_mem OpenCLCreateBuffer(cl_context, cl_mem_flags, size_t, void*);

  /* */
  void OpenCLCheckError(int, char* info = "");

  /* Multiply calibration matrix into position matrix */
  void CalibratePosMatrix(float*, float*);

  /* */
  void InsertPlanePoints(float*);

  /* Fill bscan_plane_equation */
  void InsertPlaneEquation();

  /* */
  int FindIntersections(int axis);

  /* */
  void FillVoxels();

  /* Shifts data queues. 1 - if the queues are full  0- otherwise */
  int ShiftQueues();

  /* Wait for input data */
  void GrabInputData();

  /* Update output volume */
  void UpdateOutputVolume();

  /* Print the content of a matrix */
  void DumpMatrix(int, int, float*);

protected:
  typedef struct
  {
    float4 corner0;
    float4 cornerx;
    float4 cornery;
  } plane_pts;

  /* CL Kernels */
  cl_kernel transform;
  cl_kernel round_off_translate;
  cl_kernel fill_volume;
  cl_kernel fill_holes;
  cl_kernel adv_fill_voxels;
  cl_kernel trace_intersections;
  cl_command_queue reconstruction_cmd_queue;

  /* CL Device */
  cl_device_id device;

  /* CL Context */
  cl_context context;

  /* CL Program */
  cl_program program;

  /* device Lock */
  std::mutex cl_device_lock;

  /* Output volume width */
  int volume_width;

  /* Output volume height */
  int volume_height;

  /* Output volume depth */
  int volume_depth;

  /* Output volume spacing */
  float volume_spacing;

  /* Output volume origin */
  std::array<float, 3> volume_origin;

  /* Output Extent */
  std::array<int, 6> volume_extent;

  /* bscan width and height */
  int bscan_w, bscan_h;

  /* bscan spacing */
  float bscan_spacing_x, bscan_spacing_y;

  /* Input Image Data */
  vtkImageData* image_data;

  /* Input pose data matrix */
  vtkMatrix4x4* pose_data;

  /* timestamp of input data */
  float timestamp;

  /* Use this device idx */
  int device_index;

  /* Path to the CL Program source */
  std::string program_src;

  /* Private Constants */
  static const int BSCAN_WINDOW = 4; // must be >= 4 if PT

  // Host variables
  int                         intersections_size;
  int                         volume_size;
  int                         max_vol_dim;
  int                         x_vector_queue_size;
  int                         y_vector_queue_size;
  int                         plane_points_queue_size;
  int                         mask_size;
  int                         bscans_queue_size;
  int                         bscan_timetags_queue_size;
  int                         bscan_plane_equation_queue_size;
  size_t                      global_work_size[1];
  size_t                      local_work_size[1];

  // Host buffers
  float4*                     x_vector_queue;
  float4*                     y_vector_queue;
  unsigned char**             bscans_queue;
  float**                     pos_matrices_queue;
  float*                      bscan_timetags_queue;
  float*                      pos_timetags_queue;
  float4*                     bscan_plane_equation_queue;
  plane_pts*                  plane_points_queue;
  unsigned char*              mask;
  std::queue<float>           timestamp_queue;
  std::queue<vtkImageData*>   imageData_queue;
  std::queue<vtkMatrix4x4*>   poseData_queue;

  // Device variables
  int                         dev_x_vector_queue_size;
  int                         dev_y_vector_queue_size;
  int                         dev_plane_points_queue_size;

  // Devices buffers
  cl_mem                      dev_intersections;
  cl_mem                      dev_volume;
  cl_mem                      dev_x_vector_queue;
  cl_mem                      dev_y_vector_queue;
  cl_mem                      dev_plane_points_queue;
  cl_mem                      dev_mask;
  cl_mem                      dev_bscans_queue;
  cl_mem                      dev_bscan_timetags_queue;
  cl_mem                      dev_bscan_plane_equation_queue;

  // cal_matrix is the 1x16 us calibration matrix
  float*                      pos_timetags;
  float*                      pos_matrices;
  float*                      bscan_timetags;
  float*                      calibration_matrix;
  int                         axis;

  vtkSmartPointer<vtkTransform> image_pose;
  vtkSmartPointer<vtkImageData> reconstructed_volume;
  std::mutex                    input_data_mutex;
};

#endif //_vtkCLVolumeReconstruction_h_
