/*==========================================================================

  Copyright (c) 2015 Uditha L. Jayarathne, ujayarat@robarts.ca

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

#include <iostream>
#include "CudaReconstruction.h"

//# define KERNEL_DEBUG

//--------------------------------------------------------------
CudaReconstruction::CudaReconstruction()
{
  size_t temp;
  device_index = 1;
  program_src = file2string("./kernels.cl", "", &temp);

  // Default values for bscan size
  bscan_w = 640;
  bscan_h = 480;

  // Default values for the output volume
  volume_depth = volume_height = volume_width = 0;
  volume_spacing = 0.5;


  volume_origin[0] = 0;
  volume_origin[1] = 0;
  volume_origin[2] = 0;

  // Default value for cal_matrix is identity matrix
  cal_matrix = (float *)malloc(sizeof(float)*16);
  for(int i=0; i<4; i++)
  {
    for(int j=0; j<4; j++)
    {
      cal_matrix[4*i+j] = 0;
      if (i == j)
      {
        cal_matrix[4*i+j] = 1;
      }
    }
  }

  imageData = vtkImageData::New();
  poseData = vtkMatrix4x4::New();

  // Print device information
  PrintInfo();
}

//--------------------------------------------------------------
CudaReconstruction::~CudaReconstruction()
{

  // Release device memory
  clReleaseCommandQueue( reconstruction_cmd_queue );

  clReleaseMemObject( dev_intersections );
  clReleaseMemObject( dev_volume );
  clReleaseMemObject( dev_x_vector_queue );
  clReleaseMemObject( dev_y_vector_queue );
  clReleaseMemObject( dev_plane_points_queue );
  clReleaseMemObject( dev_mask );
  clReleaseMemObject( dev_bscans_queue );
  clReleaseMemObject( dev_bscan_plane_equation_queue );
  clReleaseMemObject( dev_bscan_timetags_queue );

  clReleaseProgram( program );
  clReleaseContext( context );

  // Release host memory
  free(x_vector_queue);
  free(y_vector_queue );
  free( bscans_queue );
  free( pos_matrices_queue );
  free(bscan_timetags_queue );
  free(pos_timetags_queue );
  free( bscan_plane_equation_queue );
  free( plane_points_queue );
  free( mask );
  free( volume );
}

//--------------------------------------------------------------
void CudaReconstruction::Initialize()
{
  cl_int err;
  cl_device_id devices[250];

# ifdef KERNEL_DEBUG
  int platform_idx = 1; // 0 for NVIDIA and 1 for Intel CPU
# else
  int platform_idx = 0;
# endif

  // Get available platforms
  cl_uint nPlatforms;
  clGetPlatformIDs(0, NULL, &nPlatforms);

  cl_platform_id * platformIDs = (cl_platform_id *)malloc(sizeof(cl_platform_id)*nPlatforms);
  clGetPlatformIDs(nPlatforms, platformIDs, NULL);

  cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[platform_idx], 0};

# ifdef KERNEL_DEBUG
  // Get handles to available CPU devices
  clGetDeviceIDs(platformIDs[platform_idx], CL_DEVICE_TYPE_CPU, 256, devices, NULL);
# else
  // Get handles to available GPU devices
  clGetDeviceIDs(platformIDs[platform_idx], CL_DEVICE_TYPE_GPU, 256, devices, NULL);
# endif

  device = devices[device_index];
  std::cout << "CL Device ID: " <<  device << std::endl;

  // Create CL context
  context = clCreateContext(cps, 1, &device, NULL, NULL, &err);

  // Create CommandQueue
  reconstruction_cmd_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  if (err != CL_SUCCESS)
  {
    std::cout << "ERROR clCreateCommandQueue: " << err << std::endl;
    exit(err);
  }

  // Create CL Program
  program = clCreateProgramWithSource(context, 1, (const char **)&program_src, 0, &err);

# ifdef KERNEL_DEBUG
  err     = clBuildProgram( program, 0, NULL,  "-g -s E:\\Work\\In-situ_US_Visualization\\IPCAI2016\\Codes\\Accelerated_Reconstruction\\src\\kernels.cl", NULL, NULL);
# else
  err     = clBuildProgram( program, 0, 0, 0, 0, 0);
# endif

  if( err != CL_SUCCESS )
  {
    size_t len;
    char buffer[512*512];
    memset(buffer, 0, 512*512);
    std::cout << "ERROR: Failed to build program on device " << device << ". Error code: " << err << std::endl;

    clGetProgramBuildInfo(
      program,                // the program object being queried
      device,                 // the device for which the OpenCL code was built
      CL_PROGRAM_BUILD_LOG,   // specifies that we want the build log
      sizeof(char)*512*512,   // the size of the buffer
      buffer,                 // on return, holds the build log
      &len);                  // on return, the actual size in bytes of the data returned

    std::cout << buffer << std::endl;
    exit(1);
  }

  // Now build kernels
  fill_volume     = ocl_kernel_build(program, device, "fill_volume");
  round_off_translate = ocl_kernel_build(program, device, "round_off_translate");
  transform     = ocl_kernel_build(program, device, "transform");
  fill_holes      = ocl_kernel_build(program, device, "fill_holes");
  trace_intersections = ocl_kernel_build(program, device, "trace_intersections");
  adv_fill_voxels   = ocl_kernel_build(program, device, "adv_fill_voxels");

  // Initialize Buffers
  initialize_buffers();

  omp_init_lock(&lock);

  //
  // TODO: call reconstruction thread
  //updateReconstruction();
}

//--------------------------------------------------------------
void CudaReconstruction::PrintInfo()
{
  cl_uint platforms_n;
  cl_uint devices_n;
  size_t temp_size;
  cl_platform_id * platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*256);
  cl_device_id * devices = (cl_device_id *) malloc(sizeof(cl_device_id)*256);
  char * str = (char *) malloc(sizeof(char)*2048);

  clGetPlatformIDs(256, platforms, &platforms_n);

  clGetPlatformIDs(256, platforms, &platforms_n);
  for (cl_uint i = 0; i < platforms_n; i++)
  {
    std::cout << "Platform " << i+1 << " of " << platforms_n << std::endl;

    clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 2048, str, &temp_size);
    std::cout << "\t CL_PLATFORM_VERSION: " << str << std::endl;

    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 2048, str, &temp_size);
    std::cout << "\t CL_PLATFORM_NAME: " << str << std::endl;

    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 2048, str, &temp_size);
    std::cout << "\t CL_PLATFORM_VENDOR: " << str << std::endl;

    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 256, devices, &devices_n);
    for(cl_uint j=0; j< devices_n; j++)
    {

      std::cout << "\t device " << j+1 << " of " << devices_n << std::endl;
      cl_device_type type;

      clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(type), &type, &temp_size);
      if (type == CL_DEVICE_TYPE_CPU)
      {
        std::cout << "\t\t CL_DEVICE_TYPE: CL_DEVICE_TYPE_CPU " << std::endl;
      }
      else if (type == CL_DEVICE_TYPE_GPU)
      {
        std::cout << "\t\t CL_DEVICE_TYPE: CL_DEVICE_TYPE_GPU" << std::endl;
      }
      else if (type == CL_DEVICE_TYPE_ACCELERATOR)
      {
        std::cout << "\t\t CL_DEVICE_TYPE: CL_DEVICE_TYPE_ACCELERATOR " << std::endl;
      }
      else if (type == CL_DEVICE_TYPE_DEFAULT)
      {
        std::cout << "\t\t CL_DEVICE_TYPE: CL_DEVICE_TYPE_DEFAULT " << std::endl;
      }
      else
      {
        std::cout << "\t\t CL_DEVICE_TYPE: (combination) " << std::endl;
      }

      cl_uint temp_uint;
      cl_ulong temp_ulong;
      size_t temp_size_t;
      size_t * size_t_array = (size_t *) malloc(sizeof(size_t)*3);

      std::cout << "\t\t device ID: " << devices[j] << std::endl;
      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(temp_uint), &temp_uint, &temp_size);
      std::cout << "\t\t CL_DEVICE_MAX_COMPUTE_UNITS: " << temp_uint << std::endl;

      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(temp_uint), &temp_uint, &temp_size);
      std::cout << "\t\t CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << temp_uint << std::endl;

      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, size_t_array, &temp_size);
      std::cout << "\t\t CL_DEVICE_MAX_WORK_ITEM_SIZES: " << size_t_array[0] << " " << size_t_array[1] << " "
                << size_t_array[2] << std::endl;

      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(temp_size_t), &temp_size_t, &temp_size);
      std::cout << "\t\t CL_DEVICE_MAX_WORK_GROUP_SIZE: " << temp_size_t << std::endl;

      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(temp_uint), &temp_uint, &temp_size);
      std::cout << "\t\t CL_DEVICE_MAX_CLOCK_FREQUENCY: " << temp_uint << std::endl;

      clGetDeviceInfo(devices[j], CL_DEVICE_ADDRESS_BITS, sizeof(temp_uint), &temp_uint, &temp_size);
      std::cout << "\t\t CL_DEVICE_ADDRESS_BITS: " << temp_uint << std::endl;

      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(temp_ulong), &temp_ulong, &temp_size);
      std::cout << "\t\t CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << temp_ulong << std::endl;

      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(temp_size_t), &temp_size_t, &temp_size);
      std::cout << "\t\t CL_DEVICE_MAX_PARAMETER_SIZE: " << temp_size_t << std::endl;

      clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(temp_ulong), &temp_ulong, &temp_size);
      std::cout << "\t\t CL_DEVICE_GLOBAL_MEM_SIZE: " << temp_ulong << std::endl;

      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 2048, str, &temp_size);
      std::cout << "\t\t CL_DEVICE_NAME: " << str << std::endl;

      clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 2048, str, &temp_size);
      std::cout << "\t\t CL_DEVICE_VENDOR: " << str << std::endl;

      clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, 2048, str, &temp_size);
      std::cout << "\t\t CL_DEVICE_EXTENSIONS: " << str << std::endl;
    }
  }

  std::cout << "\n";
}

//--------------------------------------------------------------
void CudaReconstruction::SetProgramSourcePath(char *path)
{
  size_t temp;
  this->program_src = file2string(path, "", &temp);
}

//--------------------------------------------------------------
void CudaReconstruction::SetBScanSize( int w, int h)
{
  // Set width and height
  this->bscan_w = w;
  this->bscan_h = h;
}

//--------------------------------------------------------------
void CudaReconstruction::SetBScanSpacing(double sx, double sy)
{
  // Set BScan spacing
  this->bscan_spacing_x = (float)sx;
  this->bscan_spacing_y = (float)sy;
}

//--------------------------------------------------------------
void CudaReconstruction::SetOutputOrigin( double x, double y, double z)
{
  this->volume_origin[0] = (float)x;
  this->volume_origin[1] = (float)y;
  this->volume_origin[2] = (float)z;
}

//--------------------------------------------------------------
void CudaReconstruction::SetCalMatrix(float *m)
{
  memcpy(this->cal_matrix, m, sizeof(float)*12);
}

//--------------------------------------------------------------
void CudaReconstruction::SetInputImageData(double time, vtkImageData *data)
{
  if( data != NULL )
  {
    imageData->DeepCopy( data );
    timestamp = (float)time;
  }
}

//--------------------------------------------------------------
void CudaReconstruction::SetInputPoseData(double time, vtkMatrix4x4 *mat)
{
  if( mat != NULL )
  {
    poseData->DeepCopy( mat );
    timestamp = (float)time;
  }
}

//--------------------------------------------------------------
void CudaReconstruction::SetOutputSpacing(double spacing)
{
  this->volume_spacing = (float)spacing;
}

//--------------------------------------------------------------
void CudaReconstruction::SetOutputExtent(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax)
{
  this->volume_extent[0] = xmin;
  this->volume_extent[1] = xmax;
  this->volume_extent[2] = ymin;
  this->volume_extent[3] = ymax;
  this->volume_extent[4] = zmin;
  this->volume_extent[5] = zmax;

  volume_width = this->volume_extent[1] - this->volume_extent[0];
  volume_height= this->volume_extent[3] - this->volume_extent[2];
  volume_depth = this->volume_extent[5] - this->volume_extent[4];
}

//--------------------------------------------------------------
void CudaReconstruction::StartReconstruction()
{
  max_vol_dim = max3(volume_width, volume_height, volume_depth);

  global_work_size[0] = ((max_vol_dim*max_vol_dim)/256+1)*256;
  local_work_size[0] = 256;

  // Initialize output volume to zero
  memset(volume, 0, sizeof(unsigned char)*volume_width*volume_height*volume_depth);

  // Set mask. Default is no mask (val 1 --> white). In mask Black is outside ROI while White is insite the ROI.
  memset(mask, 1, sizeof(unsigned char)*bscan_w*bscan_h);

  intersections_size = sizeof(cl_float4)*2*max_vol_dim*max_vol_dim;
  volume_size = volume_width * volume_height * volume_depth * sizeof(cl_uchar);
  x_vector_queue_size = BSCAN_WINDOW*sizeof(cl_float4);
  y_vector_queue_size = BSCAN_WINDOW*sizeof(cl_float4);
  plane_points_queue_size = BSCAN_WINDOW*sizeof(plane_pts);
  mask_size = bscan_w * bscan_h * sizeof(cl_uchar);
  bscans_queue_size = BSCAN_WINDOW * bscan_w * bscan_h * sizeof(cl_uchar);
  bscan_timetags_queue_size = BSCAN_WINDOW * sizeof(cl_float);
  bscan_plane_equation_queue_size = BSCAN_WINDOW * sizeof(float4);

  dev_x_vector_queue_size = BSCAN_WINDOW*sizeof(float)*4;
  dev_y_vector_queue_size = BSCAN_WINDOW*sizeof(float)*4;
  dev_plane_points_queue_size = BSCAN_WINDOW*sizeof(float)*4*3;

  dev_intersections = ocl_create_buffer(context, CL_MEM_READ_WRITE, intersections_size, NULL);
  dev_volume = ocl_create_buffer(context, CL_MEM_READ_WRITE, volume_size, volume);
  dev_x_vector_queue = ocl_create_buffer(context, CL_MEM_READ_WRITE, dev_x_vector_queue_size, NULL);
  dev_y_vector_queue = ocl_create_buffer(context, CL_MEM_READ_WRITE, dev_y_vector_queue_size, NULL);
  dev_plane_points_queue = ocl_create_buffer(context, CL_MEM_READ_WRITE, dev_plane_points_queue_size, NULL);
  dev_mask = ocl_create_buffer(context, CL_MEM_READ_WRITE, mask_size, NULL);
  dev_bscans_queue = ocl_create_buffer(context, CL_MEM_READ_WRITE, bscans_queue_size, NULL);
  dev_bscan_timetags_queue = ocl_create_buffer(context, CL_MEM_READ_WRITE, bscan_timetags_queue_size, NULL);
  dev_bscan_plane_equation_queue = ocl_create_buffer(context, CL_MEM_READ_WRITE, bscan_plane_equation_queue_size, NULL);
}

//--------------------------------------------------------------
void CudaReconstruction::UpdateReconstruction()
{
  // Retrieve US data and perform reconstruction
  if(shift_queues())
  {
    // Multiply the position matrix with the calibration matrix
    calibrate_pos_matrix( pos_matrices_queue[BSCAN_WINDOW-1], cal_matrix);

    insert_plane_points(pos_matrices_queue[BSCAN_WINDOW-1]);

    // Fill BPlane Eq
    insert_plane_eq();

    // Fill voxels
    fill_voxels();

    // Fill Holes
    // TODO

    // Update output volume. Copy GPU buffers to vtkImage buffer.
    // This takes time, so avoid this if possible.
    update_output_volume();
  }
}

//--------------------------------------------------------------
void CudaReconstruction::GetOutputVolume(vtkImageData *v)
{
  v->DeepCopy( reconstructedvolume );
  v->Modified();
}

//--------------------------------------------------------------
void CudaReconstruction::GetOrigin(double *ptr)
{
  ptr[0] = (double)volume_origin[0];
  ptr[1] = (double)volume_origin[1];
  ptr[2] = (double)volume_origin[2];
}

//--------------------------------------------------------------
double * CudaReconstruction::GetSpacing()
{
  double _volume_spacing[3] = {(double)volume_spacing,
                               (double)volume_spacing,
                               (double)volume_spacing
                              };
  return _volume_spacing;
}

//--------------------------------------------------------------
int * CudaReconstruction::GetOutputExtent()
{
  return volume_extent;
}

//--------------------------------------------------------------
void CudaReconstruction::GetOutputExtent(int * ptr)
{
  memcpy(ptr, volume_extent, sizeof(int)*6);
}

//--------------------------------------------------------------
char* CudaReconstruction::file2string(const char* filename, const char* preamble, size_t* final_length)
{
  FILE * file_stream = NULL;
  size_t source_length;

  // open the OpenCL source code file
  file_stream = fopen(filename, "rb");
  if(file_stream == 0)
  {
    return NULL;
  }

  size_t preamble_length = strlen(preamble);

  // get the length of the source code
  fseek(file_stream, 0, SEEK_END);
  source_length = ftell(file_stream);
  fseek(file_stream, 0, SEEK_SET);

  // allocate a buffer for the source code string and read it in
  char* source_str = (char *)malloc(source_length + preamble_length + 1);
  memcpy(source_str, preamble, preamble_length);
  if (fread((source_str) + preamble_length, source_length, 1, file_stream) != 1)
  {
    fclose(file_stream);
    free(source_str);
    return 0;
  }

  // close the file and return the total length of the combined (preamble + source) string
  fclose(file_stream);
  if(final_length != 0)
  {
    *final_length = source_length + preamble_length;
  }
  source_str[source_length + preamble_length] = '\0';

  return source_str;
}

//--------------------------------------------------------------
cl_kernel CudaReconstruction::ocl_kernel_build(cl_program program, cl_device_id device, char * kernel_name)
{
  cl_int err;
  cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
  if (err != CL_SUCCESS)
  {
    size_t len;
    char buffer[2048];
    std::cout << "ERROR: Failed to build program executable: " << kernel_name << std::endl;

    clGetProgramBuildInfo(
      program,              // the program object being queried
      device,            // the device for which the OpenCL code was built
      CL_PROGRAM_BUILD_LOG, // specifies that we want the build log
      sizeof(buffer),       // the size of the buffer
      buffer,               // on return, holds the build log
      &len);                // on return, the actual size in bytes of the data returned

    std::cout << buffer << std::endl;
    exit(1);
  }
  return kernel;

}

//--------------------------------------------------------------
void CudaReconstruction::initialize_buffers()
{
  x_vector_queue = (float4 *) malloc(BSCAN_WINDOW*sizeof(float4));
  y_vector_queue = (float4 *) malloc(BSCAN_WINDOW*sizeof(float4));
  bscans_queue = new unsigned char*[BSCAN_WINDOW];
  pos_matrices_queue = new float*[BSCAN_WINDOW];

  for(int i=0; i<BSCAN_WINDOW; i++)
  {
    pos_matrices_queue[i] = (float *)malloc(sizeof(float)*12);
    bscans_queue[i]     = (unsigned char *)malloc(bscan_w * bscan_h * sizeof(unsigned char));
  }

  bscan_timetags_queue = (float *) malloc(BSCAN_WINDOW * sizeof(float));
  pos_timetags_queue = (float *) malloc(BSCAN_WINDOW * sizeof(float));
  bscan_plane_equation_queue = (float4 *) malloc(BSCAN_WINDOW*sizeof(float4));
  plane_points_queue = (plane_pts *) malloc(BSCAN_WINDOW*sizeof(plane_pts));
  mask = (unsigned char *)malloc(sizeof(unsigned char)*bscan_w*bscan_h);
  volume = (unsigned char *) malloc(sizeof(unsigned char)*volume_width*volume_height*volume_depth);
  memset(volume, 0, sizeof(unsigned char)*volume_width*volume_height*volume_depth);

  reconstructedvolume = vtkSmartPointer< vtkImageData >::New();
  reconstructedvolume->SetOrigin( (double)volume_origin[0], (double)volume_origin[1], (double)volume_origin[2]  );
  reconstructedvolume->SetSpacing( (double)volume_spacing, (double)volume_spacing, (double)volume_spacing);
  reconstructedvolume->SetDimensions( volume_width, volume_height, volume_depth);
  reconstructedvolume->SetExtent( volume_extent );
  reconstructedvolume->AllocateScalars( VTK_UNSIGNED_CHAR, 1);
  memset(reconstructedvolume->GetScalarPointer(0,0,0), 0, sizeof(unsigned char)*volume_width*volume_height*volume_depth);
  reconstructedvolume->Modified();
}

//--------------------------------------------------------------
cl_mem CudaReconstruction::ocl_create_buffer(cl_context context, cl_mem_flags flags, size_t size, void * host_data)
{
  if (host_data != NULL)
  {
    flags |= CL_MEM_COPY_HOST_PTR;
  }
  cl_int err;
  cl_mem dev_mem = clCreateBuffer(context, flags, size, host_data, &err);
  if (err != CL_SUCCESS)
  {
    std::cout << "ERROR clCreateBuffer of size " << size << ":" << " " << err << std::endl;
    exit(err);
  }
  std::cout << "clCreateBuffer of " << size << " bytes(" << size/1024.0f/1024.0f << " MB)" << std::endl;
  return dev_mem;
}

//--------------------------------------------------------------
void CudaReconstruction::ocl_check_error(int err, char * info)
{
  if (err != CL_SUCCESS)
  {
    std::cout << "ERROR  " << info << ": " << err << std::endl;
    exit(err);
  }
}

//--------------------------------------------------------------
int CudaReconstruction::shift_queues()
{
  // Shift it to left
  for (int i = 0; i < BSCAN_WINDOW-1; i++)
  {
    x_vector_queue[i] = x_vector_queue[i+1];
    y_vector_queue[i] = y_vector_queue[i+1];
    bscans_queue[i] = bscans_queue[i+1];
    pos_matrices_queue[i] = pos_matrices_queue[i+1];
    bscan_timetags_queue[i] = bscan_timetags_queue[i+1];
    pos_timetags_queue[i] = pos_timetags_queue[i+1];
    bscan_plane_equation_queue[i] = bscan_plane_equation_queue[i+1];
    plane_points_queue[i] = plane_points_queue[i+1];
  }

  // Grab frame and insert it
  grab_input_data();

  if( timestamp_queue.size() < BSCAN_WINDOW )  // Queues are not full
  {
    // Multiply the position matrix with the calibration matrix
    calibrate_pos_matrix( pos_matrices_queue[BSCAN_WINDOW-1], cal_matrix);

    insert_plane_points(pos_matrices_queue[BSCAN_WINDOW-1]);

    // Fill BPlane Eq
    insert_plane_eq();

    return 0;
  }
  else
  {
    // Remove the elements from the top level data queues
    timestamp_queue.pop();
    imageData_queue.pop();
    poseData_queue.pop();

    return 1;
  }
}

//--------------------------------------------------------------
void CudaReconstruction::grab_input_data()
{
  timestamp_queue.push( timestamp );
  imageData_queue.push( imageData );
  poseData_queue.push( poseData );

  // Set this element to zero
  memset(pos_matrices_queue[BSCAN_WINDOW-1], '\0', sizeof(float)*12);
  memset(bscans_queue[BSCAN_WINDOW-1], '\0', sizeof(unsigned char)*bscan_w * bscan_h );

  // Convert to a floating point array
  double *matrixPtr = poseData_queue.front()->Element[0];
  float *matrixPtr2 = new float[12];
  matrixPtr2[0] = (float)matrixPtr[0];
  matrixPtr2[1] = (float)matrixPtr[1];
  matrixPtr2[2] = (float)matrixPtr[2];
  matrixPtr2[3] = (float)matrixPtr[3];
  matrixPtr2[4] = (float)matrixPtr[4];
  matrixPtr2[5] = (float)matrixPtr[5];
  matrixPtr2[6] = (float)matrixPtr[6];
  matrixPtr2[7] = (float)matrixPtr[7];
  matrixPtr2[8] = (float)matrixPtr[8];
  matrixPtr2[9] = (float)matrixPtr[9];
  matrixPtr2[10] = (float)matrixPtr[10];
  matrixPtr2[11] = (float)matrixPtr[11];

  unsigned char *imgDataPtr = (unsigned char *)imageData_queue.front()->GetScalarPointer(0, 0, 0);

  // Copy data to buffers
  bscan_timetags_queue[BSCAN_WINDOW-1] = timestamp_queue.front();
  pos_timetags_queue[BSCAN_WINDOW-1]   = timestamp_queue.front();
  memcpy( pos_matrices_queue[BSCAN_WINDOW-1], matrixPtr2, sizeof(float)*12);
  memcpy( bscans_queue[BSCAN_WINDOW-1], imgDataPtr, sizeof(unsigned char)*bscan_h*bscan_w);

  delete matrixPtr2;
  // TODO: Interpolate the pos matrix to the timetag of the bscan
}

//--------------------------------------------------------------
void CudaReconstruction::update_output_volume()
{
  // Copy volume to outptVolume
  memcpy( (unsigned char *)this->reconstructedvolume->GetScalarPointer( 0, 0, 0), volume, sizeof(unsigned char)*volume_width*volume_height*volume_depth);
}

//--------------------------------------------------------------
void CudaReconstruction::dump_matrix(int r, int c, float * mat)
{
  std::cout << "Content of the matrix.. " << std::endl;
  for(int i=0; i<r; i++)
  {
    for(int j=0; j<c; j++)
    {
      std::cout << mat[i*c + j] << " ";
    }

    std::cout << std::endl;
  }
}

//--------------------------------------------------------------
void CudaReconstruction::calibrate_pos_matrix(float * pos_matrix, float * cal_matrix)
{
  // Multiply cal_matrix into pos_matrix

  float * new_matrix = (float *) malloc(sizeof(float)*12);
  for (int b = 0; b < 3; b++)
    for (int c = 0; c < 4; c++)
    {
      float sum = 0;
      for (int k = 0; k < 3; k++)
      {
        sum += pos_matrix[b*4 + k]*cal_matrix[k*4 + c];
      }

      if(c ==3 )
      {
        sum += pos_matrix[b*4 + 3];
      }

      new_matrix[b*4 + c] = sum;
    }

  memcpy(pos_matrix, new_matrix, 12*sizeof(float));
  free(new_matrix);
}

//--------------------------------------------------------------
void CudaReconstruction::insert_plane_points(float * pos_matrix)
{
  // Fill plane_points
  plane_points_queue[BSCAN_WINDOW-1].corner0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  plane_points_queue[BSCAN_WINDOW-1].cornerx = make_float4(bscan_w*bscan_spacing_x, 0.0f, 0.0f, 0.0f);
  plane_points_queue[BSCAN_WINDOW-1].cornery = make_float4(0.0f, bscan_h*bscan_spacing_y, 0.0f, 0.0f);

  // Transform plane_points
  float4 * foo = (float4 *) &plane_points_queue[BSCAN_WINDOW-1];
  float4 _volume_orig = {volume_origin[0], volume_origin[1], volume_origin[2]};
  float * sums = (float *) malloc(sizeof(float)*3);
  for (int i = 0; i < 3; i++)
  {
    for (int y = 0; y < 3; y++)
    {
      float sum = 0;
      sum += pos_matrix_a(0,y)*foo[i].x;
      sum += pos_matrix_a(1,y)*foo[i].y;
      sum += pos_matrix_a(2,y)*foo[i].z;
      sum += pos_matrix_a(3,y);
      sums[y] = sum;
    }
    memcpy(&foo[i], sums, 3*sizeof(float));
    foo[i] = foo[i] - _volume_orig;
  }
}

//--------------------------------------------------------------
void CudaReconstruction::insert_plane_eq()
{
  // Fill bscan_plane_equation
  float4 a = plane_points_queue[BSCAN_WINDOW-1].corner0;
  float4 b = plane_points_queue[BSCAN_WINDOW-1].cornerx;
  float4 c = plane_points_queue[BSCAN_WINDOW-1].cornery;
  float4 normal = normalize(cross(a-b, c-a));

  bscan_plane_equation_queue[BSCAN_WINDOW-1].x = normal.x;
  bscan_plane_equation_queue[BSCAN_WINDOW-1].y = normal.y;
  bscan_plane_equation_queue[BSCAN_WINDOW-1].z = normal.z;
  bscan_plane_equation_queue[BSCAN_WINDOW-1].w = -normal.x*a.x - normal.y*a.y - normal.z*a.z;

  x_vector_queue[BSCAN_WINDOW-1] = normalize(plane_points_queue[BSCAN_WINDOW-1].cornerx - plane_points_queue[BSCAN_WINDOW-1].corner0);
  y_vector_queue[BSCAN_WINDOW-1] = normalize(plane_points_queue[BSCAN_WINDOW-1].cornery - plane_points_queue[BSCAN_WINDOW-1].corner0);
}

//--------------------------------------------------------------
void CudaReconstruction::fill_voxels()
{
  int axis = 2; // Use axis 2 ( along Z axis )
  int intersection_counter = find_intersections(axis);

  omp_set_lock(&lock);
  ocl_check_error(clEnqueueWriteBuffer(reconstruction_cmd_queue, dev_x_vector_queue, CL_TRUE, 0, dev_x_vector_queue_size, x_vector_queue, 0, 0, 0));
  omp_unset_lock(&lock);

  omp_set_lock(&lock);
  ocl_check_error(clEnqueueWriteBuffer(reconstruction_cmd_queue, dev_y_vector_queue, CL_TRUE, 0, dev_y_vector_queue_size, y_vector_queue, 0, 0, 0));
  omp_unset_lock(&lock);

  omp_set_lock(&lock);
  ocl_check_error(clEnqueueWriteBuffer(reconstruction_cmd_queue, dev_plane_points_queue, CL_TRUE, 0, dev_plane_points_queue_size, plane_points_queue, 0, 0, 0));
  omp_unset_lock(&lock);

  omp_set_lock(&lock);
  ocl_check_error(clEnqueueWriteBuffer(reconstruction_cmd_queue, dev_mask, CL_TRUE, 0, mask_size, mask, 0, 0, 0));
  omp_unset_lock(&lock);

  unsigned char * temp = (unsigned char *) malloc(bscans_queue_size);
  for (int i = 0; i < BSCAN_WINDOW; i++)
  {
    memcpy(&temp[bscan_w*bscan_h*i], bscans_queue[i], sizeof(unsigned char)*bscan_w*bscan_h);
  }

  omp_set_lock(&lock);
  ocl_check_error(clEnqueueWriteBuffer(reconstruction_cmd_queue, dev_bscans_queue, CL_TRUE, 0, bscans_queue_size, temp, 0, 0, 0));
  omp_unset_lock(&lock);
  free(temp);

  omp_set_lock(&lock);
  ocl_check_error(clEnqueueWriteBuffer(reconstruction_cmd_queue, dev_bscan_timetags_queue, CL_TRUE, 0, bscan_timetags_queue_size, bscan_timetags_queue, 0, 0, 0));
  omp_unset_lock(&lock);

  clSetKernelArg(adv_fill_voxels, 0, sizeof(cl_mem), &dev_intersections);
  clSetKernelArg(adv_fill_voxels, 1, sizeof(cl_mem), &dev_volume);
  clSetKernelArg(adv_fill_voxels, 2, sizeof(cl_float), &volume_spacing);
  clSetKernelArg(adv_fill_voxels, 3, sizeof(cl_int), &volume_width);
  clSetKernelArg(adv_fill_voxels, 4, sizeof(cl_int), &volume_height);
  clSetKernelArg(adv_fill_voxels, 5, sizeof(cl_int), &volume_depth);
  clSetKernelArg(adv_fill_voxels, 6, sizeof(cl_mem), &dev_x_vector_queue);
  clSetKernelArg(adv_fill_voxels, 7, sizeof(cl_mem), &dev_y_vector_queue);
  clSetKernelArg(adv_fill_voxels, 8, sizeof(cl_mem), &dev_plane_points_queue);
  clSetKernelArg(adv_fill_voxels, 9, sizeof(cl_mem), &dev_bscan_plane_equation_queue);
  clSetKernelArg(adv_fill_voxels, 10, sizeof(cl_float), &bscan_spacing_x);
  clSetKernelArg(adv_fill_voxels, 11, sizeof(cl_float), &bscan_spacing_y);
  clSetKernelArg(adv_fill_voxels, 12, sizeof(cl_int), &bscan_w);
  clSetKernelArg(adv_fill_voxels, 13, sizeof(cl_int), &bscan_h);
  clSetKernelArg(adv_fill_voxels, 14, sizeof(cl_mem), &dev_mask);
  clSetKernelArg(adv_fill_voxels, 15, sizeof(cl_mem), &dev_bscans_queue);
  clSetKernelArg(adv_fill_voxels, 16, sizeof(cl_mem), &dev_bscan_timetags_queue);
  clSetKernelArg(adv_fill_voxels, 17, sizeof(cl_int), &intersection_counter);

  omp_set_lock(&lock);
  ocl_check_error(clEnqueueNDRangeKernel(reconstruction_cmd_queue, adv_fill_voxels, 1, NULL, global_work_size, local_work_size, NULL, NULL, NULL));
  omp_unset_lock(&lock);

  // Readout the device volume to local buffer
  omp_set_lock(&lock);
  ocl_check_error(clEnqueueReadBuffer(reconstruction_cmd_queue, dev_volume, CL_TRUE, 0, volume_size, volume, 0, 0, 0));
  omp_unset_lock(&lock);
}

//--------------------------------------------------------------
int CudaReconstruction::find_intersections(int axis)
{
  omp_set_lock(&lock);
  ocl_check_error(clEnqueueWriteBuffer(reconstruction_cmd_queue, dev_bscan_plane_equation_queue, CL_TRUE, 0, bscan_plane_equation_queue_size, bscan_plane_equation_queue, 0, 0, 0));
  omp_unset_lock(&lock);

  clSetKernelArg(trace_intersections, 0, sizeof(cl_mem), &dev_intersections);
  clSetKernelArg(trace_intersections, 1, sizeof(cl_int), &volume_width);
  clSetKernelArg(trace_intersections, 2, sizeof(cl_int), &volume_height);
  clSetKernelArg(trace_intersections, 3, sizeof(cl_int), &volume_depth);
  clSetKernelArg(trace_intersections, 4, sizeof(cl_float), &volume_spacing);
  clSetKernelArg(trace_intersections, 5, sizeof(cl_mem), &dev_bscan_plane_equation_queue);
  clSetKernelArg(trace_intersections, 6, sizeof(cl_int), &axis);
  omp_set_lock(&lock);
  ocl_check_error(clEnqueueNDRangeKernel(reconstruction_cmd_queue, trace_intersections, 1, NULL, global_work_size, local_work_size, NULL, NULL, NULL));
  omp_unset_lock(&lock);

  return max_vol_dim*max_vol_dim;
}