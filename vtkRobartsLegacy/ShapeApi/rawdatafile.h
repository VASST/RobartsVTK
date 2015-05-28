#ifndef RAW_DATA_FILE
#define RAW_DATA_FILE

#define DATA_READ 0
#define DATA_WRITE 1

#define TRUE 1
#define FALSE 0

//#include <fstream>
#include <stdio.h>
#include "shapeAPI.h"

class SHAPEAPI_API rawdatafile
{
public:
  rawdatafile(int num_sensors);
  ~rawdatafile();

  int openFile(char *file_name, int mode);
  int closeFile();

  // Write function - write a data frame at the end of the data file.
  int write(int frame_number, double time_stamp, int raw_data[]);

  // Read functions
  // Read next frame ... returns 1 if sucessful 0 if not.
  int read(int &frame_number, double &time_stamp, int raw_data[]);
  // Read a specific time ... returns 1 if sucessful 0 if not.
  int readtime(double target_time, int &frame_number, double &time_stamp, int raw_data[]);
  // Read a specific frame ... returns 1 if sucessful 0 if not.
  int readframe(int target_frame, int &frame_number, double &time_stamp, int raw_data[]);

private:
  int findIndexes(char *databuffer, int buffersize, int indexes[]); //returns the index in filedata where the description string is first found following start_index
  int getNumDigits(int data);
  int num_sense;
  int current_index;
};
#endif
/*
  //record raw data for test purposes
  FILE *testFile;
  testFile = fopen("test.txt","wt");
  char caldata[1024];
  sprintf(caldata,"pure_bend_up_tiny_top: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",pure_bend_up_tiny[0],pure_bend_up_tiny[2],
    pure_bend_up_tiny[4],pure_bend_up_tiny[6],pure_bend_up_tiny[8],pure_bend_up_tiny[10],pure_bend_up_tiny[12],
    pure_bend_up_tiny[14],pure_bend_up_tiny[16],pure_bend_up_tiny[18],pure_bend_up_tiny[20],pure_bend_up_tiny[22],
    pure_bend_up_tiny[24],pure_bend_up_tiny[26],pure_bend_up_tiny[28],pure_bend_up_tiny[30]);
  
  fwrite(caldata,1,strlen(caldata),testFile);
*/