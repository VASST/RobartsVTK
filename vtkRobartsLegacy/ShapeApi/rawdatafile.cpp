#include "stdafx.h"
#include "rawdatafile.h"

FILE *file; // Data file pointer.
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

rawdatafile::rawdatafile(int num_sensors)
{
  num_sense = num_sensors;
  // create a string for scanning the raw data set 
}

rawdatafile::~rawdatafile()
{
  if (file != NULL)
    fclose(file);
}

int rawdatafile::openFile(char *file_name, int mode)
{
  current_index = 0;
  
  if (mode == DATA_READ)
    file = fopen(file_name, "r");
  else if (mode == DATA_WRITE)
    file = fopen(file_name, "w+t");
  else
    return 0;
  if (file == NULL)
    return 0;
  else
    return 1;
}

int rawdatafile::closeFile()
{
  if (file != NULL)
  {
    fclose(file);
    return 1;
  }
  else
    return 0;
}

int rawdatafile::write(int frame_number, double time_stamp, int raw_data[])
{
  char frame[306];// calculated max of 306 chars / frame
  int num_written, index;
  index = 0;
  
  // fill the frame buffer with the appropriate data.
  index = sprintf(frame, "%d ", frame_number);
  index += sprintf(frame+index, "%f ", time_stamp);

  for (int i = 0; i < num_sense; i++)
    index += sprintf(frame+index, "%d ", raw_data[i]);

  index += sprintf(frame+index, "\n");

  // write the buffer to file.
  num_written = fwrite(frame, 1, index, file); 

  if (num_written == index)
    return 1;
  else
    return 0;
}

int rawdatafile::read(int &frame_number, double &time_stamp, int raw_data[])
{
  char buffer[2048];
  int num_read, index;
  index = 0;
  frame_number = 0;

  // start at the current position in the file.
  // uses the current_index variable to hold this position.
  fseek(file, current_index, SEEK_SET);

  // read from the file to the buffer.
  num_read = fread(buffer, sizeof( char ), 2048, file);

  while (index<num_read&&(buffer[index]==' '||buffer[index]==','||buffer[index]=='\t'||buffer[index]=='\r'||buffer[index]=='\n')) index++;
  // parse the buffer for the frame data.
  sscanf(&buffer[index], "%d", &frame_number);
  while (index<num_read&&buffer[index]>='.') index++;
  while (index<num_read&&buffer[index]==' '||buffer[index]==','||buffer[index]=='\t') index++;
  sscanf(&buffer[index], "%lf", &time_stamp);

  for (int i = 0; i < num_sense; i++) {
    while (index<num_read&&buffer[index]>='.') index++;
    while (index<num_read&&buffer[index]==' '||buffer[index]==','||buffer[index]=='\t') index++;
    sscanf(&buffer[index], "%d", &raw_data[i]);
  }

  //read to end of line
  while (index<num_read&&buffer[index]!='\n'&&buffer[index]!='\r') index++;
  while (index<num_read&&(buffer[index]=='\n'||buffer[index]=='\r')) index++;

  current_index += index;
  return 1;
}

int rawdatafile::readtime(double target_time, int &frame_number, double &time_stamp, int raw_data[])
{
  char buffer[2048];
  int index=0;
  double time;
  
  frame_number = 0;
  time = 0.0;
  current_index=0;

  while (time < target_time) // exit when the 
  {
    // start at the current position in the file.
    // uses the current_index variable to hold this position.
    fseek(file, current_index, SEEK_SET);
    
    // read from the file to the buffer.
    int numRead=fread(buffer, sizeof( char ), 2048, file);
    if (!numRead) return 0;
    index=0;
    //skip past initial CR's, LF's,  commas and white space
    while (index<numRead&&(buffer[index]==' '||buffer[index]==','||buffer[index]=='\t'||buffer[index]=='\n'||buffer[index]=='\r')) index++;
    // parse the buffer for the frame data.
    sscanf(&buffer[index], "%d", &frame_number);
    //skip past frame data
    while (index<numRead&&buffer[index]>='.') index++;
    //skip past commas and white space
    while (index<numRead&&(buffer[index]==' '||buffer[index]==','||buffer[index]=='\t')) index++;
    sscanf(&buffer[index], "%lf", &time);
    //scan to end of line
    while (index<numRead&&buffer[index]!='\n'&&buffer[index]!='\r') index++;
    while (index<numRead&&(buffer[index]=='\n'||buffer[index]=='\r')) index++;
    current_index += index;
  }
  current_index-=index;
  read(frame_number,time_stamp,raw_data);
  return 1;
}

int rawdatafile::readframe(int target_frame, int &frame_number, double &time_stamp, int raw_data[])
{
  char buffer[2048];
  int index=0;
  int frame=0;
  frame_number = 0;
  current_index=0;
  
  while (frame != target_frame) // exit when the 
  {
    // start at the current position in the file.
    // uses the current_index variable to hold this position.
    fseek(file, current_index, SEEK_SET);
    
    // read from the file to the buffer.
    int numRead = fread(buffer, sizeof( char ), 2048, file);
    if (!numRead) return 0;
    index=0;
    //skip past initial CR's, LF's, commas and white space
    while (index<numRead&&(buffer[index]==' '||buffer[index]==','||buffer[index]=='\t'||buffer[index]=='\r'||buffer[index]=='\n')) index++;
    // parse the buffer for the frame data.
    sscanf(&buffer[index], "%d", &frame);
    //continue to end of line
    while (index<numRead&&buffer[index]!='\r'&&buffer[index]!='\n') index++;
    while (index<numRead&&(buffer[index]=='\r'||buffer[index]=='\n')) index++;
    current_index += index; 
  }
  current_index-=index;
  read(frame_number,time_stamp,raw_data);
  return 1;
}

int rawdatafile::getNumDigits(int data)
{
  int nDigits=0;
  char datastring[20];
  sprintf(datastring,"%d",data);
  while (datastring[nDigits]!=0) nDigits++;
  return nDigits;
}

int rawdatafile::findIndexes(char *databuffer, int buffersize, int indexes[]) //returns the index in filedata where the description string is first found following start_index
{
  int bufferpos=0;
  int index = 0;

  // find the first two spaces and new line character the on the data buffer 
  // and put them in indexes[].
  do
  {
    if (bufferpos<buffersize)
    {
      if (databuffer[bufferpos] == ' ')
      {
        indexes[index] = bufferpos;
        index ++;
      }
    }
    else 
      return 0; // could not find the spaces.

    bufferpos ++;
  } while (databuffer[bufferpos] != '\n');

  indexes[index] = bufferpos;

  return 1; // found all spaces and .
}

