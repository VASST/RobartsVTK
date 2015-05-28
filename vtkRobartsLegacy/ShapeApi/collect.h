#ifndef COLLECT_H
#define COLLECT_H

// Collect class inherits the serial class to read from
// to the serial port.  The purpose of this class is to 
// handle the complexities of the shape tape communication
// protocol.  For now, only the pollTape function is used.
// Written by Jordan Lutes
// for Measurand Inc. 30/06/2000 

// Modified to no longer need the serial routine to function.
// In the new version, the data from the tape is passed to this
// object from an outside source.  (allows for serial, network, ...)
// Also added support for the input ports.  Two buttons and one 
// trigger status variable.
// 13/10/2000

//#include <filedata.h>
#include "serial.h"
#include "shapeAPI.h"
#include "rawdatafile.h"

class SHAPEAPI_API collect
{
public:
  void SetReadMode(bool bDirect);
  collect(int num_sense, int baud_rate); // Constructor.
  collect(char* config_file); //
  ~collect(); // Destructor.
  
  // Set raw data file.
  setRawDataFile(char *file_name);

  // Collect a set of raw data from the shape tape.
  bool pollTape(int data[]);
  bool pollTape(int data[], int frame, double time);
  
  // Set the current data buffer for pollTape to read from.
  // Returns 1 if successful and 0 if not
  setBuffer(unsigned char message[117]);
  

  // Get the current status of the input ports.
  int getSwitch0();
  int getSwitch1();
  int getTrigger();

  // Change the direct read switch.
  //void setReadMethod(bool read_method);

private:
  int type;
  int num_sensors;
  unsigned char buffer[117]; // Max. size of single tape read.
  int buffer_index;
  // Similar to serial read, although from local buffer.
  readFromBuffer(int number_of_bytes, unsigned char* message);
  reset();

  // input port variables.
  int switch0, switch1, trigger;

  // Raw data file - if used. (set baud_rate = 0)
  //filedata raw_data_file;
  char *data_file_name;  
  // the rawdatafile object.  This class enables the writing to
  // / reading from raw data files. 
  rawdatafile *data_file;

  // Boolean switch to determine if the polltape function is 
  // to read from an input buffer or serial port. if direct_read
  // is true then the system reads from the serial port, if not,
  // it attempts to read from the input buffer.  The default is to 
  // read from the input buffer.
  bool direct_read; 

  // Serial object to write to / read from serial ports if the above
  // direct_read flag is set to true.
  serial *port;


};
#endif
