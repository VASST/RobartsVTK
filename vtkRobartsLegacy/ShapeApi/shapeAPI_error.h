#ifndef ERROR_BASE

#define ERROR_BASE 20000
// Error 20001 set if the baud rate is not set to either 57600, 115200, or 0 
// (if data read from file).
#define SERIAL_BAUDRATE_ERROR ERROR_BASE+1
// Error 20002 set if the system has problem closing the serial port connection. 
#define SERIAL_CLOSE_ERROR ERROR_BASE+2
// Error 20003 set is there is an error while setting the DCB variable for the 
// serial connection.
#define SERIAL_DCB_ERROR ERROR_BASE+3
// Error 20004 set if the system has trouble opening the serial port.
#define SERIAL_OPEN_ERROR ERROR_BASE+4
// Error 20005 set if there is a problem reading the serial port.
#define SERIAL_READ_ERROR ERROR_BASE+5
// Error 20006 set if the serial timeout variable cannot be set correctly.
#define SERIAL_TIMEOUT_ERROR ERROR_BASE+6
// Error 20007 set if there is a problem writing to the serial connection.
#define SERIAL_WRITE_ERROR ERROR_BASE+7
// Error 20008 set if the serial number is not betwen 0 and 65535 (16 bit integer).
#define INVALID_SERIAL_NUMBER ERROR_BASE+8
// Error 20009 set if a boolean value is not 0 or 1.
#define INVALID_BOOLEAN_VALUE ERROR_BASE+9
// Error 20010 set if the interpolation interval is less than 1.
#define INVALID_INTERPOLATION_VALUE ERROR_BASE+10
// Error 20011 set if the num_regions variable is not 0 or less and not greater 
// then the number of sensors.
#define INVALID_NUM_REGIONS_VARIABLE ERROR_BASE+11
// Error 20012 set if the length is less tahn 0.
#define NEGATIVE_TAPE_LENGTH ERROR_BASE+12
// Error 20013 set if the number of sensors is set to less than 0 or greater than 
// 48.
#define INVALID_NUMBER_OF_SENSORS ERROR_BASE+13
// Error 20014 set if the sum of the region lengths is greater then the length of 
// the tape or a region has a negative value.
#define INVALID_REGION_LENGTH ERROR_BASE+14
// Error 20015 set if the interpolation interval is set to 0 or less.
#define NEGATIVE_INTERPOLATION_VALUE ERROR_BASE+15

#endif

