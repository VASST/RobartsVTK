#ifndef API_EXCEPTION_H
#define API_EXCEPTION_H

#include <iostream>

class API_exception
{
public:
	API_exception();
	const char *message; // Message to be sent.
	void print_message(); 
	const char *return_message() {  return message;  }  
};

class serial_open_exception:public API_exception
{
public:
	serial_open_exception() {  message = "Unable to open serial port."; } // Constructor
	~serial_open_exception() {} // Destructor
};

class serial_close_exception:public API_exception
{
public:
	serial_close_exception() {  message = "Unable to close serial port."; } // Constructor
	~serial_close_exception() {} // Destructor
};

class serial_baudrate_exception:public API_exception
{
public:
	serial_baudrate_exception() {  message = "Invalid serial port baud rate."; } // Constructor
	~serial_baudrate_exception() {} // Destructor
};

class serial_timeout_exception:public API_exception
{
public:
	serial_timeout_exception() {  message = "Problem setting serial time outs."; } // Constructor
	~serial_timeout_exception() {} // Destructor
};

class serial_dcb_exception:public API_exception
{
public:
	serial_dcb_exception() {  message = "Problem setting DCB variable.";  } // Constructor
	~serial_dcb_exception() {} // Destructor
};

class serial_read_exception:public API_exception
{
public:
	serial_read_exception() {  message = "Problem reading from serial port.";  } // Constructor
	~serial_read_exception() {} // Destructor
};

class serial_write_exception:public API_exception
{
public:
	serial_write_exception() {  message = "Problem writing to serial port.";  } // Constructor
	~serial_write_exception() {} // Destructor
};

class invalid_data_exception:public API_exception
{
public:
	invalid_data_exception() {  message = "Invalid serial port baud rate.";  } // Constructor
	~invalid_data_exception() {} // Destructor
};

#endif



