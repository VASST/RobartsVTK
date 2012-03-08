#ifndef SERIAL_H
#define SERIAL_H

// This is a serial port class used to serparate the 
// windows code from the ANSI C++.
// Written by Jordan Lutes
// for Measurand Inc. 30/06/2000

#include <windows.h>
#include <cstdio>
#include <iostream>
#include "shapeAPI.h"

class SHAPEAPI_API serial
{
public:
	serial(char* com_port, int baud_rate, int num_sensors, int serial_num); // Constructor.
	serial(char* config_file);
	~serial(); // Destructor.

	// writes/reads the number_of_bytes to/from the port
	// returns the number of bytes sucessful.
	int read(int number_of_bytes, unsigned char* message);
	int write(int number_of_bytes, unsigned char* message);

	// Sets the communications to wait or not for a trigger
	void setTrigger(bool trigger);

	// Reads a single frome from a shape tape.
	int readBuffer(unsigned char message[117]);

private:
	void initialize(char* com_port, int num_sensors, int serial_num);
	// windows handle.
	HANDLE hPort;
	int buffer_size;
	int baud;
	bool extTrigger;
};

#endif; // serial class