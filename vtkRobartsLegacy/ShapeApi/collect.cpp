#define TRUE 1
#define FALSE 0

#include "stdafx.h"
#include "windows.h"
#include <stdio.h>
#include <iostream>
#include "collect.h"
//#include "API_exception.h"
#include "shapeAPI_error.h"
#include "filedata.h"

/*
Name:		collect
Purpose:	constructor
Accepts:	num_sense = the number of sensors on the tape.
			baud_rate = the serial speed of the tapes (has led near serial connection, then 115200, else 57600).
Returns:    void
*/
collect::collect(int num_sense, int baud_rate) // Constructor.
{
	data_file = NULL;
	num_sensors = num_sense;
	if (baud_rate == 57600)
		type = 0;
	else if (baud_rate == 115200)
		type = 1;
	else if (baud_rate == 0)
	{
		type = 2;
		data_file = new rawdatafile(num_sensors);
	}
	else
		SetLastError(SERIAL_BAUDRATE_ERROR);
		//throw serial_baudrate_exception();

	// Initialize the buffer to 0's
	for (int i = 0 ; i < 113; i++)
		buffer[i] = 0;

	// Initialize the buffer index.
	buffer_index = 0;

	direct_read = false;
	port = NULL;
}
	
/*
Name:		collect
Purpose:	constructor
Accepts:	config_file = file name for the tape settings file. Contains all the variables of the
			above constructor in the filedata file format.
Returns:    void
*/
collect::collect(char* config_file)
{
	int baud_rate;
	int temp_direct_read;
	char *com_port;
	int serial_num;
	filedata file(config_file);

	num_sensors = file.getInteger("[settings]","number of sensors");
	baud_rate = file.getInteger("[settings]","baud rate");
	file.getString("[settings]","com port", &com_port);
	serial_num = file.getInteger("[settings]","serial number");
	temp_direct_read = file.getInteger("[settings]","direct read");

	data_file = NULL;

	if (baud_rate == 57600)
		type = 0;
	else if (baud_rate == 115200)
		type = 1;
	else if (baud_rate == 0)
	{
		type = 2;
		file.getString("[config]","data file",&data_file_name);
		data_file = new rawdatafile(num_sensors);
		data_file->openFile(data_file_name, DATA_READ);
	}
	else
		SetLastError(SERIAL_BAUDRATE_ERROR);
		//throw serial_baudrate_exception();

	// Initialize the buffer to 0's
	for (int i = 0 ; i < 113; i++)
		buffer[i] = 0;

	// Initialize the buffer index.
	buffer_index = 0;

	// convert the integer flag to a bool
	if (temp_direct_read == 0)
		direct_read = false;
	else if (temp_direct_read == 1)
		direct_read = true;
	else 
		direct_read = false;

	port = NULL;

	if (direct_read == true)
		port = new serial(com_port, baud_rate, num_sensors, serial_num);
}

/*
Name:		~collect
Purpose:	destructor
Accepts:	void
Returns:    void
*/
collect::~collect() // Destructor.
{
	if (port != NULL)
		delete port;
	if (data_file != NULL)
		delete data_file;
}

/*
Name:		collect
Purpose:	constructor
Accepts:	config_file = file name for the tape settings file. Contains all the variables of the
			above constructor in the filedata file format.
Returns:    void
*/
collect::setRawDataFile(char *file_name)
{
	data_file_name = file_name;
	data_file->openFile(data_file_name, DATA_READ);
}

/*
Name:		collect
Purpose:	constructor
Accepts:	config_file = file name for the tape settings file. Contains all the variables of the
			above constructor in the filedata file format.
Returns:    void
*/
bool collect::pollTape(int data[])
{
	int frame_number = 0;
	double time_stamp = 0;
	return pollTape(data, frame_number, time_stamp);
}

/*
Name:		collect
Purpose:	constructor
Accepts:	config_file = file name for the tape settings file. Contains all the variables of the
			above constructor in the filedata file format.
Returns:    void
*/
bool collect::pollTape(int data[], int frame, double time)
{
	// Grabs a set of raw data from the shapetape.
	int raw_data[48];
	int test = 0;
	unsigned char data_buffer[16];
	unsigned char BytesRead = 0;
	int num_led;
	num_led = num_sensors / 8;

	if (type == 0)// Type 0 is the old protocol.
	{
		for (int led = 1; led <= num_led; led++)
		{
			if (direct_read == true)
			{
				if (port != NULL)
					BytesRead = port->read(16,data_buffer);
			}
			else
				BytesRead = readFromBuffer(16,data_buffer);
			
			if (BytesRead != 16)
				return FALSE;
			// If there is a problem, quit now.
			
			// Now we are ready to translate the data from the buffer.
			for(int i=0 ; i < 8 ; i++)
			{
				// Each channel consists of two of the  BYTES in the buffer.
				// We'll combine the bytes into one WORD (int).

				raw_data[8*(led-1) + i] = (int) (data_buffer[2*i] << 8  |  data_buffer[2*i+1]);
			}
		}
	}
	else if(type == 1) // Type 1 is the new protocol. Nov. 99
	{
		unsigned char serial[2];
		unsigned char leds;
		unsigned char test;
		unsigned char check; // Sum calculated on the board.
		unsigned char sum; // Sum calculated on the computer.
		unsigned char counter[2];
		unsigned char timer[3];
		unsigned char filler[8];
		sum = 0;
		
		// Read in the number of leds on the tape.
		if (direct_read == true)
		{	
			unsigned char in_message;
			in_message = 0x20;

			if (port != NULL)
				port->write(1,&in_message);
				port->read(1,&leds);
		}
		else
			readFromBuffer(1,&leds);
		
		// Use masking to test the input ports in the high nibble of this byte.
		// Test for switch 0 in bit 4.
		test = leds;
		if (((test & 0x10) >> 4)==1)
			switch0 = TRUE;
		else
			switch0 = FALSE;
		
		// Test trigger status in bit 5.
		test = leds;
		if (((test & 0x20) >> 5)==1)
			trigger = TRUE;
		else
			trigger = FALSE;
		
		// Test for switch 1 in bit 6.
		test = leds;
		if (((test & 0x40) >> 6)==1)
			switch1 = TRUE;
		else
			switch1 = FALSE;
		
		int num = (int)(leds & 0x07);
			
		// Read in the serial number.
		if (direct_read == true)
		{	
			if (port != NULL)
				port->read(2,serial);
		}
		else
			readFromBuffer(2,serial);

		// Need to check BytesRead to see if it is correct for each read.
		int ser;
		ser = (int)(serial[0] << 8 | serial[1]);
        
		if (ser > 4096)
        {
		    // Read in the frame number.
			if (direct_read == true)
			{	
				if (port != NULL)
					port->read(2,counter);
			}
			else
				readFromBuffer(2,counter);

			frame = (int)(counter[0] << 8 | counter[1]);
		    
			// Read in the time.
            
			if (direct_read == true)
			{	
				if (port != NULL)
					port->read(3,timer);
			}
			else
				readFromBuffer(3,timer);

			time = (double)(timer[0] << 16 | timer[1] << 8 | timer[2]);
		    
			// Read in the 5's (filler) used later.
			if (direct_read == true)
			{
				if (port != NULL)
					port->read(8,filler);
			}
			else
				readFromBuffer(8,filler);	
		}
		for (int x = 0;x < num;x++)
		{		
			BytesRead = 0;// Reset the byte counter for the next read.
			
			// Read in one led worth of sensor data.
			if (direct_read == true)
			{
				if (port != NULL)
					BytesRead = port->read(16,data_buffer);
			}
			else	
				BytesRead = readFromBuffer(16,data_buffer);
			
			if (BytesRead != 16)
				return FALSE;
			// If there is a problem, quit now.

			// Now we are ready to translate the data from the buffer.
			for(int i=0 ; i < 8 ; i++)
			{
				// Each channel consists of two of the  BYTES in the buffer.
				// We'll combine the bytes into one WORD (int).
				raw_data[8*(x) + i] = (int) (data_buffer[2*i] << 8  |  data_buffer[2*i+1]);
				// Calculate the check sum value.
				sum += data_buffer[2*i];
				sum += data_buffer[2*i+1];
			}
		}
		
		BytesRead = 0;// Reset the byte counter for the next read.
		
		// Read in the checksum calculated on the board.
		if (direct_read == true)
		{
			if (port != NULL)
				port->read(1,&check);
		}
		else
			readFromBuffer(1,&check);
		
		if (sum != check)
		{
			return FALSE;
		}
	}
	else if(type == 2)
	{
		// Read in a frame of data from a raw data file.
		data_file->read(frame, time, data);
	}
	else
	{
		// Should never reach here unless baud rate is incorrect.
		return FALSE;
	}
	for (int i=0;i<num_sensors;i++)
	{
		data[i] = raw_data[i];
	}
	reset();
	return TRUE;
}

/*
Name:		collect
Purpose:	constructor
Accepts:	config_file = file name for the tape settings file. Contains all the variables of the
			above constructor in the filedata file format.
Returns:    void
*/
collect::setBuffer(unsigned char message[117])
{
	for (int i = 0; i < 117; i ++)
	{
		buffer[i] = message[i];
	}
	buffer_index = 0;
}

/*
Name:		readFromBuffer
Purpose:	Reads a known number of bytes from the tape buffer and moves the buffer_index to the 
			new position.
Accepts:	number_of_bytes = the number of bytes requested from the buffer.
			message         = the characters read from the buffer.
Returns:    the number of bytes read.
*/
int collect::readFromBuffer(int number_of_bytes, unsigned char* message)
{
	for (int i = 0; i < number_of_bytes; i++)
	{
		message[i] = buffer[i + buffer_index]; 
	}
	buffer_index += number_of_bytes;

	return number_of_bytes;
}

/*
Name:		getSwitch0
Purpose:	Gets the switch0 value.  The two switch values are built into the hardware of the
			serial shapetapes and are meant to ahndle buttons or switches placed at the end
			of the tape.
Accepts:	void
Returns:    a boolean value for switch0.
*/
int collect::getSwitch0()
{
	return switch0;
}

/*
Name:		getSwitch1
Purpose:	Gets the switch1 value.  The two switch values are built into the hardware of the
			serial shapetapes and are meant to ahndle buttons or switches placed at the end
			of the tape.
Accepts:	void
Returns:    a boolean value for switch1.
*/
int collect::getSwitch1()
{
	return switch1;
}

/*
Name:		getTrigger
Purpose:	Gets the trigger value.  The trigger value indicates whether the tape was external 
			triggered or not. 
Accepts:	void
Returns:    a boolean trigger value
*/
int collect::getTrigger()
{
	return trigger;
}

/*
Name:		reset
Purpose:	Resets the buffer index back to zero, effectly allowing a new buffer to be read from.
Accepts:	void
Returns:    void
*/
collect::reset()
{
	buffer_index = 0;
}

/*
Name:		setReadMethod
Purpose:	Sets the boolean direct_read which determines if the polltape function is to read 
			from an input buffer or serial port. if direct_read	is true then the system reads 
			from the serial port, if not, it attempts to read from the input buffer.  The 
			default is to read from the input buffer.
Accepts:	void
Returns:    void
*/
/*void collect::setReadMethod(bool read_method)
{
	direct_read = read_method;
}*/

void collect::SetReadMode(bool bDirect) {
	direct_read=bDirect;
}
