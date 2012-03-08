#include "stdafx.h"
#include "serial.h"
#include "filedata.h"
#include "shapeAPI_error.h"
//#include "API_exception.h"

using namespace std;
// Function: Constructor.
/* Open the com port.  Set all the apropriate settings. */
serial::serial(char* com_port, int baud_rate, int num_sensors,  int serial_num) // Constructor 
{
	baud = baud_rate;
	initialize(com_port, num_sensors, serial_num);
}

serial::serial(char* config_file)
{
	char *com_port;
	int serial_num;
	int num_sensors;
	filedata file(config_file);

	num_sensors = file.getInteger("[settings]","number of sensors");
	baud = file.getInteger("[settings]","baud rate");
	file.getString("[settings]","com port", &com_port);
	serial_num = file.getInteger("[settings]","serial number");

	initialize(com_port, num_sensors, serial_num); 
}

void serial::initialize(char* com_port, int num_sensors, int serial_num)
{
	hPort = CreateFile(com_port, 
		GENERIC_READ | GENERIC_WRITE,
		0,
		NULL,
		OPEN_EXISTING,
		0,
		NULL);

	if (hPort == INVALID_HANDLE_VALUE) // Port was not opened.
	{
		PurgeComm(hPort, PURGE_TXABORT | PURGE_RXABORT | PURGE_TXCLEAR | PURGE_RXCLEAR);
		SetLastError(SERIAL_OPEN_ERROR);
		//throw serial_open_exception();
		return;
	}

	// Set extTrigger to the default of false.
	setTrigger(false);

	// set timeouts
	
	COMMTIMEOUTS cto = { 80,10,40,0,0 };
	
	if(!SetCommTimeouts(hPort, &cto))
	{
		SetLastError(SERIAL_TIMEOUT_ERROR);
		//throw serial_timeout_exception();
	}
	
	// set DCB variable
	
	DCB dcb;
	
	memset(&dcb, 0, sizeof(dcb));
	
	dcb.DCBlength = sizeof(dcb);
	dcb.BaudRate = baud;
	dcb.fBinary = 1;
	dcb.fParity = 0;
	dcb.fOutxCtsFlow = 0;
	dcb.fOutxDsrFlow = 0;
	dcb.fDtrControl = DTR_CONTROL_DISABLE;
	dcb.fDsrSensitivity = 0;
	dcb.fTXContinueOnXoff = TRUE;
	dcb.fOutX = 0;
	dcb.fInX = 0;
	dcb.fErrorChar = 0;
	dcb.fNull = 0;
	dcb.fRtsControl = RTS_CONTROL_DISABLE;
	dcb.fAbortOnError = 0;
	dcb.ByteSize = 8;
	dcb.Parity = NOPARITY;
	dcb.StopBits = ONESTOPBIT;
	
	if(!SetCommState(hPort, &dcb))
	{
		SetLastError(SERIAL_DCB_ERROR);
		//throw serial_dcb_exception();
	}

	buffer_size = 0;
	// Figure out how many bytes are in a single shape tape frame.
	if (baud == 57600)
	{
		buffer_size = num_sensors * 2;
	}
	else if (baud == 115200)
	{
		buffer_size += 3; // led / serial number
		if (serial_num > 4096)
			buffer_size += 13; // counter / frame number
		buffer_size += num_sensors * 2; // data points
		buffer_size += 1; // check sum byte
	}
	else
	{
		SetLastError(SERIAL_BAUDRATE_ERROR);
		//throw serial_baudrate_exception();
	}
}

serial::~serial() // Destructor.
{
	if(!CloseHandle(hPort))
	{
		SetLastError(SERIAL_CLOSE_ERROR);
		//throw serial_close_exception();
	}
}

/* Reads some information from the com port and put it in the 
message variable.  Return the number of bytes read. */
int serial::read(int number_of_bytes, unsigned char* message)
{
	DWORD bytes_read, error;
	ReadFile(hPort, message, number_of_bytes, &bytes_read, NULL);
	ClearCommError(hPort, &error, NULL);
	if (bytes_read != number_of_bytes)
		SetLastError(SERIAL_READ_ERROR);
		//throw serial_read_exception();
	return bytes_read;
}

/* Writes the message variable (or number_of_bytes of it) to the 
com port.  Returns the number of bytes written. */
int serial::write(int number_of_bytes, unsigned char* message)
{
	DWORD bytes_written;
	WriteFile(hPort, message, number_of_bytes, &bytes_written, NULL);
	if (bytes_written != number_of_bytes)
		SetLastError(SERIAL_WRITE_ERROR);
		//throw serial_write_exception();
	return bytes_written;
}


/* Sets the value of extTrigger to determine whether there is an 
   external trigger used to trigger the tape. */
void serial::setTrigger(bool trigger)
{
	extTrigger = trigger;
}

/* Reads a frame from the shape tape. */
int serial::readBuffer(unsigned char message[117])
{
	unsigned char out_message[16];
	unsigned char in_message;
	unsigned char base_message = 0x20;

	if (baud == 57600)
	{
		int bytes_read = 0;
		for (int i = 1; i <= buffer_size / 16; i++)
		{
			in_message = base_message |= ((unsigned char)i);
			write(1, &in_message);
			bytes_read += read(16, out_message);
			for (int j=0; j < 16; j++)
				message[(i-1)*16+j] = out_message[j];
		}
		return bytes_read;
	}
	else if (baud == 115200)
	{
		in_message = 0x20;
		if (extTrigger)
		{
			// to recieve the data from external pulse.LPDWORD 
			WaitCommEvent(hPort, (unsigned long *)EV_RXCHAR, NULL);
			//success = WaitCommEvent(hPort, result, NULL);
		}
		else 
		{
			write(1,&in_message);
		}
		return read(buffer_size, message);
	}
	else 
	{
		SetLastError(SERIAL_BAUDRATE_ERROR);
		//throw serial_baudrate_exception();
		return -1;
	}
}