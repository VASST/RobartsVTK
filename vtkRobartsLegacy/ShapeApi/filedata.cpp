// filedata.cpp: implementation of the filedata class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "filedata.h"
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

filedata::filedata()
{
	fileopened=0;
	filedataindex=0;
	fileexists=0;
	found_category=0;
	nRead=0;
	charbuffer=0;
}

filedata::filedata(char *filename)
{
	fileopened=0;
	filedataindex=0;
	fileexists=0;
	found_category=0;
	nRead=0;
	charbuffer=0;
	openDataFile(filename);
}

filedata::filedata(unsigned char *buffer, int nSize) {
	fileopened=0;
	filedataindex=0;
	fileexists=1;
	charbuffer=0;
	found_category=0;
	nRead=nSize;
	databuffer = (char *)buffer;
}

filedata::~filedata()
{
	if (fileopened==1) closeDataFile();
	if (fileexists==1) delete (databuffer);
	if (charbuffer==1) delete (databuffer);
}

int filedata::openDataFile(char *filename)
{
	int filesize;
	//datafile=fopen(filename,"r+");//try to open existing file for reading and writing
	datafile = _open(filename, _O_RDWR, _S_IREAD | _S_IWRITE);
	if (datafile==-1)//NULL) //check to see if file exists
	{
		//datafile=fopen(filename, "w+"); //open empty file for reading and writing
		datafile = _open(filename, _O_CREAT | _O_RDWR, _S_IREAD | _S_IWRITE);
		if (datafile==-1) return -1; //error opening file
		databuffer=new char[1];
		charbuffer=1;
	}
	else fileexists=1;
	fileopened=1;
	
	if (fileexists==1)
	{
		//load file data into buffer:
		filesize=getFileLength(filename);
		databuffer=new char[filesize+1];
		//nRead=fread(databuffer,1,filesize,datafile);
		nRead = _read(datafile,databuffer,filesize);
		if (nRead<0) //file is empty or could not be read
		{
			delete (databuffer);
			//fclose(datafile);
			_close(datafile);
			return -2;
		}
	}
	//add NULL value to end of buffer:
	databuffer[nRead]=0;
	//nRead++;
	return 0;
}

int filedata::getInteger(char *category, char *field)
{
	int retval;
	int stringpos=FindString(category,field);
	if (stringpos<0) return 0; //could not find category and / or field (just set value to zero).
	sscanf(&databuffer[stringpos],"%d",&retval);
	return retval;
}

int filedata::getInteger(char *category, char *field, int numItems, int arrayItems[])
{
	int numScanned;
	int i=0;
	int stringpos=FindString(category,field);
	if (stringpos<0) 
	{
		//could not find category or field, so just fill array with zeros
		for (i=0;i<numItems;i++)
		{
			arrayItems[i]=0;
		}
		return 0; 
	}
	do
	{
		numScanned=sscanf(&databuffer[stringpos],"%d",&arrayItems[i]);
		while (databuffer[stringpos]<=32) stringpos++; 
		while (databuffer[stringpos]>32) stringpos++; //skip to next item in buffer
		if (numScanned>0) i++;
	} while ((i<numItems)&&(numScanned>0));
	if (i != numItems)
	{
		for (int j = i; j < numItems;j++)
			arrayItems[j] = 0;
	}
	return i;
}



double filedata::getDouble(char *category, char *field)
{
	double retval;
	int stringpos=FindString(category,field);
	if (stringpos<0) return 0; //could not find category or field
	sscanf(&databuffer[stringpos],"%lf",&retval);
	return retval;
}

int filedata::getDouble(char *category, char *field, int numItems, double arrayItems[])
{
	int numScanned;
	int i=0;
	int stringpos=FindString(category,field);
	if (stringpos<0) 
	{
		//could not find category or field
		//fill array with zeros
		for (int i=0;i<numItems;i++) arrayItems[i]=0;
		return 0; 
	}
	do
	{
		numScanned=sscanf(&databuffer[stringpos],"%lf",&arrayItems[i]);
		while (databuffer[stringpos]<=32) stringpos++; 
		while (databuffer[stringpos]>32) stringpos++; //skip to next item in buffer
		if (numScanned>0) i++;
	} while ((i<numItems)&&(numScanned>0));
	if (i != numItems)
	{
		for (int j = i; j < numItems;j++)
			arrayItems[j] = 0;
	}
	return i;
}

int filedata::getString(char *category, char *field, char **stringval)
{
	int retval=0; //the number of characters in stringval
	int startpos=0;	
	int stringpos=FindString(category,field);
	startpos=stringpos;
	if (stringpos<0)
		return 0; //could not find category or field
	while ((databuffer[stringpos]!=10)&&(databuffer[stringpos]!=13)&&(stringpos<nRead))
	{
		stringpos++;
		retval++;
	}
	char *retstring=new char[retval];
	for (int i=0;i<(retval-1);i++)
	{
		retstring[i]=databuffer[startpos+1+i];
	}
	retstring[retval-1]=0; //append null character	
	//copy pointer
	*stringval=retstring;
	return retval-1;
}

int filedata::FindString(char *category, char *field) //returns the index in filedata where the description string is first found following start_index
{
	int stringpos=0;
	int bufferpos=0;
	int stringcomparison=0;
	int exitloop=0;
	
	found_category=0;	
	//first find the desired category in the file buffer:
	do
	{
		if ((category[stringpos]!=0)&&(bufferpos<nRead))
		{
			if (category[stringpos]==databuffer[bufferpos])
			{
				bufferpos++;
				stringpos++;
				stringcomparison=1;
			}
			else
			{
				bufferpos=bufferpos+1-stringpos;
				stringpos=0;
				stringcomparison=0;
			}
		}
		else exitloop=1;
	} while (exitloop==0);
	if (stringcomparison==0) 
	{
		return -1; //error could not find category string in datafile
	}
	found_category=1;
	filedataindex=bufferpos;
	//Now find the position of the field string
	exitloop=0;
	stringpos=0;
	stringcomparison=0;
	do
	{
		if ((field[stringpos]!=0)&&(bufferpos<nRead)&&(databuffer[bufferpos]!='['))
		{
			if (field[stringpos]==databuffer[bufferpos])
			{
				bufferpos++;
				stringpos++;
				stringcomparison=1;
			}
			else
			{
				bufferpos=bufferpos+1-stringpos;
				stringpos=0;
				stringcomparison=0;
			}
		}
		else exitloop=1;
	} while (exitloop==0);
	if (stringcomparison==0) 
	{
		return -2; //error could not find description string in datafile
	}
	else return bufferpos;
}

int filedata::closeDataFile()
{
	int testwrite;
	fileopened=0;
	//write whatever data is in the buffer to file:
	//rewind(datafile); //reposition the file pointer back to the beginning of the file
	_lseek(datafile, 0L, SEEK_SET );
	//testwrite=fwrite(databuffer,1,nRead,datafile);
	testwrite = _write(datafile, databuffer, nRead);
	//fclose(datafile);
	_close(datafile);
	return testwrite;
}



void filedata::writeData(char *category, char *field, int data)
{
	char *datastring;
	int index;
	int i=0;
				
	//allocate space for datastring:
	int category_size, field_size, data_size;
	category_size=0;
	while (category[category_size]!=0) category_size++;
	field_size=0;
	while (field[field_size]!=0) field_size++;
	data_size=getNumDigits(data);

	datastring = new char[category_size+field_size+data_size+6]; //add extra room for CR/LF, spaces, and NULL characters
	
	
	//find index in databuffer where data should be written:
	if (nRead>0)  //has the databuffer been created?
	{
		index=FindString(category,field);
		if (index>=0) // 
		{
			eraseLine(index);
			sprintf(datastring," %d\n",data);
			doWrite(datastring,index);
		}
		else if (found_category==1) 
		{
			index=filedataindex;
			while ((databuffer[index]!=10)&&(databuffer[index]!=13)) index++; //skip past whitespace
			while ((databuffer[index]==10)||(databuffer[index]==13)) index++; //skip past CR/LF
			sprintf(datastring,"%s %d\n",field,data);
			doWrite(datastring,index);
		}		
		else
		{
			index=nRead-1; //just append info to the end of the buffer
			while ((databuffer[index]!=10)&&(databuffer[index]!=13)&&(index>0)) index--;
			index++;
			sprintf(datastring,"%s\n%s %d\n",category,field,data);
			doWrite(datastring,index);
		}
	}
	else
	{
		index=nRead-1; //just append info to the end of the buffer
		while ((databuffer[index]!=10)&&(databuffer[index]!=13)&&(index>0)) index--;
		index++;
		sprintf(datastring,"%s\n%s %d\n",category,field,data);
		doWrite(datastring,index);
	}
	delete datastring;
}

void filedata::writeData(char *category, char *field, int numItems, int dataItems[])
{
	char *datastring;
	char *tempstring;
	int index;
	int i;
		
	//allocate space for datastring:
	int category_size, field_size, data_size;
	category_size=0;
	while (category[category_size]!=0) category_size++;
	field_size=0;
	while (field[field_size]!=0) field_size++;
	data_size=0;
	for (i=0;i<numItems;i++) data_size+=getNumDigits(dataItems[i]);
	
	datastring = new char[category_size+field_size+data_size+numItems+6];
	tempstring = new char[category_size+field_size+data_size+numItems+6];
	
	//find index in databuffer where data should be written:
	if (nRead>0)  //has the databuffer been created?
	{
		index=FindString(category,field);
		if (index>=0)
		{
			eraseLine(index);
			strcpy(datastring,"");
			for (i=0;i<numItems;i++) 
			{
				sprintf(tempstring," %d",dataItems[i]);
				strcat(datastring,tempstring);
			}
			strcat(datastring,"\n");
			doWrite(datastring,index);
		}
		else if (found_category==1) 
		{
			index=filedataindex;
			while ((databuffer[index]!=10)&&(databuffer[index]!=13)) index++; //skip past whitespace
			while ((databuffer[index]==10)||(databuffer[index]==13)) index++; //skip past CR/LF
			sprintf(datastring,"%s",field);
			for (i=0;i<numItems;i++)
			{
				sprintf(tempstring," %d",dataItems[i]);
				strcat(datastring,tempstring);
			}
			strcat(datastring,"\n");
			doWrite(datastring,index);
		}		
		
		
		else
		{
			index=nRead-1; //just append info to the end of the buffer
			while ((index>0)&&(databuffer[index]!=10)&&(databuffer[index]!=13)) index--;
			index++;
			sprintf(datastring,"%s\n%s",category,field);
			for (i=0;i<numItems;i++) 
			{
				sprintf(tempstring," %d",dataItems[i]);
				strcat(datastring,tempstring);
			}
			strcat(datastring,"\n");
			doWrite(datastring,index);
		}
	}
	else
	{
		index=nRead-1; //just append info to the end of the buffer
		while ((index>0)&&(databuffer[index]!=10)&&(databuffer[index]!=13)) index--;
		index++;
		sprintf(datastring,"%s\n%s",category,field);
		for (i=0;i<numItems;i++) 
		{
			sprintf(tempstring," %d",dataItems[i]);
			strcat(datastring,tempstring);
		}
		strcat(datastring,"\n");
		doWrite(datastring,index);
	}
	delete datastring;
	delete tempstring;
}



void filedata::writeData(char *category, char *field, double data)
{
	char *datastring;
	int index;
	int i=0;
				
	//allocate space for datastring:
	int category_size, field_size, data_size;
	category_size=0;
	while (category[category_size]!=0) category_size++;
	field_size=0;
	while (field[field_size]!=0) field_size++;
	data_size=getNumDigits(data);
	datastring = new char[category_size+field_size+data_size+6]; //add extra room for CR/LF, spaces, and NULL characters
	
	
	//find index in databuffer where data should be written:
	if (nRead>=0)  //has the databuffer been created?
	{
		index=FindString(category,field);
		if (index>=0)
		{
			eraseLine(index);
			sprintf(datastring," %lf\n",data);
			doWrite(datastring,index);
		}
		else if (found_category==1) 
		{
			index=filedataindex;
			while ((databuffer[index]!=10)&&(databuffer[index]!=13)) index++; //skip past whitespace
			while ((databuffer[index]==10)||(databuffer[index]==13)) index++; //skip past CR/LF
			sprintf(datastring,"%s %lf\n",field,data);
			doWrite(datastring,index);
		}		
		
		else
		{
			index=nRead-1; //just append info to the end of the buffer
			while ((index>0)&&(databuffer[index]!=10)&&(databuffer[index]!=13)) index--;
			index++;
			sprintf(datastring,"%s\n%s %lf\n",category,field,data);
			doWrite(datastring,index);
		}
	}
	else
	{
		index=nRead-1; //just append info to the end of the buffer
		while ((index>0)&&(databuffer[index]!=10)&&(databuffer[index]!=13)) index--;
		index++;
		sprintf(datastring,"%s\n%s %lf\n",category,field,data);
		doWrite(datastring,index);
	}
	delete datastring;
}


void filedata::writeData(char *category, char *field, int numItems, double dataItems[])
{
	char *datastring;
	char *tempstring;
	int index;
	int i;
		
	//allocate space for datastring:
	int category_size, field_size, data_size;
	category_size=strlen(category);
	field_size=strlen(field);
	data_size=0;
	for (i=0;i<numItems;i++) data_size+=getNumDigits(dataItems[i]);
		
	datastring = new char[category_size+field_size+data_size+numItems+6];
	tempstring = new char[category_size+field_size+data_size+numItems+6];
	
	//find index in databuffer where data should be written:
	if (nRead>0)  //has the databuffer been created?
	{
		index=FindString(category,field);
		if (index>=0)
		{
			eraseLine(index);
			strcpy(datastring,"");
			for (i=0;i<numItems;i++)
			{
				sprintf(tempstring," %lf",dataItems[i]);
				strcat(datastring,tempstring);
			}
			strcat(datastring,"\n");
			doWrite(datastring,index);
		}
		else if (found_category==1) 
		{
			index=filedataindex;
			while ((databuffer[index]!=10)&&(databuffer[index]!=13)) index++; //skip past whitespace
			while ((databuffer[index]==10)||(databuffer[index]==13)) index++; //skip past CR/LF
			sprintf(datastring,"%s",field);
			for (i=0;i<numItems;i++) 
			{
				sprintf(tempstring," %lf",dataItems[i]);
				strcat(datastring,tempstring);
			}
			strcat(datastring,"\n");
			doWrite(datastring,index);
		}		
		else
		{
			index=nRead-1; //just append info to the end of the buffer
			while ((index>0)&&(databuffer[index]!=10)&&(databuffer[index]!=13)) index--;
			index++;
			sprintf(datastring,"%s\n%s",category,field);
			for (i=0;i<numItems;i++) 
			{
				sprintf(tempstring," %lf",dataItems[i]);
				strcat(datastring,tempstring);
			}
			strcat(datastring,"\n");
			doWrite(datastring,index);
		}
	}
	else
	{
		index=nRead-1; //just append info to the end of the buffer
		while ((index>0)&&(databuffer[index]!=10)&&(databuffer[index]!=13)) index--;
		index++;
		sprintf(datastring,"%s\n%s",category,field);
		for (i=0;i<numItems;i++) 
		{
			sprintf(tempstring," %lf",dataItems[i]);
			strcat(datastring,tempstring);
		}
		strcat(datastring,"\n");
		doWrite(datastring,index);
	}
	delete datastring;
	delete tempstring;
}

void filedata::writeData(char *category, char *field, char *data)
{
	char *datastring;
	int index;
	int i=0;
				
	//allocate space for datastring:
	int category_size, field_size, data_size;
	category_size=0;
	while (category[category_size]!=0) category_size++;
	field_size=0;
	while (field[field_size]!=0) field_size++;
	data_size=0;
	while (data[data_size]!=0) data_size++;
	
	datastring = new char[category_size+field_size+data_size+6]; //add extra room for CR/LF, spaces, and NULL characters
	
	
	//find index in databuffer where data should be written:
	if (nRead>0)  //has the databuffer been created?
	{
		index=FindString(category,field);
		if (index>=0)
		{
			eraseLine(index);
			sprintf(datastring," %s\n",data);
			doWrite(datastring,index);
		}
		else if (found_category==1) 
		{
			index=filedataindex;
			while ((databuffer[index]!=10)&&(databuffer[index]!=13)) index++; //skip past whitespace
			while ((databuffer[index]==10)||(databuffer[index]==13)) index++; //skip past CR/LF
			sprintf(datastring,"%s %s\n",field,data);
			doWrite(datastring,index);
		}		
		
		else
		{
			index=nRead-1; //just append info to the end of the buffer
			while ((index>0)&&(databuffer[index]!=10)&&(databuffer[index]!=13)) index--;
			index++;
			sprintf(datastring,"%s\n%s %s\n",category,field,data);
			doWrite(datastring,index);
		}
	}
	else
	{
		index=nRead-1; //just append info to the end of the buffer
		while ((index>0)&&(databuffer[index]!=10)&&(databuffer[index]!=13)) index--;
		index++;
		sprintf(datastring,"%s\n%s %s\n",category,field,data);
		doWrite(datastring,index);
	}
	delete datastring;
}


int filedata::getFileLength(char *filename) //Finds the length of an opened file stream in bytes
{
	int totalBytes, fh;
	
	
	fh = _open( filename, _O_RDONLY );
	
	totalBytes = _lseek(fh, 0L, SEEK_END );
	_lseek(fh, 0L, SEEK_SET );

    _close( fh );

	/*totalBytes=0;
	do
	{
		nBytes=fread(inbuffer,1,1024,datafile);
		totalBytes+=nBytes;
	} while (nBytes==1024);
	*/
	//reset file pointer to beginning of file:
	//rewind(datafile);
	return totalBytes;
}


int filedata::getNumDigits(int data)
{
	int nDigits=0;
	char datastring[20];
	sprintf(datastring,"%d",data);
	while (datastring[nDigits]!=0) nDigits++;
	return nDigits;
}

int filedata::getNumDigits(double data) //include the decimal point
{	
	int nDigits=0;
	char datastring[40];
	sprintf(datastring,"%f",data);
	while (datastring[nDigits]!=0) nDigits++;
	return nDigits;
}

void filedata::insertToBuffer(int itemsize, char *itemBuffer, int insertionpoint) //inserts itemBuffer string into databuffer
{
	char *tmpbuffer = new char[nRead];
	int i;
	for (i=0;i<nRead;i++) tmpbuffer[i]=databuffer[i];
	if (fileexists==1) delete databuffer;
	databuffer=new char[nRead+itemsize];
	for (i=0;i<insertionpoint;i++) databuffer[i]=tmpbuffer[i];
	for (i=0;i<itemsize;i++) databuffer[insertionpoint+i]=itemBuffer[i];
	for (i=insertionpoint;i<nRead;i++) databuffer[i+itemsize]=tmpbuffer[i];
	delete tmpbuffer;
	nRead+=itemsize;
	return;
}

int filedata::eraseLine(int index) //erases a line of databuffer, starting at index -- returns number of characters erased
{
	int numErased=0;
	int i;
	int oldnRead=nRead;
	char *delbuffer;
	i=index;
	while (databuffer[i]>=32||databuffer[i]=='\t')
	{
		numErased++;
		i++;		
	}
	while ((databuffer[i]==10)||(databuffer[i]==13)) 
	{
		i++; //skip past CR/LFnumErased+=2; //erase old carriage return / linefeed sequence
		numErased++;
	}
		
	//shift contents of buffer downwards:
	nRead-=numErased;
	for (i=index;i<nRead;i++) 	databuffer[i]=databuffer[i+numErased];
	delbuffer=new char[numErased];
	for (i=0;i<numErased;i++) delbuffer[i]=0;
	//fseek(datafile,nRead,SEEK_SET);
	_lseek(datafile, nRead, SEEK_SET);
	//fwrite(delbuffer,1,numErased,datafile); //erase extra data from file
	_write(datafile, delbuffer, numErased);
	delete delbuffer;
	return numErased;
}

void filedata::doWrite(char *datastring,int index) //performs writing to buffer
{
	int stringsize=strlen(datastring);
	insertToBuffer(stringsize,datastring,index);
	return;
}
