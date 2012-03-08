#include "stdafx.h"
#include "API_exception.h"
using namespace std;

API_exception::API_exception()
{

}

void API_exception::print_message()
{  
    cout << message << endl; 
} // Display the message.