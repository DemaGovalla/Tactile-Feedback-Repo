// Reading DS18B20 By Address

//#include <OneWire.h>
//#include <DallasTemperature.h>
//
//// Data wire is plugged into port 2 on the Arduino
//#define ONE_WIRE_BUS 2
//
//// Setup a oneWire instance to communicate with any OneWire devices
//OneWire oneWire(ONE_WIRE_BUS);
//
//// Pass our oneWire reference to Dallas Temperature.
//DallasTemperature sensors(&oneWire);
//
//// variable to hold device addresses
//DeviceAddress Thermometer;
//
//int deviceCount = 0;
//
//void setup(void)
//{
//  // start serial port
//  Serial.begin(9600);
//
//  // Start up the library
//  sensors.begin();
//
//  // locate devices on the bus
//  Serial.println("Locating devices...");
//  Serial.print("Found ");
//  deviceCount = sensors.getDeviceCount();
//  Serial.print(deviceCount, DEC);
//  Serial.println(" devices.");
//  Serial.println("");
//  
//  Serial.println("Printing addresses...");
//  for (int i = 0;  i < deviceCount;  i++)
//  {
//    Serial.print("Sensor ");
//    Serial.print(i+1);
//    Serial.print(" : ");
//    sensors.getAddress(Thermometer, i);
//    printAddress(Thermometer);
//  }
//}
//
//void loop(void)
//{}
//
//void printAddress(DeviceAddress deviceAddress)
//{ 
//  for (uint8_t i = 0; i < 8; i++)
//  {
//    Serial.print("0x");
//    if (deviceAddress[i] < 0x10) Serial.print("0");
//    Serial.print(deviceAddress[i], HEX);
//    if (i < 7) Serial.print(", ");
//  }
//  Serial.println("");
//}



// // ***********************************
// // Reading the data starts here 
// // ***********************************
// #include <OneWire.h>
// #include <DallasTemperature.h>

// // Data wire is plugged into port 2 on the Arduino
// #define ONE_WIRE_BUS 2

// // Setup a oneWire instance to communicate with any OneWire devices
// OneWire oneWire(ONE_WIRE_BUS);

// // Pass our oneWire reference to Dallas Temperature.
// DallasTemperature sensors(&oneWire);

// // Addresses of 3 DS18B20s
// uint8_t sensor1[8] = { 0x28, 0x42, 0x1E, 0x49, 0xF6, 0x4D, 0x3C, 0xFB };
// uint8_t sensor2[8] = { 0x28, 0x8D, 0xF2, 0x49, 0xF6, 0xB0, 0x3C, 0xAB };
// uint8_t sensor3[8] = { 0x28, 0x7B, 0xC0, 0x49, 0xF6, 0xE2, 0x3C, 0x8A };

// void setup(void)
// {
//   Serial.begin(9600);
//   sensors.begin();
// }

// void loop(void)
// {
//   sensors.requestTemperatures();
  
//   Serial.print("Sensor 1: ");
//   printTemperature(sensor1);
  
//   Serial.print("Sensor 2: ");
//   printTemperature(sensor2);
  
//   Serial.print("Sensor 3: ");
//   printTemperature(sensor3);
  
//   Serial.println();
// //  delay(50);
// }

// void printTemperature(DeviceAddress deviceAddress)
// {
//   float tempC = sensors.getTempC(deviceAddress);
// //  Serial.print(tempC);
// //  Serial.print((char)176);
// //  Serial.print("C  |  ");
//   float tempF = DallasTemperature::toFahrenheit(tempC);
//   Serial.print(tempF);
//   Serial.print("\xC2\xB0");
//   Serial.println("F");
// }


//Reading DS18B20 By Index 
#include <OneWire.h>
#include <DallasTemperature.h>


// Data wire is plugged into digital pin 2 on the Arduino
#define ONE_WIRE_BUS 2
// Setup a oneWire instance to communicate with any OneWire device
OneWire oneWire(ONE_WIRE_BUS);  
// Pass oneWire reference to DallasTemperature library
DallasTemperature sensors(&oneWire);

int deviceCount = 0;
float tempAverage = 0;
float tempC;
float tempF;


void setup(void)
{
 sensors.begin();  // Start up the library
 Serial.begin(9600);
 
 // locate devices on the bus
// Serial.print("Locating devices...");
// Serial.print("Found ");


 deviceCount = sensors.getDeviceCount();
// Serial.print(deviceCount, DEC);
// Serial.println(" devices.");
// Serial.println("");
}

void loop(void)
{ 
 // Send command to all the sensors for temperature conversion
 sensors.requestTemperatures(); 

float sum = 0.0, average, variance, std, range, coeffOfRange;
int classes;
double arr[deviceCount] = {};
float sqDevSum = 0.0;

 // Takes the readings from all three sensors and averages it. 
 for (int i = 0;  i < deviceCount;  i++)
 {
//   Serial.print("Sensor ");
//   Serial.print(i+1);
//   Serial.print(" : ");
   tempC = sensors.getTempCByIndex(i);
//  Serial.print(tempC);
//  Serial.print("\xC2\xB0");//shows degrees character
//  Serial.print("C  |  ");  

   tempF= DallasTemperature::toFahrenheit(tempC);
   arr[i] = tempF;
   sum += arr[i];
   
//   Serial.print(tempF);
//   Serial.print("\xC2\xB0");
//   Serial.println("F");
 }

float minimum = arr[0], maximum = arr[0];
//Find the Range and Range of Coefficient
for (int i = 0; i < deviceCount; i++)
{
    minimum = min(minimum, arr[i]);
    maximum = max(maximum, arr[i]);
}


// Set up for the STD of the sensors
for (int i = 0;  i < deviceCount;  i++)
 {
  // pow(x, 2) is x squared. Basically used to square the (average - float(arr[i]) computation. 
  sqDevSum += pow((average - float(arr[i])), 2);
 }


average = sum/deviceCount;
range = maximum - minimum;
coeffOfRange = (100*range) / (maximum + minimum);
variance = sqDevSum/deviceCount;
std = sqrt(variance);




if (average >= 80){
  Serial.print(average,4);
  Serial.print(",");
  Serial.print(range,4);
  Serial.print(",");
  Serial.print(coeffOfRange,4);
  Serial.print(",");
  Serial.print(variance,4);
  Serial.print(",");
  Serial.print(std,4);
  Serial.print(",");
  Serial.println(classes = 1);
}

else if (average >= 78)
{
  Serial.print(average,4);
  Serial.print(",");
  Serial.print(range,4);
  Serial.print(",");
  Serial.print(coeffOfRange,4);
  Serial.print(",");
  Serial.print(variance,4);
  Serial.print(",");
  Serial.print(std,4);
  Serial.print(",");
  Serial.println(classes = 2);
}

else if (average >= 76)
{
  Serial.print(average,4);
  Serial.print(",");
  Serial.print(range,4);
  Serial.print(",");
  Serial.print(coeffOfRange,4);
  Serial.print(",");
  Serial.print(variance,4);
  Serial.print(",");
  Serial.print(std,4);
  Serial.print(",");
  Serial.println(classes = 3);
}

else if (average >= 74)
{
  Serial.print(average,4);
  Serial.print(",");
  Serial.print(range,4);
  Serial.print(",");
  Serial.print(coeffOfRange,4);
  Serial.print(",");
  Serial.print(variance,4);
  Serial.print(",");
  Serial.print(std,4);
  Serial.print(",");
  Serial.println(classes = 4);
}

else if (average >= 72)
{
  Serial.print(average,4);
  Serial.print(",");
  Serial.print(range,4);
  Serial.print(",");
  Serial.print(coeffOfRange,4);
  Serial.print(",");
  Serial.print(variance,4);
  Serial.print(",");
  Serial.print(std,4);
  Serial.print(",");
  Serial.println(classes = 5);
}

else if (average >= 70)
{
  Serial.print(average,4);
  Serial.print(",");
  Serial.print(range,4);
  Serial.print(",");
  Serial.print(coeffOfRange,4);
  Serial.print(",");
  Serial.print(variance,4);
  Serial.print(",");
  Serial.print(std,4);
  Serial.print(",");
  Serial.println(classes = 6);
}

else if (average >= 68)
{
  Serial.print(average,4);
  Serial.print(",");
  Serial.print(range,4);
  Serial.print(",");
  Serial.print(coeffOfRange,4);
  Serial.print(",");
  Serial.print(variance,4);
  Serial.print(",");
  Serial.print(std,4);
  Serial.print(",");
  Serial.println(classes = 7);
}

else if (average >= 66)
{
  Serial.print(average,4);
  Serial.print(",");
  Serial.print(range,4);
  Serial.print(",");
  Serial.print(coeffOfRange,4);
  Serial.print(",");
  Serial.print(variance,4);
  Serial.print(",");
  Serial.print(std,4);
  Serial.print(",");
  Serial.println(classes = 8);
}

else if (average >= 64)
{
  Serial.print(average,4);
  Serial.print(",");
  Serial.print(range,4);
  Serial.print(",");
  Serial.print(coeffOfRange,4);
  Serial.print(",");
  Serial.print(variance,4);
  Serial.print(",");
  Serial.print(std,4);
  Serial.print(",");
  Serial.println(classes = 9);
}

else
{
  
  Serial.print(average,4);
  Serial.print(",");
  Serial.print(range,4);
  Serial.print(",");
  Serial.print(coeffOfRange,4);
  Serial.print(",");
  Serial.print(variance,4);
  Serial.print(",");
  Serial.print(std,4);
  Serial.print(",");
  Serial.println(classes = 10);
}








// Serial.print("Sensors variance: ");
// Serial.println(variance);
// Serial.print("Sensors STD: ");
// Serial.println(std);

// Serial.println("");
 delay(100);
}
