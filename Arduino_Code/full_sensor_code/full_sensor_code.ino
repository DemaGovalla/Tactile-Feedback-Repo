/*
 * Module: full_sensor_code.ino
 * Author: Dema N. Govalla
 * Date: November 12, 2023
 * Description: The file is used to collect Force, Magnetometer, Accelerometer and Gyroscope data coming from the Arduino. 
                Printing the data on the serial monitor for data processing. 
 */

#include <Wire.h> // Include the Wire library for I2C communication
#include <Arduino_LSM6DSOX.h>

#define MAGNETOMETER_ADDRESS 0x0D // Define the address of the Magnetomer
#define Force_sensor A0 // Define the analog pin on the Arduino Uno to communicate with

int Class = 1; // What Label/Class is in training. 
long data[7]; // Number of bit being read (should be a total of 8 bit)
byte i;
bool label = true;  
float Time = 0.000, Force;

// Define the labels being printed as headers
String dataLabel1 = "Data[0]";
String dataLabel2 = "Time";
String dataLabel3 = "Force";
String dataLabel4 = "X_axis_mag";
String dataLabel5 = "Y_axis_mag";
String dataLabel6 = "Z_axis_mag";
String dataLabel7 = "X_axis_accel";
String dataLabel8 = "Y_axis_accel";
String dataLabel9 = "Z_axis_accel";
String dataLabel10 = "X_axis_gyro";
String dataLabel11 = "Y_axis_gyro";
String dataLabel12 = "Z_axis_gyro";

float xySensitivity = 1.5;  // Sensitivity for a typical 50mT range is 1.5. Same for the z-axis. Page 13
float zSensitivity = 1.5;

void setup() {
  Wire.begin();
  Wire.setClock(100000); // Initialize the I2C bus
  Serial.begin(115200); // Initialize serial communication for debugging

  while (!Serial);
    if (!IMU.begin()) {
      Serial.println("Failed to initialize IMU!"); // Identify the IMU is connected to the arduino. 
      while (1);
    }

  pinMode(Force_sensor, INPUT); // Initilize the force sensor as an input
  
  Wire.beginTransmission(MAGNETOMETER_ADDRESS);
  Wire.write(0x11); // Write command to read data from address 0x11, Addressed RESET when users sends an I2C_ADDRESSED_RESET command, page 33
  Wire.write(0x06); // Defalt setting to 0x06 to reset the IC chip
  Wire.endTransmission(true);

  Wire.beginTransmission(MAGNETOMETER_ADDRESS);
  Wire.write(0x14); // Write command to read data from address 0x14, associated with OSR_DIG_FILT. Page 22, 29, and 34
  Wire.write(0xE9); // E9 equals 1110 1001 corresponds to a refresh rate of 50 Hz, see page 22
  Wire.endTransmission(true);

  Wire.beginTransmission(MAGNETOMETER_ADDRESS);
  Wire.write(0x15); // Write command to read data from address 0x15, associated with T_EN_DIG_FILT_Z. Page 22, 29, and 35
  Wire.write(0xB6); // E9 equals 1011 0110.
  Wire.endTransmission(true);

  Wire.beginTransmission(MAGNETOMETER_ADDRESS);
  Wire.write(0x0E); // Write command to read data from address 0x0E, associated with CTRL. Page 29, and 32
  Wire.write(0x74); // 74 equals 0111 0100. Mode: 4 – Continuous measurement mode 50Hz 
  Wire.endTransmission(true);
  
  Wire.beginTransmission(MAGNETOMETER_ADDRESS);
  Wire.write(0x0F); // Write command to read data from address 0x0F, associated with CUST_CTRL2. Page 29, 33
  Wire.write(0x03); // 00 equals 0000 0011. Range select: 3 – X and Y axes = 50mT, Z axis = 50mT
  Wire.endTransmission(true);  
}

void loop() {
  while(label){ 
    //print headers
    Serial.print(dataLabel1);
    Serial.print(",");
    Serial.print(dataLabel2);
    Serial.print(",");
    Serial.print(dataLabel3);
    Serial.print(",");
    Serial.print(dataLabel4);
    Serial.print(",");
    Serial.print(dataLabel5);
    Serial.print(",");
    Serial.print(dataLabel6);
    Serial.print(",");
    Serial.print(dataLabel7); 
    Serial.print(",");
    Serial.print(dataLabel8);
    Serial.print(",");
    Serial.print(dataLabel9);
    Serial.print(",");
    Serial.print(dataLabel10);
    Serial.print(",");
    Serial.print(dataLabel11);
    Serial.print(",");
    Serial.println(dataLabel12);

    label = false;
  }
  
  Wire.beginTransmission(MAGNETOMETER_ADDRESS);
  Wire.write(0x00); // Write command to read data from address 0x00
  Wire.endTransmission(false);
  
  Wire.requestFrom(MAGNETOMETER_ADDRESS, 7); // Request 2 bytes of data starting from address 0x00
  for (i = 0; i < 7; i++) {
    data[i] = Wire.read(); // Read 2 bytes of data
  }

  int xMagneticField = ((data[2] << 8) | data[1]);
  int yMagneticField = ((data[4] << 8) | data[3]);
  int zMagneticField = ((data[6] << 8) | data[5]);
  
  float xMicroTesla = xMagneticField * xySensitivity;
  float yMicroTesla = yMagneticField * xySensitivity;
  float zMicroTesla = zMagneticField * zSensitivity;

  float xAccelerometer, yAccelerometer, zAccelerometer;
  float xGyroscope, yGyroscope, zGyroscope;

  if (IMU.gyroscopeAvailable() && IMU.accelerationAvailable() ) {
      IMU.readAcceleration(xAccelerometer, yAccelerometer, zAccelerometer);
      IMU.readGyroscope(xGyroscope, yGyroscope, zGyroscope);
    }

  Force = abs(1023-analogRead(Force_sensor));

  Serial.print(data[0]);
  Serial.print(",");
  Serial.print(Time,4);
  Serial.print(",");
  Serial.print(Force);
  Serial.print(",");
  Serial.print(xMicroTesla, 4);
  Serial.print(",");
  Serial.print(yMicroTesla, 4);
  Serial.print(",");
  Serial.print(zMicroTesla, 4);
  Serial.print(",");
  Serial.print(xAccelerometer, 4);
  Serial.print(",");
  Serial.print(yAccelerometer, 4);
  Serial.print(",");
  Serial.print(zAccelerometer, 4);
  Serial.print(",");
  Serial.print(xGyroscope, 4);
  Serial.print(",");
  Serial.print(yGyroscope, 4);
  Serial.print(",");
  Serial.println(zGyroscope, 4);

  Time += 0.02; // Increment the time by 50 Hz

  delay(20); // delay by 20 miliseconds - collecting data by 50Hz sampling frequency
}
