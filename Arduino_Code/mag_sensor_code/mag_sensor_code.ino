#include <Wire.h> // Include the Wire library for I2C communication

#define MAGNETOMETER_ADDRESS 0x0D // Replace with the actual address of your magnetometer

long data[7];
byte i;



void setup() {
  Wire.begin();
  Wire.setClock(100000); // Initialize the I2C bus
  Serial.begin(115200); // Initialize serial communication for debugging


  Wire.beginTransmission(MAGNETOMETER_ADDRESS);
  Wire.write(0x11); // Write command to read data from address 0x03
  Wire.write(0x06);
  Wire.endTransmission(true);

  Wire.beginTransmission(MAGNETOMETER_ADDRESS);
  Wire.write(0x14); // Write command to read data from address 0x03
  Wire.write(0xE9);
  Wire.endTransmission(true);

  Wire.beginTransmission(MAGNETOMETER_ADDRESS);
  Wire.write(0x15); // Write command to read data from address 0x03
  Wire.write(0xB6);
  Wire.endTransmission(true);

  Wire.beginTransmission(MAGNETOMETER_ADDRESS);
  Wire.write(0x0F); // Write command to read data from address 0x03
  Wire.write(0x00);
  Wire.endTransmission(true);


  Wire.beginTransmission(MAGNETOMETER_ADDRESS);
  Wire.write(0x0E); // Write command to read data from address 0x03
  Wire.write(0x74);
  Wire.endTransmission(true);
  
}

void loop() {

  

  
  // Read data from address 0x03 and 0x04
  Wire.beginTransmission(MAGNETOMETER_ADDRESS);
  Wire.write(0x00); // Write command to read data from address 0x03
  Wire.endTransmission(false);

  Wire.requestFrom(MAGNETOMETER_ADDRESS, 7); // Request 2 bytes of data from address 0x03
  for (i = 0; i < 7; i++) {
    data[i] = Wire.read(); // Read 2 bytes of data
  }

  // Convert the received data to a Y-axis magnetic field value
  int xMagneticField = ((data[2] << 8) | data[1]);
  int yMagneticField = ((data[4] << 8) | data[3]);
  int zMagneticField = ((data[6] << 8) | data[5]);

  Serial.print(data[0]);
  Serial.print(" X: ");
  Serial.print(xMagneticField);

  
  Serial.print(" Y: ");
  Serial.print(yMagneticField);


  Serial.print(" Z: ");
  Serial.println(zMagneticField);



  delay(20);
}
