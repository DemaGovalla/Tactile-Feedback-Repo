"""
Module: arduino_to_csv.py
Author: Dema N. Govalla
Date: November 12, 2023
Description: The file collects Arduino data from the serial port and saves it in a CSV file.
The data collected is the Data[0], Time, Force, and the X-axis, Y-axis, and Z-axis values for the 
Magnetometer, Accelerometer, and Gyroscope, and Class values. 
For a predetermined time, the file collects X number of points for X/(50Hz * 60) minutes and saves it to a CSV file. 
The data is used to train and test the RFMN and TSC-LS algorithms.
"""

# Import necessary module, serial and time
import serial

# Define constants used in the code
arduino_port = "COM5"
fileName = "arduino_to_csv.csv"
baud = 115200
# samples = 4001
samples = 41  
line = 0
ser = serial.Serial(arduino_port, baud)
file = open(fileName, "w")
Class = 2
headers = (
    "Data[0],Time,Force,X_axis_mag,Y_axis_mag,Z_axis_mag,"
    "X_axis_acel,Y_axis_acel,Z_axis_acel,X_axis_gyro,"
    "Y_axis_gyro,Z_axis_gyro"
)

# Try and except to help handle exception and prevent the program from crashing
try:
    # Skip lines in serial port until headers are found
    while True:
        getData = ser.readline().decode('utf-8')
        if headers in getData:

            break

    # collect the samples
    while line <= samples:
        if line == 0:
            data = f"{headers},Class"
        else:
            getData = ser.readline().decode('utf-8')
            # data = getData[0:][:-2]
            data = f"{getData[0:][:-2]},{Class}"
        print(data)   
        file = open(fileName, "a")
        file.write(data + "\n")
        line += 1

except Exception as e:
    print(f"An unexpected error occured: {e}") 

print("Data collection complete!")