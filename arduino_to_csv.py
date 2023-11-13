"""
Module: arduino_to_csv
Author: Dema N. Govalla
Date: November 12, 2023
Description: The file collects Arduino data from the serial port and saves it in a CSV file.
The data collected is the Time, data[0], Force, X-axis, Y-axis, Z-axis, and Class values. 
For a predetermined time, the file collects X number of points for X/(50Hz * 60) minutes and saves it to a CSV file. 
The data is used to train and test the RFMN and TSC-LS algorithms.
"""

# Import necessary module, serial and time
import serial
import time

# Define constants used in the code
arduino_port = "COM5"
fileName = "arduino_to_csv.csv"
baud = 115200
samples = 4001   
line = 0
ser = serial.Serial(arduino_port, baud)
file = open(fileName, "w")

# Try and except to help handle exception and prevent the program from crashing
try:
    # Skip lines in serial port until headers are found
    while True:
        getData = ser.readline().decode('utf-8')
        if "Data[0],Time,Force,X_axis,Y_axis,Z_axis,Class" in getData:
            break

    # collect the samples
    while line <= samples:
        if line == 0:
            data = "Data[0],Time,Force,X_axis,Y_axis,Z_axis,Class"
            start_time = time.time()
        else:
            # Get the start time               
            getData = ser.readline().decode('utf-8')
            data = getData[0:][:-2]
        print(data)   
        file = open(fileName, "a")
        file.write(data + "\n")
        line += 1

except Exception as e:
    print(f"An unexpected error occured: {e}") 

# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time", elapsed_time)
print("Data collection complete!")