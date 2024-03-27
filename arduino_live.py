"""
Module: arduino_live.py
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file collect and prints out live data coming from the Arduino Uno in real-time. 
            The file is used to simulate real-time data coming from the various sensors. 
            The real time data is saved onto a file names called arduino_live.csv
"""

import serial, os

arduino_port = "COM5"
baud = 115200
fileName = "arduino_live.csv"
samples = 10
print_labels = False
trigger_line = "Data[0],Time,Force,Mag,Accel,Gyro"  # Define the trigger line
ser = serial.Serial(arduino_port, baud)
print("Connected to Arduino port:" + arduino_port)
os.remove(fileName)
line = 0

file = open(fileName, "w")
print("Created file")

trigger_found = False  # Flag to indicate if trigger line is found

while True:
    if print_labels:
        if line == 0:
            print("Printing Column Headers")
        else:
            print("Line" + str(line) + ": writing..")

    getData = ser.readline()
    dataString = getData.decode('utf-8')
    data = dataString[0:][:-2]
    print(data)

    if trigger_line in data:  # Check if trigger line is found
        print("Trigger line found. Start saving data.")
        trigger_found = True

    if trigger_found:  # Start saving data once trigger line is found
        file = open(fileName, "a")
        file.write(data + "\n")
