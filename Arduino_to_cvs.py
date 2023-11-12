'''
This file is used to collect Arduino data coming into the Serial port and save it it a csv file.
For a given predetermined time (in Arduino_code folder) it collects 1200 points and saves it to a csv file. 
This data is used to train and test the RFMN
'''

import serial

arduino_port = "COM5"
baud = 115200
fileName = "Arduino_to_cvs.csv"
samples = 20
print_labels = False

# Next, set up the serial connection and create the file. You can use the input parameter “w” to write a new file 
# or “a” to append to an existing file.
ser = serial.Serial(arduino_port, baud)
print("Connected to Arduino port: " + arduino_port)
file = open(fileName, "w")
print("Created file")
line = 0


# Skip lines until headers are found
while True:
    getData = ser.readline()
    dataString = getData.decode('utf-8')
    if "Time,data[0],Force,X_axis,Y_axis,Z_axis,Class" in dataString:
        break

# collect the samples
while line <= samples:
    if line == 0:
        data = "Time,Data[0],Force,X_axis,Y_axis,Z_axis,Class"

    else: 
        getData = ser.readline()
        dataString = getData.decode('utf-8')
        data = dataString[0:][:-2]
    print(data)   
    file = open(fileName, "a")
    file.write(data + "\n")
    line = line +1

print("Data collection complete!")