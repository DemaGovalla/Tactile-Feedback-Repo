import serial
import os

arduino_port = "COM5"
baud = 115200
fileName = "Arduino_live.csv"
samples = 10
print_labels = False
ser = serial.Serial(arduino_port, baud)
print("Connected to Arduino port:" + arduino_port)
os.remove(fileName)
line = 0

file = open(fileName, "w")
print("Created file")


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

    file = open(fileName, "a")

    file.write(data + "\n")

