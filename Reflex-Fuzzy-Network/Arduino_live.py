import serial
import os

arduino_port = "COM5"
baud = 115200
# fileName = "Arduino_live.csv"

fileName = r"C:\Users\dema2\OneDrive\Desktop\PhD\RFMN\Reflex-Fuzzy-Network\Arduino_live.csv"
samples = 10
print_labels = False


# Next, set up the serial connection and create the file. You can use the input parameter “w” to write a new file 
# or “a” to append to an existing file.
ser = serial.Serial(arduino_port, baud)
print("Connected to Arduino port:" + arduino_port)


# if(os.path.exists(fileName) and os.path.isfile(fileName)):
#   os.remove(fileName)
#   print("file deleted")
# else:
#   print("file not found")

os.remove(fileName)

# f = open(fileName, "w+")
# f.close()

'''

'''
file = open(fileName, "w")
print("Created file")

line = 0


# collect the samples
# while line <= samples:
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

    # line += 1


# print("Data collection complete!")