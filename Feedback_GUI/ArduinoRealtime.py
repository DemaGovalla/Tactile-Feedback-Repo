import serial
import serial.tools.list_ports
# import time
# import matplotlib as plt
# import matplotlib.pyplot as plt
# import numpy as np


# Create a blank instance of the serial port. 
ports = serial.tools.list_ports.comports()
SerialInst = serial.Serial()

portList = []

# Read each port line 
for onePort in ports:10
portList.append(str(onePort))
print(str(onePort))


val = input("select Port: COM")

for x in range(0,len(portList)):
    if portList[x].startswith("COM" + str(val)):
        portVar = "COM" + str(val)
        print(portList[x])

SerialInst.baudrate = 9600
SerialInst.port = portVar
SerialInst.open()

while True:
    if SerialInst.in_waiting:
        try:
            packet = SerialInst.readline()
            print(packet)
            s =packet.decode('utf').rstrip('\n') # stand for UNI-code
            print(s)
        except:
            pass

# plt.ion()
# fig = plt.figure()

# x = list()
# y = list()
# i = 0
# ser = serial.Serial('COM10', 9600)
# ser.close()
# ser.open()


# while True:
#     data = ser.readline()
#     print(data.decode())
#     x.append(i)
#     y.append(data.decode())

#     plt.scatter(i, float(data.decode()))
#     i +=1
#     plt.show()
#     plt.pause(0.001)