# --- Import Modules --- #
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from RFMN import ReflexFuzzyNeuroNetwork
import time


# --- Import Iris data, split, and scale --- #
data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\Arduino_train_test_labels.csv')


data = data.iloc[:,1:]


X = data.iloc[:,:-1] # for every y (class) we get a 4-D array. E.g., I'm in the 5th dimension. 
y = data.iloc[:,-1] # same as saying y coresponds to the respective classes. E.g., w = 1,2 or 3.


# Creating an object of class MinMax scalar and fiting in X_train
scaler_min_max = MinMaxScaler(feature_range=(0.01, .99))
# X = scaler_min_max.fit_transform(X)
X_norm = scaler_min_max.fit_transform(X)

# My created norm
# X_norm = (X-X.min())/(X.max()-X.min())
# X_norm = X_norm.values
# print(X_norm.shape)
# print(X_norm)

# # Split the data between train and test. 
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42) # Split the data to 33% to test, and 66% to training
                                                      #These value come in four 66X1 matrices for X_train and X_test
                                                      # and one 66X1 matrix for y_train and y_test. 

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.33, random_state=42)


y_train, y_test = y_train.values, y_test.values # Transpose the y_train and y_test data. 
                                    # Essentailly we go from a 66X1 matrices to a 1x66 matrices. 
X_train, X_test = X_train.T, X_test.T # Transpose the X_train and X_test data. 
                                    # Essentailly we go from four 66X1 matrices to four 1x66 matrices. 





# # # --- Declare network --- "
nn = ReflexFuzzyNeuroNetwork(gamma=1, theta=.1)

# --- Train network --- #
nn.train(X_train, y_train)

# print(X.min())
# print(X.max())
print("I was here just wait.... and trust")
# # --- Test Network --- #
nn.test(X_test,y_test)


import time
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate(i, dataList, ser):
      

        ser.write(b'g')                                                                                                                                                                      # Transmit the char 'g' to receive the Arduino data point
        arduinoData_string = ser.readline().decode('ascii') # Decode receive Arduino data as a formatted string
        a = list(map(str.strip, arduinoData_string.split(',')))
       
        #print(i)                                           # 'i' is a incrementing variable based upon frames = x argument

        # print(type(arduinoData_string))

        try:

                # arduinoData_float = float(arduinoData_string)   # Convert to float
                # print("This is Arduino Data ", arduinoData_float)
                # print("This is Arduino Data type ", type(arduinoData_float), "\n")

                # dataList.append(arduinoData_float)
                # print("This is dataList", dataList)
                # print("This is dataList type ", type(dataList), "\n")

                # b = [float(i) for i in a]
                # print(b)
                # print(type(b))  
                # c = b[0]+ b[1]+ b[2]
                # print(c)
                # print(type(c))
                # dataList.append(c)


                print("I was here")
                bb = [float(i) for i in a]
                print("This is df", bb)
                print("This is type df", type(bb[1]))

                norm_ful = (bb-X.min())/(X.max()-X.min())
                print("This is norm_df", norm_ful)
                print("This is type norm_df", type(norm_ful))

                norm_ful = norm_ful.values
                print("This is norm_df.values", norm_ful)  # this has to be the same as b but with balues smaller than 1
                print("This is type norm_df.values", type(norm_ful))  # this has to be the same as b but with balues smaller than 1      
                
                prediction = nn.predict(norm_ful)
                print("This is prediction", prediction) # this has to be the same as c but with balues smaller than 1
                print("This is type prediction", type(prediction)) 

                prediction1 = float(prediction.item())
                print("This is prediction", prediction1)
                print("This is type prediction", type(prediction1)) 

                dataList.append(prediction1)              # Add to the list holding the fixed number of points to animate
                print("I was here too")
                
        except:                                             # Pass if data point is bad                               
                pass    

        dataList = dataList[-50:]                           # Fix the list size so that the animation plot 'window' is x number of points

        ax.clear()                                          # Clear last data frame
        ax.plot(dataList)                                   # Plot new data frame


        # def getPlotFormat(self):
        ax.set_ylim([1, 9])                              # Set Y axis limit of plot
        ax.set_title("Arduino Data")                        # Set title of figure
        ax.set_ylabel("Value")                              # Set title of y axis 

dataList = []                                           # Create empty list variable for later use
                                                        
fig = plt.figure()                                      # Create Matplotlib plots fig is the 'higher level' plot window
ax = fig.add_subplot(111)                               # Add subplot to main fig window

ser = serial.Serial("COM10", 9600)                       # Establish Serial object with COM port and BAUD rate to match Arduino Port/rate
print(ser)

time.sleep(2)                                           # Time delay for Arduino Serial initialization 

                                                        # Matplotlib Animation Fuction that takes takes care of real time plot.
                                                        # Note that 'fargs' parameter is where we pass in our dataList and Serial object. 
ani = animation.FuncAnimation(fig, animate, frames=100, fargs=(dataList, ser), interval=50) 

plt.show()                                              # Keep Matplotlib plot persistent on screen until it is closed
ser.close()                                             # Close Serial connection when plot is closed

    


