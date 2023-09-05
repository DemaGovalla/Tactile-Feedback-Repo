import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename
import time, sys, json

import os

import matplotlib
matplotlib.use("TkAgg")

from matplotlib import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation 
from matplotlib import style

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import urllib
import json

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from GRMMFN import ReflexFuzzyNeuroNetwork


from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef


import matplotlib.ticker as mticker



import asyncio
import sys
import datetime as dt

import time
import os

from bleak import BleakScanner, BleakClient, discover
from bleak.exc import BleakError
from bleak.backends.scanner import AdvertisementData
from bleak.backends.device import BLEDevice



HAPTIC_SERVICE_UUID = "18424398-7CBC-11E9-8F9E-2A86E4085A59"
PWM2_DUTY_CYCLE_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE20"
PWM2_T1_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE21"
PWM2_T2_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE22"

PWM3_DUTY_CYCLE_UUID = "5A87B4EF-3BFA-76A8-E642-92933C31434F"
PWM3_T1_UUID = "5A87B4EF-3BFA-76A8-E642-92933C314350"
PWM3_T2_UUID = "5A87B4EF-3BFA-76A8-E642-92933C314351"

# All BLE devices have MTU of at least 23. Subtracting 3 bytes overhead, we can
# safely send 20 bytes at a time to any device supporting this service.
UART_SAFE_SIZE = 20



LARGE_FONT = ("Verdana" , 12)
NORM_FONT = ("Verdana" , 10)
SMALL_FONT = ("Verdana" , 8)
style.use("ggplot")


f = Figure()
a = f.add_subplot(111)

y_label = []
x_label = []
pred_x = 0

global selected 
selected = False

def onCut(e):
    # global selected
    # if my_text.selection_get():
    #     #Grab selected text from text box
    #     selected = my_text.selection_get()
    #     # Delete selected text from text box
    #     my_text.delete("sel.first", "sel.last")
    pass

def onPaste(e):
    # if selected:
    #     posistion = my_text.index(INSERT)
    #     my_text.insert(posistion, selected)
    pass


def onCopy(e):
    pass



# def onClear(self):
#     self.text.delete('1.0', END)

# def onClick(self, event=None):
#     messagebox.showinfo("Ohh yaaaa!!!","This program written by Dema!\nNow: "+time.strftime('%H:%M:%S'))
# def onOpen(self):
#     filename = askopenfilename(filetypes = ((_("Text files"), "*.csv;*.txt"),(_("All files"), "*.*") ))
#     if filename == "":
#         return
#     self.text.delete('1.0', END)
#     fd = open(filename,"r")
#     for line in fd:
#         self.text.insert(INSERT, line)
#     fd.close()
# def onSave(self):
#     filename = asksaveasfilename(filetypes = ((_("Text files"), "*.csv;*.txt"),(_("All files"), "*.*") ))
#     if filename == "":
#         return
#     fd = open(filename,"w")
#     fd.write(self.text.get('1.0', END))
#     fd.close()

"""

"""
data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\Arduino_train_test_labels.csv')


data = data.iloc[:,1:]


X = data.iloc[:,:-1] # for every y (class) we get a 4-D array. E.g., I'm in the 5th dimension. 
y = data.iloc[:,-1] # same as saying y coresponds to the respective classes. E.g., w = 1,2 or 3.


'''
RFMN network 
'''
# scaler_min_max = MinMaxScaler(feature_range=(0.01, .99))
# X_norm = scaler_min_max.fit_transform(X)

# # My created norm
# # X_norm = (X-X.min())/(X.max()-X.min())
# # X_norm = X_norm.values
# # print(X_norm.shape)
# # print(X_norm)

# X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.33, random_state=42)
# y_train, y_test = y_train.values, y_test.values # Transpose the y_train and y_test data. 
#                                     # Essentailly we go from a 66X1 matrices to a 1x66 matrices. 
# X_train, X_test = X_train.T, X_test.T # Transpose the X_train and X_test data. 
#                                     # Essentailly we go from four 66X1 matrices to four 1x66 matrices. 
# nn = ReflexFuzzyNeuroNetwork(gamma=1, theta=.1)
# nn.train(X_train, y_train)
# print("I was here just wait.... and trust")
# nn.test(X_test,y_test)
'''
END of RF Netwok
'''



'''
RF network 
'''
rf= RandomForestClassifier()  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rf.fit(X_train,y_train)
y_predlr = rf.predict(X_test)
accuracy_score1 = accuracy_score(y_test, y_predlr)
print("Accuracy: ", accuracy_score1)
'''
END of RF Netwok
'''


def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill = "x", pady=10)
    B1 = ttk.Button(popup, text = "Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()



def animate(i):  # i for interal

        # data1 = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\Arduino_train_test_labels.csv')


        # data1 = data1.iloc[:,1:]


        # X = data1.iloc[:,:-1] # for every y (class) we get a 4-D array. E.g., I'm in the 5th dimension. 
        # y = data1.iloc[:,-1] # same as saying y coresponds to the respective classes. E.g., w = 1,2 or 3.

        ful = []
        global pred_x
        global pred_y
        data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\Arduino_live.csv')

        # x = data['x_value']
        y1 = data['Displacement']
        y2 = data['Force']
        y3 = data['Work']


        len1 = y1.size
        len2 = y2.size
        len3 = y3.size



        ful.append(y1[len1-1])
        ful.append(y2[len2-1])
        ful.append(y3[len3-1])


        # prediction = ful[0]+ful[1]+ful[2]
        # # print("This is type prediction", type(prediction))
        # prediction1 = float(prediction.item())
        # print("This is type prediction", type(prediction1))


        '''
        RFMN
        '''
        # norm_ful = (ful-X.min())/(X.max()-X.min())
        # norm_ful = norm_ful.values
        # prediction = nn.predict(norm_ful)



        '''
        RF
        '''
        norm_ful = (ful-X.min())/(X.max()-X.min())
        prediction = rf.predict([norm_ful])





        prediction1 = float(prediction.item())
        
        # prediction = ful[0] + ful[1] + ful[2]

        y = [0, 1, 2, 3]

        print(prediction1)


        if prediction1 < 3:
            print("very soft")
            pred_y = y[0]
            # print(pred_y)


        elif 3 <= prediction1 < 4:
            print("soft")
            pred_y = y[1]
            # print(pred_y)



        elif 4 <= prediction1 < 7:
            print("med")
            pred_y = y[2]
            # print(pred_y)

        else:
            print("hard")
            pred_y = y[3]
            # print(pred_y)

   
         
        y_label.append(pred_y)
        # y_label.append(prediction1)

        print(y_label)

        x_label.append(pred_x)
      

        pred_x = pred_x + 1

        data_y = pd.Series(y_label)
        data_x = pd.Series(x_label)

        a.clear()
        a.plot(data_x, data_y, "#00A3E0", label = "Deformation")

        # a.set_yticks(data_y, label = 'Tom' 'Dick' 'Harry' 'Slim')
        # a.set_yticks(y[0::3])
        a.set_yticks(y[0:4:1])


        # a.set_yticklabels(['very soft',' ', 'soft', '', 'medium', '', 'high'])
        a.set_yticklabels(['very soft', 'soft',  'medium',  'high'])



        a.set_xlim(i-30, i+10)                            

        a.legend(bbox_to_anchor = (0, 1.02, 1, .102), loc = 3, ncol = 2, borderaxespad = 0)
        a.set_ylim([0, 3])                              # Set Y axis limit of plot
        tittle = "DEFORMATION TYPE\nLast Label: "+ str(y_label[-1])
        a.set_title(tittle)                        # Set title of figure
        a.set_ylabel("Deformation")   
        a.set_xlabel("TIme (50ms)")   

        
    # pullData = open(r"C:\Users\dema2\OneDrive\Desktop\PhD\RFMN\Feedback_GUI\sampleData.txt", "r").read()
    # dataList = pullData.split('\n')
    # xList = []
    # yList = []
    # for eachLine in dataList:
    #     if len(eachLine) > 1:
    #         x, y = eachLine.split(',')
    #         xList.append(int(x))
    #         yList.append(int(y))
    # a.clear()
    # a.plot(xList, yList)


class VTDapp(tk.Tk):

    def __init__(self, *args, **kwargs): # args: arguments (anythng you want) ; kwargs: keyword arguments (dictionaries)
        tk.Tk.__init__(self, *args, **kwargs)

        
        # self.title("Visual Tactile Display")
        # self.iconbitmap('{}/Feedback_GUI/VTD.ico'.format(cwd))
        # self.config(bg="black")
        # self.state('zoomed')
        # self.protocol("WM_DELETE_WINDOW", self.on_closing)


        tk.Tk.iconbitmap(self, default= r"C:\Users\dema2\OneDrive\Desktop\PhD\RFMN\Feedback_GUI\VTD.ico")
        tk.Tk.wm_title(self, "Visual Tactile Display")


        container = tk.Frame(self)
        container.pack(side = "top", fill="both", expand= True)
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)


        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label = "Save settings", command= lambda: popupmsg("Not supported just yet!"))
        filemenu.add_separator()
        filemenu.add_command(label = "Exit", command = quit)
        menubar.add_cascade(label = "File", menu = filemenu)


        editmenu = tk.Menu(menubar, takefocus=1)
        editmenu.add_command(label="Cut", command= lambda: onCut(False))


        editmenu.add_command(label="Copy", command= lambda: onCopy(False))
        editmenu.add_command(label="Paste", command=lambda: onPaste(False))
        # editmenu.add_separator()
        # editmenu.add_command(label="Clear", command=onClear)
        menubar.add_cascade(label = "Edit", menu = editmenu)


        # Add View Menu
        view_menu = tk.Menu(menubar, takefocus=1)
        menubar.add_cascade(label = "View", menu = view_menu)

        # Add Settings Menu
        settingsmenu = tk.Menu(menubar, takefocus=1)
        menubar.add_cascade(label = "Settings", menu = settingsmenu)   
        

        # Add a Help Menu
        helpmenu = tk.Menu(menubar, takefocus=1)
        helpmenu.add_command(label='Welcome')
        helpmenu.add_command(label='About...', command=lambda: popupmsg("Not supported just yet!"))
        menubar.add_cascade(label ="Help", menu = helpmenu)

        tk.Tk.config(self, menu = menubar)

    


        # specify a dictionary
        self.frames = {}

        for F in (StartPage, Deformation_Page): 

            frame = F(container, self)

            self.frames[F] = frame
            
            frame.grid(row = 0, column = 0, sticky= "nsew")

        self.show_frame(StartPage)





    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            app.destroy()

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()



class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text = """ALPHA Bitcoin trading application: 
        use at your own risk - change this later""", font = LARGE_FONT)
        label.pack(pady = 10, padx=10)

        button1 = ttk.Button(self, text = "Agree",
                            command=lambda: controller.show_frame(Deformation_Page))
        button1.pack()


        button2 = ttk.Button(self, text = "Disagree",
                              command=quit)
        button2.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text = "Page One", font = LARGE_FONT)
        label.pack(pady = 10, padx=10)

        button1 = ttk.Button(self, text = "Back to Home",
                              command=lambda: controller.show_frame(StartPage))
        button1.pack()



class Deformation_Page(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text = "Graph Page", font = LARGE_FONT)
        label.pack(pady = 10, padx=10)

        button1 = ttk.Button(self, text = "Back to Home",
                              command=lambda: controller.show_frame(StartPage))
        button1.pack()


        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expan = True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expan = True)

app = VTDapp()
app.geometry("1280x720")
ani = animation.FuncAnimation(f, animate, interval = 50)
app.mainloop()

