from operator import index
from re import L
from unicodedata import name
from tkinter import *
import tkinter as tk
from tkinter import ttk
import os
from os.path import expanduser,isfile
import time, sys, json
from matplotlib import style
from tkinter import messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename

from matplotlib import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.sankey import DOWN

from tkinter import filedialog
# from tkFileDialog import asksaveasfilename,askopenfilename
# from tkMessageBox import *
from serial import Serial
from serial.tools.list_ports import comports
import time, sys, json
from os.path import expanduser,isfile
import serial
import serial.tools.list_ports
import functools

from ast import expr_context
import threading
from pyparsing import col
import signal
import sys


import time
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np



# style.use("ggplot")
fig = Figure(figsize = (5,5), dpi = 100)
fig.suptitle('I-V Curve')
ax = fig.add_subplot(111)
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('Current (A)')
ax.grid(color = "green", linestyle = '--', linewidth = 0.5)


dataList = []  


def _(text):
	return text

class Graphics():
    pass
class VTD_GUI(Tk):


    def __init__(self, *args, **kwargs):

        Tk.__init__(self)
        cwd = os.getcwd().replace("\\","/")
        self.title("Visual Tactile Display")
        self.iconbitmap('{}/Feedback_GUI/VTD.ico'.format(cwd))
        self.config(bg="black")
        self.state('zoomed')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.font = ("Times New Roman", 12)

        # Create a blank instance of the serial port. 
        self.ports = serial.tools.list_ports.comports()
        self.SerialObj = serial.Serial()

        
        # Create Menu for the VTD
        self.VTD_menu = Menu(self)
        self.config(menu = self.VTD_menu)

        # Add File Menu
        self.File_menu = Menu(self.VTD_menu, tearoff=0)
        self.VTD_menu.add_cascade(label = _("File"), menu = self.File_menu)
        # self.File_menu.add_command(label='New')
        self.File_menu.add_command(label=_('Open...'), command=self.onOpen)
        # self.File_menu.add_command(label='Close')
        self.File_menu.add_separator()
        self.File_menu.add_command(label=_('Save'), command=self.onSave)
        # self.File_menu.add_command(label='Save As...')
        self.File_menu.add_separator()
        self.sub_menu = Menu(self.File_menu, tearoff=0)
        self.sub_menu.add_command(label=_('Keyboard Shortcuts'))
        self.sub_menu.add_command(label=_('Color Themes'))
        self.File_menu.add_cascade(label=_("Preferences"), menu=self.sub_menu)
        self.File_menu.add_separator()
        self.File_menu.add_command(label= _("Exit"), command= self.on_closing)

        # Add Edit Menu
        self.Edit_menu = Menu(self.VTD_menu, tearoff=0)
        self.VTD_menu.add_cascade(label = "Edit", menu = self.Edit_menu)
        self.Edit_menu.add_command(label=_("Cut"), command=self.onCut)
        self.Edit_menu.add_command(label=_("Copy"), command=self.onCopy)
        self.Edit_menu.add_command(label=_("Paste"), command=self.onPaste)
        self.Edit_menu.add_separator()
        self.Edit_menu.add_command(label=_("Clear"), command=self.onClear)


        # Add View Menu
        self.View_menu = Menu(self.VTD_menu, tearoff=0)
        self.VTD_menu.add_cascade(label = _("View"), menu = self.View_menu)

        # Add Settings Menu
        self.Settings_menu = Menu(self.VTD_menu, tearoff=0)
        self.VTD_menu.add_cascade(label = _("Settings"), menu = self.Settings_menu)   
        

        # Add a Help Menu
        self.Help_menu = Menu(self.VTD_menu, tearoff=0)
        self.VTD_menu.add_cascade(label = _("Help"), menu = self.Help_menu)
        self.Help_menu.add_command(label=_('Welcome'))
        self.Help_menu.add_command(label=_('About...'), command=self.onClick)
        self.config(menu=self.VTD_menu)



        # Creat a Main Frame
        self.main_frame = Frame(self)
        self.main_frame.pack(fill = BOTH, expand = 1)

        # Create A Canvas
        self.canvas=Canvas(self.main_frame, bg='#4A7A8C', width=500, height=400, scrollregion=(0,0,700,700))
        
        # Add a vertical scrollbar To The Canvas
        self.vertibar=Scrollbar(self.main_frame, orient=VERTICAL)
        self.vertibar.pack(side=RIGHT,fill=Y)
        self.vertibar.config(command=self.canvas.yview)

        # Add a horizontal scrollbar To The Canvas
        self.horibar=Scrollbar(self.main_frame, orient=HORIZONTAL)
        self.horibar.pack(side=BOTTOM,fill=X)
        self.horibar.config(command=self.canvas.xview)

        self.canvas.config(width=500,height=400)
        self.canvas.config(xscrollcommand=self.horibar.set, yscrollcommand=self.vertibar.set)
        self.canvas.pack(expand=True,side=LEFT,fill=BOTH)

        ''' 
        Left Frame:  
        '''
        self.LeftFrame = LabelFrame(self.canvas, text = "Set parameters")
        self.LeftFrame.grid(row = 0, column=0, ipady= 430)

        # set up nasic part of the Menu
        global connect_btn, refresh_btn, graph
        # global root
        # root = Tk()

        self.portBaud = LabelFrame(self.LeftFrame)
        self.portBaud.grid(row = 0, column=0)


        self.port_lable = Label(self.portBaud, text = "Available Port(s): ", bg = "white")
        self.port_lable.grid(row = 0, column =0)

        self.port_bd = Label(self.portBaud, text = "Baude Rate: ", bg = "white")
        self.port_bd.grid(row = 1, column =0)

        refresh_btn = Button(self.portBaud, text = "RESET", command = self.update_coms) 
        refresh_btn.grid(row = 2, column =0)

        connect_btn = Button(self.portBaud, text = "Connect", state = "disabled", command= self.connection)
        connect_btn.grid(row = 2, column =1)
        self.baud_select()
        self.update_coms()

        

        '''
        Right Frame: 
        '''
        self.RightFrame = LabelFrame(self.canvas, text = "Analysis")
        self.RightFrame.grid(row = 0, column=1, ipady = 12, ipadx= 150)




        graph = Graphics()
        graph.canvas = Canvas(self.RightFrame, width = 300, height= 300, bg = "white", highlightthickness=0)
        graph.canvas.grid(row = 0, columnspan=5)

        
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill = BOTH, expand = True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.RightFrame)
        self.toolbar.update()  


        """
        New code
        """


    '''
    Start of new methods
    '''
    def onCopy(self):
        self.text.event_generate("<<Copy>>")

    def onPaste(self):
        self.text.event_generate("<<Paste>>")

    def onPaste(self):
        self.text.event_generate("<<Paste>>")

    def onCut(self):
        self.text.event_generate("<<Cut>>")

    def onClear(self):
        self.text.delete('1.0', END)

    def onClick(self, event=None):
        messagebox.showinfo("Ohh yaaaa!!!","This program written by Dema!\nNow: "+time.strftime('%H:%M:%S'))
    def onOpen(self):
        filename = askopenfilename(filetypes = ((_("Text files"), "*.csv;*.txt"),(_("All files"), "*.*") ))
        if filename == "":
            return
        self.text.delete('1.0', END)
        fd = open(filename,"r")
        for line in fd:
            self.text.insert(INSERT, line)
        fd.close()
    def onSave(self):
        filename = asksaveasfilename(filetypes = ((_("Text files"), "*.csv;*.txt"),(_("All files"), "*.*") ))
        if filename == "":
            return
        fd = open(filename,"w")
        fd.write(self.text.get('1.0', END))
        fd.close()
    def openPort(self):
        return
    def closePort(self):
        return
    def setBaud(self):
        return

    '''
    End of new methods
    '''

    '''
    Start of new-old methods
    '''
    def connect_check(self, args):
        if "-" in clicked_com.get() or "-" in clicked_bd.get():
            connect_btn["state"] = "disable"
        else:
            connect_btn["state"] = "active"

    def baud_select(self):
        global clicked_bd, drop_bd
        clicked_bd = StringVar()
        bds = ["-", "9600", "115200", "256000"]
        clicked_bd.set(bds[0])
        drop_bd = OptionMenu(self.portBaud, clicked_bd, *bds, command = self.connect_check)
        drop_bd.config(width=20)
        drop_bd.grid(row = 1, column=1, padx =5)

    def update_coms(self):
        global clicked_com, drop_COM
        ports = serial.tools.list_ports.comports()
        coms = [com[0] for com in ports]
        coms.insert(0, "-")

        try:
            drop_COM.destroy()
        except:
            pass

        clicked_com = StringVar()
        clicked_com.set(coms[0])
        drop_COM = OptionMenu(self.portBaud, clicked_com, *coms, command = self.connect_check)
        drop_COM.config(width=20)
        drop_COM.grid(row = 0, column=1, padx =5)
        self.connect_check(0)

    def graph_control(self, graph):
        graph.canvas.itemconfig(graph.outer, 
            exten = int(359*graph.sensor/100))
        graph.canvas.itemconfig(graph.text, 
            text = f"{int(graph.sensor)}")

        # graph.canvas.itemconfig(graph.outer, 
        #     exten = int(359*graph.c/100))
        # graph.canvas.itemconfig(graph.text, 
        #     text = f"{int(graph.c)}")

    def readSerial(self):
        global serialData, graph
        average = 0
        sampling = 1
        sample = 0
        while serialData:
            data = ser.readline()





            # if len(data) > 0:
                
            #     # print(" I was here as well")
            #     try: 

            #         # ser.write(b'g') 
            #                                         # Transmit the char 'g' to receive the Arduino data point
            #         arduinoData_string = ser.readline().decode('ascii') # Decode receive Arduino data as a formatted string
            #         a = list(map(str.strip, arduinoData_string.split(',')))
            #         print(arduinoData_string)
            #         print(type(arduinoData_string))
            #         b = [float(i) for i in a]
            #         print(b)
            #         print(type(b))  
            #         c = int(b[0]+ b[1]+ b[2])
            #         print(c)
            #         print(type(c))
            #         # dataList.append(c)
            #         graph.c = c
            #         t2 = threading.Thread(target=self.graph_control, args=(graph,))
            #         t2.deamon = True
            #         t2.start(

            if len(data) > 0:
                try:
                    

                    print(" I was here ")
                    sensor = int(data.decode('utf8'))
                    a = list(map(str.strip, sensor.split(',')))
                    b = [float(i) for i in a]
                    sensor = int(b[0]+ b[1]+ b[2])
                    print(sensor)
                    print(" I was here as well")
                    graph.sensor = sensor
                    t2 = threading.Thread(target=self.graph_control, args=(graph,))
                    t2.deamon = True
                    t2.start()


                    # sensor = int(data.decode('utf8'))
                    # data_sensor = int(data.decode('utf8'))
                    # average += data_sensor
                    # sample += 1
                    # if sample == sampling:
                    #     sensor= int(average/sampling)
                    #     average = 0
                    #     sample = 0
                    #     print(sensor)
                        
                    #     graph.sensor = sensor
                    #     t2 = threading.Thread(target=self.graph_control, args=(graph,))
                    #     t2.deamon = True
                    #     t2.start()

                except:

                    pass
    
    def connection(self):
        global ser, serialData
        if connect_btn["text"] in "Disconnect":
            serialData = False
            connect_btn["text"] = "Connect"
            refresh_btn["state"] = "active"
            drop_bd["state"] = "active"
            drop_COM["state"] = "active"
            
        else:
            serialData = True
            connect_btn["text"] = "Disconnect"
            refresh_btn["state"] = "disable"
            drop_bd["state"] = "disable"
            drop_COM["state"] = "disable"
            port = clicked_com.get()
            baud = clicked_bd.get()
            try:
                ser = serial.Serial(port, baud, timeout =0)
            except:
                pass
            t1 = threading.Thread(target=self.readSerial)
            t1.daemon = True
            t1.start()

    '''
    End of new-old methods
    '''


    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
 


if __name__ == "__main__":

    root = VTD_GUI()
    root.mainloop()
    sys.exit()
    


