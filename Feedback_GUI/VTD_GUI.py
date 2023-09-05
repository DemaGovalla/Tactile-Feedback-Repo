from doctest import master
from locale import currency
from tkinter import *
from tkinter import Tk, Frame, Menu
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import messagebox
from tkinter.messagebox import showerror
from tkinter.font import Font
from PIL import Image, ImageTk

import os
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import append
from pip import main
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
from matplotlib.widgets import Cursor

import datetime
import logging
import time
from atom import Atom
import operator

from tkinter import filedialog

import shutil



import statistics as stats
from cmath import *
from array import *

from GRMMFN import ReflexFuzzyNeuroNetwork

cwd = os.getcwd().replace("\\","/")
atom = Atom()
fig = Figure(figsize = (5,5), dpi = 100)
fig.suptitle('I-V Curve')
ax = fig.add_subplot(111)
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('Current (A)')
ax.grid(color = "green", linestyle = '--', linewidth = 0.5)
i = 0
j = 0
coord = []
cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True, color = 'r', linewidth = 1)
annot = ax.annotate("", xy=(0,0), xytext=(-40,40),textcoords="offset points",
                    bbox=dict(boxstyle='round4', fc='linen',ec='k',lw=1),
                    arrowprops=dict(arrowstyle='-|>'))
annot.set_visible(False)

class SMU2430GUI():
    def __init__(self, master):

        root.title("SMU2430")
        root.iconbitmap('{}/keithley_symbol.ico'.format(cwd))
        root.config(bg="black")
        root.geometry("1200x650")
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        style.use("ggplot")
        LARGE_FONT = ("Verdana", 12)
        
        Open_On = Image.open('{}/On.png'.format(cwd))
        Open_OFF = Image.open('{}/OFF.png'.format(cwd))
        ON_r = Open_On.resize((32,32))
        OFF_r = Open_OFF.resize((32,32))
        self.ON = ImageTk.PhotoImage(ON_r)
        self.OFF = ImageTk.PhotoImage(OFF_r)

        # Keep track of the button state on/off
        global is_off 
        global sweep
        is_off = True
        sweep = True

        self.my_menu = Menu(root)
        root.config(menu = self.my_menu)
        # Add File Menu
        self.file_menu = Menu(self.my_menu, tearoff=0)
        self.my_menu.add_cascade(label = "File", menu = self.file_menu)
        self.file_menu.add_command(label= "Save Screenshot... ", command= self.screenshot)
        self.file_menu.add_command(label= "Save Sweep Data...", command= self.savesweepdata)
        self.file_menu.add_separator()
        self.file_menu.add_command(label= "Exit", command= self.on_closing)

        # Add a help Menu
        self.help_menu = Menu(self.my_menu, tearoff=0)
        self.my_menu.add_cascade(label = "Help", menu = self.help_menu)
        self.help_menu.add_command(label= "User Manual", command = self.help_box)
        self.help_menu.add_command(label= "About", command = self.about_box)


        # Create A Main Frame
        self.main_frame = Frame(master)
        self.main_frame.pack(fill = BOTH, expand = 1)

        # Create A Canvas
        self.my_canvas = Canvas(self.main_frame)
        self.my_canvas.pack(side=LEFT, fill =BOTH, expand=1)

        # Add A Scrollbar To The Canvas
        self.scrollbar_y = Scrollbar(self.main_frame, orient= VERTICAL, command=self.my_canvas.yview)
        self.scrollbar_y.pack(side= RIGHT, fill = Y)
        self.my_canvas.configure(yscrollcommand = self.scrollbar_y.set)
        self.my_canvas.bind('<Configure>', lambda e: self.my_canvas.configure(scrollregion=self.my_canvas.bbox("all")))
        self.Frame = Frame(self.my_canvas)
        self.Frame.grid(row = 0, column=0)
        self.my_canvas.create_window((0,0), window = self.Frame)


        ''' 
        Left Frame: The frame has the ability to change the current and voltages of the SMU 
        '''
        self.LeftFrame = LabelFrame(self.Frame, text = "Set parameters")
        self.LeftFrame.grid(row = 0, column=0, ipady = 57, ipadx = 10)


        # Button to tell the user that the Device is ON or OFF 
        self.Onoffframe = LabelFrame(self.LeftFrame)
        self.Onoffframe.grid(row = 0, column= 0)
        self.off_button_frame = LabelFrame(self.Onoffframe, text = "SMU OFF", fg = "grey")
        self.off_button_frame.grid(row = 0, column= 0)
        self.off_button = Button(self.off_button_frame, image=self.OFF, bd=0, command = self.switch)
        self.off_button.grid(row = 0, column = 0)

        # Enable a voltage sweep, or a current sweep
        self.sweep_labelframe = LabelFrame(self.Onoffframe)
        self.sweep_labelframe.grid(row = 0, column= 1)
        self.sweep_label = Label(self.sweep_labelframe, text="Voltage Sweep", width= 15)
        self.sweep_label.grid(row = 0, column = 0)
        self.sweep_button = Button(self.sweep_labelframe, text="Toggle V/I Sweep", width= 15, command = self.sweep)
        self.sweep_button.grid(row = 1, column = 0)

        # Enable the User to select the GBIP address
        self.GBIP_labelframe = LabelFrame(self.Onoffframe)
        self.GBIP_labelframe.grid(row = 0, column= 2)
        self.GBIP_label = Label(self.GBIP_labelframe, text="Enter GBIP Address", width= 15)
        self.GBIP_label.grid(row = 0, column = 0)
        self.GBIP_Entry = Entry(self.GBIP_labelframe, width= 15, fg = "grey")
        # self.GBIP_Entry.insert(0, "GPIB0::11::INSTR")
        self.GBIP_Entry.insert(0, "GPIB0::11::INSTR")
        self.GBIP_Entry.grid(row = 1, column = 0)
        self.GBIP_button = Button(self.GBIP_labelframe, text="Set", command = self.setGBIP)
        self.GBIP_button.grid(row = 1, column = 1)

        # Always start with sourcing voltage and measuring current
        self.smu1.apply_voltage()
        self.smu1.measure_current()



        # Voltage frames: Frame 1: Used to source voltage (sweep or set a constant values)
                        # Frame 2: Used to get a measured voltage given a sourced current value. 
                        # Frame 3: Used to set a compliance voltage to set a limit on the measured voltage. 
                                 # Also, we can set the range voltage in Frame 3
        self.voltagenotebook = ttk.Notebook(self.LeftFrame)
        self.voltagenotebook.grid(row = 1, column = 0)
        self.voltageframe_1 = LabelFrame(self.voltagenotebook)
        self.voltageframe_1.grid(row = 0, column = 0)
        self.voltageframe_2 = LabelFrame(self.voltagenotebook)
        self.voltageframe_2.grid(row = 0, column = 1)
        self.voltageframe_3 = LabelFrame(self.voltagenotebook)
        self.voltageframe_3.grid(row = 0, column = 2)
        self.voltagenotebook.add(self.voltageframe_1, text = "Source Voltage")
        self.voltagenotebook.add(self.voltageframe_2, text = "Measure Current")
        self.voltagenotebook.add(self.voltageframe_3, text = "Current Compliance")


        # Specifics of Frame 1 - Voltage: 
        self.startvoltage = Label(self.voltageframe_1, text="Voltage Sweep")
        self.startvoltage.grid(row= 0, column=0)
        self.startvoltage = Label(self.voltageframe_1, text="Start (V): ")
        self.startvoltage.grid(row= 1, column=0)
        self.stepvoltage = Label(self.voltageframe_1, text="Step (V): ")
        self.stepvoltage.grid(row= 2, column=0)
        self.timevoltage = Label(self.voltageframe_1, text="Time of a step (s): ")
        self.timevoltage.grid(row= 3, column=0)
        self.stopvoltage = Label(self.voltageframe_1, text="Stop (V): ")
        self.stopvoltage.grid(row= 4, column=0)

        self.startentry_v = Entry(self.voltageframe_1, width=5)
        self.startentry_v.grid(row = 1, column=1)
        self.stepentry_v = Entry(self.voltageframe_1, width=5)
        self.stepentry_v.grid(row = 2, column=1)
        self.timeentry_v = Entry(self.voltageframe_1, width=5)
        self.timeentry_v.grid(row = 3, column=1)
        self.stopentry_v = Entry(self.voltageframe_1, width=5)
        self.stopentry_v.grid(row = 4, column=1)

        self.sweep_voltage = Button(self.voltageframe_1, text = "Start", command = self.sweepvoltagebutton)
        self.sweep_voltage.grid(row = 5, column=0)

        self.setvoltage = Label(self.voltageframe_1, text="Set Voltage")
        self.setvoltage.grid(row= 0, column=2)
        self.setvoltage = Label(self.voltageframe_1, text="Voltage (V): ")
        self.setvoltage.grid(row= 1, column=2)
        self.setentry_v = Entry(self.voltageframe_1, width=5)
        self.setentry_v.grid(row = 1, column=3)
        self.setvoltage_s = Button(self.voltageframe_1, text = "Set", command = self.setvoltagebutton)
        self.setvoltage_s.grid(row = 1, column=4)

        self.c_compliance_dis = Label(self.voltageframe_1, text="Current Compliance")
        self.c_compliance_dis.grid(row= 2, column=2)
        self.c_compliance_dis_entry = Entry(self.voltageframe_1, width=5, state= DISABLED)
        self.c_compliance_dis_entry.insert(0, self.smu1.compliance_current)
        self.c_compliance_dis_entry.grid(row = 3, column=2)





        # Specifics of Frame 2 - Current: 
        # This frame is used to get a measured voltage when we set or sweep through current values. 
        self.Com_current = Label(self.voltageframe_2, text="Get Current")
        self.Com_current.grid(row= 0, column=0)
        self.labelcurrent = Label(self.voltageframe_2, text="Current (A): ")
        self.labelcurrent.grid(row= 1, column=0)
        self.getentry_c = Entry(self.voltageframe_2, width=10, state=DISABLED)
        self.getentry_c.grid(row = 1, column=1)
        self.setcurrent_s = Button(self.voltageframe_2, text = "Get", command = self.getcurrentbutton)
        self.setcurrent_s.grid(row = 2, column=1)


        # Specifics of Frame 3 - Current: 
        self.Com_current = Label(self.voltageframe_3, text="Set Current Compliance")
        self.Com_current.grid(row= 0, column=0)
        self.inputcompliance_C = Entry(self.voltageframe_3, width=5)
        self.inputcompliance_C.grid(row = 1, column=0)
        self.currentcompliance = Button(self.voltageframe_3, text = "Set", command = self.setcompliance_C)
        self.currentcompliance.grid(row = 1, column=1)

        self.Com_current_ = Label(self.voltageframe_3, text="Get Current Compliance")
        self.Com_current_.grid(row= 2, column=0)
        self.inputcompliance_C_ = Entry(self.voltageframe_3, width=5, state=DISABLED)
        self.inputcompliance_C_.grid(row = 3, column=0)
        self.currentcompliance_ = Button(self.voltageframe_3, text = "Get", command = self.getcompliance_C)
        self.currentcompliance_.grid(row = 3, column=1)

        self.range_current = Label(self.voltageframe_3, text="Set Current Range")
        self.range_current.grid(row= 0, column=2)
        self.inputrange_C = Entry(self.voltageframe_3, width=5)
        self.inputrange_C.grid(row = 1, column=2)
        self.currentrange = Button(self.voltageframe_3, text = "Set", command = self.setcurrentrange)
        self.currentrange.grid(row = 1, column=3)

        self.range_current_ = Label(self.voltageframe_3, text="Get Current Range")
        self.range_current_.grid(row= 2, column=2)
        self.inputrange_C_ = Entry(self.voltageframe_3, width=5, state=DISABLED)
        self.inputrange_C_.grid(row = 3, column=2)
        self.currentrange_ = Button(self.voltageframe_3, text = "Get", command = self.getcurrentrange)
        self.currentrange_.grid(row = 3, column=3)


        # Current frames: Frame 1: Used to source current (sweep or set a constant values)
                        # Frame 2: Used to get a measured current given a sourced voltage value. 
                        # Frame 3: Used to set/get a compliance current to set a limit on the measured current. 
                                 # Also, we can set/get the range voltage in Frame 3
        self.currentnotebook = ttk.Notebook(self.LeftFrame)
        self.currentnotebook.grid(row = 2, column = 0)
        self.currentframe_1 = LabelFrame(self.currentnotebook)
        self.currentframe_1.grid(row = 0, column = 0)
        self.currentframe_2 = LabelFrame(self.currentnotebook)
        self.currentframe_2.grid(row = 0, column = 1)
        self.currentframe_3 = LabelFrame(self.currentnotebook)
        self.currentframe_3.grid(row = 0, column = 2)
        self.currentnotebook.add(self.currentframe_1, text = "Source Current")
        self.currentnotebook.add(self.currentframe_2, text = "Measure Voltage")
        self.currentnotebook.add(self.currentframe_3, text = "Voltage Compliance")

        # Specifics of Frame 1 - Current: 
        self.startcurrent = Label(self.currentframe_1, text="Current Sweep")
        self.startcurrent.grid(row= 0, column=0)
        self.startcurrent = Label(self.currentframe_1, text="Start (A): ")
        self.startcurrent.grid(row= 1, column=0)
        self.stepcurrent = Label(self.currentframe_1, text="Step (A): ")
        self.stepcurrent.grid(row= 2, column=0)
        self.timecurrent = Label(self.currentframe_1, text="Time of a step (s): ")
        self.timecurrent.grid(row= 3, column=0)
        self.stopcurrent = Label(self.currentframe_1, text="Stop (A): ")
        self.stopcurrent.grid(row= 4, column=0)

        self.startentry_c = Entry(self.currentframe_1, width=5)
        self.startentry_c.grid(row = 1, column=1)
        self.stepentry_c = Entry(self.currentframe_1, width=5)
        self.stepentry_c.grid(row = 2, column=1)
        self.timeentry_c = Entry(self.currentframe_1, width=5)
        self.timeentry_c.grid(row = 3, column=1)
        self.stopentry_c = Entry(self.currentframe_1, width=5)
        self.stopentry_c.grid(row = 4, column=1)

        self.sweep_current = Button(self.currentframe_1, text = "Start", command = self.sweepcurrentbutton)
        self.sweep_current.grid(row = 5, column=0)

        self.setcurrent = Label(self.currentframe_1, text="Set Current")
        self.setcurrent.grid(row= 0, column=2)
        self.setcurrent = Label(self.currentframe_1, text="Current (A): ")
        self.setcurrent.grid(row= 1, column=2)
        self.setentry_c = Entry(self.currentframe_1, width=5)
        self.setentry_c.grid(row = 1, column=3)
        self.setcurrent_s = Button(self.currentframe_1, text = "Set", command = self.setcurrentbutton)
        self.setcurrent_s.grid(row = 1, column=4)

        self.v_compliance_dis = Label(self.currentframe_1, text="Voltage Compliance")
        self.v_compliance_dis.grid(row= 2, column=2)
        self.v_compliance_dis_entry = Entry(self.currentframe_1, width=5, state= DISABLED)
        self.v_compliance_dis_entry.insert(0, self.smu1.compliance_voltage)
        self.v_compliance_dis_entry.grid(row = 3, column=2)

        # Specifics of Frame 2 - Voltage: 
        # This frame is used to get a measured voltage when we set or sweep through current values. 
        self.Com_voltage = Label(self.currentframe_2, text="Get Voltage")
        self.Com_voltage.grid(row= 0, column=0)
        self.labelvoltage = Label(self.currentframe_2, text="Voltage (V): ")
        self.labelvoltage.grid(row= 1, column=0)
        self.getentry_v = Entry(self.currentframe_2, width=10, state=DISABLED)
        self.getentry_v.grid(row = 1, column=1)
        self.setvoltage_s = Button(self.currentframe_2, text = "Get", command = self.getvoltagebutton)
        self.setvoltage_s.grid(row = 2, column=1)

        # Specifics of Frame 3 - Voltage: 
        self.Com_voltage = Label(self.currentframe_3, text="Set Voltage Compliance")
        self.Com_voltage.grid(row= 0, column=0)
        self.inputcompliance_V = Entry(self.currentframe_3, width=5)
        self.inputcompliance_V.grid(row = 1, column=0)
        self.voltagecompliance = Button(self.currentframe_3, text = "Set", command = self.setcompliance_V)
        self.voltagecompliance.grid(row = 1, column=1)


        self.Com_voltage_ = Label(self.currentframe_3, text="Get Voltage Compliance")
        self.Com_voltage_.grid(row= 2, column=0)
        self.inputcompliance_V_ = Entry(self.currentframe_3, width=5, state=DISABLED)
        self.inputcompliance_V_.grid(row = 3, column=0)
        self.voltagecompliance_ = Button(self.currentframe_3, text = "Get", command = self.getcompliance_V)
        self.voltagecompliance_.grid(row = 3, column=1)

        self.range_voltage = Label(self.currentframe_3, text="Set Voltage Range")
        self.range_voltage.grid(row= 0, column=2)
        self.inputrange_V = Entry(self.currentframe_3, width=5)
        self.inputrange_V.grid(row = 1, column=2)
        self.voltagerange = Button(self.currentframe_3, text = "Set", command = self.setvoltagerange)
        self.voltagerange.grid(row = 1, column=3)


        self.range_voltage_ = Label(self.currentframe_3, text="Get Voltage Range")
        self.range_voltage_.grid(row= 2, column=2)
        self.inputrange_V_ = Entry(self.currentframe_3, width=5, state=DISABLED)
        self.inputrange_V_.grid(row = 3, column=2)
        self.voltagerange_ = Button(self.currentframe_3, text = "Get", command = self.getvoltagerange)
        self.voltagerange_.grid(row = 3, column=3)
        
        for child in self.currentframe_1.winfo_children():
            child.configure(state='disable')
        for child in self.currentframe_1.winfo_children():
            child.configure(state='disable')
        for child in self.currentframe_1.winfo_children():
            child.configure(state='disable')


        # Progress bar
        self.progressbar = ttk.Progressbar(self.LeftFrame, orient= HORIZONTAL, length= 300, mode = 'determinate')
        self.progressbar.grid(row = 3, column = 0, pady =10)
        
        self.connectionFrame = LabelFrame(self.LeftFrame)
        self.connectionFrame.grid(row = 4, column= 0)
        # Reset Button
        self.Reset = Button(self.connectionFrame, text = "Reset", width= 10, command = self.Reset)
        self.Reset.grid(row = 0, column=0)
        # Status Label Button
        self.status = Label(self.connectionFrame, text = "GPIB Connected...", fg = 'green', width= 20)
        self.status.grid(row = 0, column=1)



        '''
        Right Frame: This frame has the I-V curve on it 
        '''
        self.RightFrame = LabelFrame(self.Frame, text = "Analysis")
        self.RightFrame.grid(row = 0, column=1, ipady = 12, ipadx= 150)
        self.canvas = FigureCanvasTkAgg(fig, self.RightFrame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill = BOTH, expand = True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.RightFrame)
        self.toolbar.update()      
        
    '''
    Methods to for setting/getting voltage and current measurements.

    '''

    def sweepvoltagebutton(self):
        fig.supxlabel('Applied Voltage (V)')
        fig.supylabel('Measured Current (A)')
        gettime = str(datetime.datetime.now()).replace(":","-")
        gettime = gettime[:19]         
        atom.datalog_filename = "datalog-{}.csv".format(gettime)
        for v in np.arange(float(self.startentry_v.get()),float(self.stopentry_v.get())+float(self.stepentry_v.get()) , float(self.stepentry_v.get())):
            self.smu1.source_voltage = v
            self.smu1.source_enabled = True
            self.off_button.config(image = self.ON)
            self.off_button_frame.config(text = "SMU ON", fg = "green")
            self.getentry_c.delete(0, END)
            c = self.smu1.current
            self.getentry_c.insert(0, c)
            print(v)
            print(c)
            atom.add_condition("Voltage Source", v)
            atom.add_condition("Current Measurement", c)
            atom.log() 
            d = (100*(float(self.stepentry_v.get()))) / abs((float(self.stopentry_v.get())+float(self.stepentry_v.get()))-float(self.startentry_v.get()))
            self.progressbar['value'] +=d
            root.update_idletasks()
            time.sleep(float(self.timeentry_v.get()))
        self.off_button.config(image = self.OFF)
        self.off_button_frame.config(text = "SMU OFF", fg = "grey")
        self.smu1.source_enabled = False

        with open("C:\ValidationData\datalog-{}.csv".format(gettime), "r") as dynamic_file:
            reader_obj = csv.reader(dynamic_file) 
            with open("C:\ValidationData\static_file.csv", "w") as static_file:
                writer_obj = csv.writer(static_file, delimiter=",", lineterminator = '\n') 
                for data in reader_obj:
                    writer_obj.writerow(data)
        Voltage = []
        Current = []
        with open("C:\ValidationData\static_file.csv", "r") as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            next(lines)
            for row in lines:
                Voltage.append(float(row[1]))
                Current.append(float(row[2]))
        ax.clear()
        ax.plot(Voltage, Current, color="blue", marker="x")
        self.canvas.draw_idle()
        self.progressbar['value'] = 0

    def setvoltagebutton(self):
        self.smu1.source_voltage = float(self.setentry_v.get())

    def getvoltagebutton(self):
        self.off_button.config(image = self.ON)
        self.off_button_frame.config(text = "SMU ON", fg = "green")
        self.smu1.source_enabled = True
        self.getentry_v.configure(state='normal')
        self.getentry_v.delete(0, END)
        self.smu1.voltage
        v = self.smu1.voltage
        self.getentry_v.insert(0, v)
        self.getentry_v.configure(state='disable')
        self.off_button.config(image = self.OFF)
        self.off_button_frame.config(text = "SMU OFF", fg = "grey")
        self.smu1.source_enabled = False

    def setcompliance_V(self):
        self.smu1.compliance_voltage = float(self.inputcompliance_V.get())
        self.v_compliance_dis_entry.config(state='normal')
        self.v_compliance_dis_entry.delete(0, END)
        v = str(self.smu1.compliance_voltage)
        self.v_compliance_dis_entry.insert(0, v)
        self.v_compliance_dis_entry.configure(state='disable')

    def getcompliance_V(self):
        self.off_button.config(image = self.ON)
        self.off_button_frame.config(text = "SMU ON", fg = "green")
        self.smu1.source_enabled = True
        self.v_compliance_dis_entry.config(state='normal')
        self.v_compliance_dis_entry.delete(0, END)
        self.inputcompliance_V_.configure(state='normal')
        self.inputcompliance_V_.delete(0, END)
        v = str(self.smu1.compliance_voltage)
        self.inputcompliance_V_.insert(0, v)
        self.inputcompliance_V_.configure(state='disable')
        self.v_compliance_dis_entry.insert(0, v)
        self.v_compliance_dis_entry.configure(state='disable')
        self.off_button.config(image = self.OFF)
        self.off_button_frame.config(text = "SMU OFF", fg = "grey")
        self.smu1.source_enabled = False

    def setvoltagerange(self):
        self.smu1.source_voltage_range = float(self.inputrange_V.get())

    def getvoltagerange(self):
        self.off_button.config(image = self.ON)
        self.off_button_frame.config(text = "SMU ON", fg = "green")
        self.smu1.source_enabled = True
        self.inputrange_V_.configure(state='normal')
        self.inputrange_V_.delete(0, END)
        v = str(self.smu1.source_voltage_range)
        self.inputrange_V_.insert(0, v)
        self.inputrange_V_.configure(state='disable')
        self.off_button.config(image = self.OFF)
        self.off_button_frame.config(text = "SMU OFF", fg = "grey")
        self.smu1.source_enabled = False

    def sweepcurrentbutton(self):
        fig.supxlabel('Measured Voltage (V)')
        fig.supylabel('Applied Current (A)')
        gettime = str(datetime.datetime.now()).replace(":","-")
        gettime = gettime[:19]         
        atom.datalog_filename = "datalog-{}.csv".format(gettime) 
        for c in np.arange(float(self.startentry_c.get()),float(self.stopentry_c.get())+float(self.stepentry_c.get()) , float(self.stepentry_c.get())):
            self.smu1.source_current = c
            self.off_button.config(image = self.ON)
            self.off_button_frame.config(text = "SMU ON", fg = "green")
            self.smu1.source_enabled = True
            self.getentry_v.delete(0, END)
            v = self.smu1.voltage
            self.getentry_v.insert(0, v)
            print(v)
            print(c)
            atom.add_condition("Voltage Measurement", v)
            atom.add_condition("Current Source", c)
            atom.log() 
            d = 100/ abs((float(self.stopentry_c.get())-float(self.startentry_c.get()))/(float(self.stepentry_c.get())))
            self.progressbar['value'] +=d
            root.update_idletasks()
            time.sleep(float(self.timeentry_c.get()))
        self.off_button.config(image = self.OFF)
        self.off_button_frame.config(text = "SMU OFF", fg = "grey")
        self.smu1.source_enabled = False

        with open("C:\ValidationData\datalog-{}.csv".format(gettime), "r") as dynamic_file:
            reader_obj = csv.reader(dynamic_file) 
            with open("C:\ValidationData\static_file.csv", "w") as static_file:
                writer_obj = csv.writer(static_file, delimiter=",", lineterminator = '\n') 
                for data in reader_obj:
                    writer_obj.writerow(data)
        Voltage = []
        Current = []
        with open("C:\ValidationData\static_file.csv", "r") as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            next(lines)
            for row in lines:
                Voltage.append(float(row[1]))
                Current.append(float(row[2]))
        ax.clear()
        ax.plot(Voltage, Current, color="red", marker="x")
        self.canvas.draw_idle()
        self.progressbar['value'] = 0

    def setcurrentbutton(self):
        self.smu1.source_current = float(self.setentry_c.get())

    def getcurrentbutton(self):
        self.off_button.config(image = self.ON)
        self.off_button_frame.config(text = "SMU ON", fg = "green")
        self.smu1.source_enabled = True
        self.getentry_c.configure(state='normal')
        self.getentry_c.delete(0, END)
        c = self.smu1.current
        self.getentry_c.insert(0, c)
        self.getentry_c.configure(state='disable')
        self.off_button.config(image = self.OFF)
        self.off_button_frame.config(text = "SMU OFF", fg = "grey")
        self.smu1.source_enabled = False


    def setcompliance_C(self):
        self.smu1.compliance_current = float(self.inputcompliance_C.get())
        self.c_compliance_dis_entry.config(state='normal')
        self.c_compliance_dis_entry.delete(0, END)
        c = str(self.smu1.compliance_current)
        self.c_compliance_dis_entry.insert(0, c)
        self.c_compliance_dis_entry.configure(state='disable')

    def getcompliance_C(self):
        self.off_button.config(image = self.ON)
        self.off_button_frame.config(text = "SMU ON", fg = "green")
        self.smu1.source_enabled = True
        self.inputcompliance_C_.configure(state='normal')
        self.inputcompliance_C_.delete(0, END)
        c = str(self.smu1.compliance_current)
        self.inputcompliance_C_.insert(0, c)
        self.inputcompliance_C_.configure(state='disable')

        self.c_compliance_dis_entry.config(state='normal')
        self.c_compliance_dis_entry.delete(0, END)
        self.inputcompliance_C_.insert(0, c)
        self.c_compliance_dis_entry.insert(0, c)
        self.c_compliance_dis_entry.configure(state='disable')
        self.off_button.config(image = self.OFF)
        self.off_button_frame.config(text = "SMU OFF", fg = "grey")
        self.smu1.source_enabled = False
     
    def setcurrentrange(self):
        self.smu1.source_current_range = float(self.inputrange_C.get())
    
    def getcurrentrange(self):
        self.off_button.config(image = self.ON)
        self.off_button_frame.config(text = "SMU ON", fg = "green")
        self.smu1.source_enabled = True
        self.inputrange_C_.configure(state='normal')
        self.inputrange_C_.delete(0, END)
        c = str(self.smu1.source_current_range)
        self.inputrange_C_.insert(0, c)
        self.inputrange_C_.configure(state='disable')
        self.off_button.config(image = self.OFF)
        self.off_button_frame.config(text = "SMU OFF", fg = "grey")
        self.smu1.source_enabled = False

    def sweep(self):
        global sweep
        if sweep:
            self.sweep_label['text'] = "Current Sweep"
            self.smu1.apply_current()
            self.smu1.measure_voltage()
            self.inputrange_V_.configure(state=DISABLED)
            self.inputcompliance_V_.configure(state=DISABLED)
            self.getentry_v.configure(state=DISABLED)
            for child in self.voltageframe_1.winfo_children():
                child.configure(state='disable')
            for child in self.voltageframe_2.winfo_children():
                child.configure(state='disable')
            for child in self.voltageframe_3.winfo_children():
                child.configure(state='disable')
            for child in self.currentframe_1.winfo_children():
                child.configure(state='normal')
            for child in self.currentframe_2.winfo_children():
                child.configure(state='normal')
            for child in self.currentframe_3.winfo_children():
                child.configure(state='normal')
            self.v_compliance_dis_entry.config(state='normal')
            self.v_compliance_dis_entry.delete(0, END)
            v = str(self.smu1.compliance_voltage)
            self.v_compliance_dis_entry.insert(0, v)
            self.v_compliance_dis_entry.configure(state='disable')
            self.inputcompliance_V_.configure(state='normal')
            self.inputcompliance_V_.delete(0, END)
            v = str(self.smu1.compliance_voltage)
            self.inputcompliance_V_.insert(0, v)
            self.inputcompliance_V_.configure(state='disable')
            self.inputrange_V_.configure(state='normal')
            self.inputrange_V_.delete(0, END)
            v = str(self.smu1.source_voltage_range)
            self.inputrange_V_.insert(0, v)
            self.inputrange_V_.configure(state='disable')
            sweep = False  
        else:
            self.sweep_label['text'] = "Voltage Sweep"
            self.smu1.apply_voltage()
            self.smu1.measure_current()
            self.inputrange_C_.config(state=DISABLED)
            self.inputcompliance_C_.config(state=DISABLED)
            self.getentry_c.config(state=DISABLED)
            for child in self.voltageframe_1.winfo_children():
                child.configure(state='normal')
            for child in self.voltageframe_2.winfo_children():
                child.configure(state='normal')
            for child in self.voltageframe_3.winfo_children():
                child.configure(state='normal')
            for child in self.currentframe_1.winfo_children():
                child.configure(state='disable')
            for child in self.currentframe_2.winfo_children():
                child.configure(state='disable')
            for child in self.currentframe_3.winfo_children():
                child.configure(state='disable')
            self.c_compliance_dis_entry.config(state='normal')
            self.c_compliance_dis_entry.delete(0, END)
            c = str(self.smu1.compliance_current)
            self.c_compliance_dis_entry.insert(0, c)
            self.c_compliance_dis_entry.configure(state='disable')
            self.inputcompliance_C_.configure(state='normal')
            self.inputcompliance_C_.delete(0, END)
            c = str(self.smu1.compliance_current)
            self.inputcompliance_C_.insert(0, c)
            self.inputcompliance_C_.configure(state='disable')
            self.inputrange_C_.configure(state='normal')
            self.inputrange_C_.delete(0, END)
            c = str(self.smu1.source_current_range)
            self.inputrange_C_.insert(0, c)
            self.inputrange_C_.configure(state='disable')


            sweep = True

    def switch(self):
        global is_off 
        if is_off:
            self.off_button.config(image = self.ON)
            self.off_button_frame.config(text = "SMU ON", fg = "green")
            self.smu1.source_enabled = True
            is_off = False
        else: 
            self.off_button.config(image=self.OFF)
            self.off_button_frame.config(text = "SMU OFF", fg = "grey")
            self.smu1.source_enabled = False
            is_off = True


            
        

    def help_box(self):
        win = Toplevel(root)
        configfile = Text(win, wrap=WORD, width=100, height= 100)
        configfile.pack(fill="none", expand=TRUE)
        with open("{}\README.txt".format(cwd), "r") as f:
            configfile.insert(INSERT, f.read())  

    def about_box(self):
        win = Toplevel(root)
        configfile = Text(win, wrap=WORD, width=45, height= 20)
        configfile.pack(fill="none", expand=TRUE)
        with open("{}\Aboutfile.txt".format(cwd), "r") as f:
            configfile.insert(INSERT, f.read()) 

    def onclick(self):
        global coord
        coord.append((self.xdata, self.ydata))
        xdata = self.xdata
        ydata = self.ydata
        annot.xy = (xdata, ydata)
        text = "({:.4g}, {:.4g})".format(xdata, ydata)
        annot.set_text(text)
        annot.set_visible(True)
        fig.canvas.draw() 

    def on_press(self):
        self.canvas.draw()
        global i,j 
        global coord
        Voltage = []
        Current = []
        with open("C:\ValidationData\static_file.csv", "r") as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            next(lines)
            for row in lines:
                Voltage.append(float(row[1]))
                Current.append(float(row[2]))
        if self.key == 'right':
            i += 1
            j += 1 
            annot.xy = (Voltage[i], Current[i])
            text = "({:.4g}, {:.4g})".format(Voltage[i], Current[i])
            annot.set_text(text)
            annot.set_visible(True)
            fig.canvas.draw()
        elif self.key == 'left':
            i -=1
            j -=1
            annot.xy = (Voltage[i], Current[i])
            text = "({:.4g}, {:.4g})".format(Voltage[i], Current[i])
            annot.set_text(text)
            annot.set_visible(True)
            fig.canvas.draw()
        else:
            return

    def screenshot(self):
        myScreenshot = pyautogui.screenshot()
        file_path = filedialog.asksaveasfilename(filetypes=(
                    ("Portable Network Graphic (PNG)", "*.PNG"),
                    ("Comma Separated Values (CSV)", "*.csv"),
                    ("Text files", "*.txt"),
                    ("Python Files (py)", "*.py"),
                    ("All files", "*.*")), defaultextension='.png')
        
        myScreenshot.save(file_path)

    def savesweepdata(self):
        original = r'C:\ValidationData\static_file.csv', "r"
        file_path = filedialog.asksaveasfilename(filetypes=(
                    ("Comma Separated Values (CSV)", "*.csv"),
                    ("Portable Network Graphic (PNG)", "*.PNG"),
                    ("Text files", "*.txt"),
                    ("Python Files (py)", "*.py"),
                    ("All files", "*.*")), defaultextension='.png')
        shutil.copyfile(original, file_path)
        return

    def on_closing(self):
        self.off_button.config(image=self.OFF)
        self.off_button_frame.config(text = "SMU OFF", fg = "grey")
        self.smu1.source_enabled = False
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()

    def Reset(self):
        self.startentry_v.delete(0, END)
        self.stepentry_v.delete(0, END)
        self.timeentry_v.delete(0, END)
        self.stopentry_v.delete(0, END)
        self.startentry_c.delete(0, END)
        self.stepentry_c.delete(0, END)
        self.timeentry_c.delete(0, END)
        self.stopentry_c.delete(0, END)

if __name__ == "__main__":
    root = Tk()
    smu = SMU2430GUI(root)
    fig.canvas.mpl_connect('key_press_event',SMU2430GUI.on_press)
    fig.canvas.mpl_connect('button_press_event', SMU2430GUI.onclick)
    root.mainloop()