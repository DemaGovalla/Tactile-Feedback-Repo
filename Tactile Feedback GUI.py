<<<<<<< HEAD
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData_def.csv file using the TSC-LS algorithm. 
            After traning and testing, it returns the algorithms metrics such as accuracy, presision and more.
            The file performs cross validation for different TSC-LS parameters and returns the classification
            report. 
"""

import random, matplotlib, tkinter as tk, numpy as np, csv, matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style
from tkinter import ttk
matplotlib.use("TkAgg")

LARGE_FONT = ("Verdana", 12)
LARGE_FONT_BOLD = ("Verdana", 12, "bold")
NORM_FONT = ("Verdana", 10)
SMALL_FONT = ("Verdana", 8)

style.use("ggplot")

f = Figure()
a1 = f.add_subplot(211)
a2 = f.add_subplot(212)
f.subplots_adjust(hspace=0.5)

counter = 0
x_values = []  
y_label_def = [] 
y_label_tex = [] 


Type = "Deformation"
dataCounter = 9000
programName = "tfd"

def average_filter(column_values):
    average_value = np.mean(column_values)
    return average_value

def median_filter(column_values):
    median_value = np.median(column_values)
    return median_value

def read_last_n_rows(file_path, n):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        rows = list(csv_reader)
        if len(rows) < n:
            rows = [rows[-1]] * (n - len(rows)) + rows
        last_n_rows = rows[-n:]
    return last_n_rows

def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
    B1.pack()
    popup.mainloop()

def changeType(toWhat, pn):
    global Type
    global dataCounter
    global programName
    
    Type = toWhat
    programName = pn
    dataCounter = 9000

def tutorial():
        
    def page2():
        tut.destroy()
        tut2 = tk.Tk()
        
        def page3():
            tut2.destroy()
            tut3 = tk.Tk()
            
            tut3.wm_title("Part 3!")
            
            label = ttk.Label(tut3, text="Part 3", font=NORM_FONT)
            label.pack(side="top", fill="x", pady=10)
            B1 = ttk.Button(tut3, text="Done!", command=tut3.destroy)
            B1.pack()
            tut3.mainloop()      
            
            
        tut2.wm_title("Part 2!")
        
        label = ttk.Label(tut2, text="Part 3", font=NORM_FONT)
        label.pack(side="top", fill="x", pady=10)
        B1 = ttk.Button(tut2, text="Next", command=page3)
        B1.pack()
        tut2.mainloop() 
        
    tut = tk.Tk()
    tut.wm_title("Tutorial")
    label = ttk.Label(tut, text="What do you need help with?", font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
            
    B1 = ttk.Button(tut, text="Overview of the application", command=page2)
    B1.pack()
  
    B1 = ttk.Button(tut, text="How to save data", command=lambda:popupmsg("Not yet completed"))
    B1.pack()
    
    B1 = ttk.Button(tut, text="About...", command=lambda:popupmsg("Not yet completed"))
    B1.pack()  
    
    tut.mainloop()        
                  
            

def animate(i):
    
    global counter 
    global y_label_def
    global y_label_tex
    global current_def_value
    global current_tex_value
    global def_interval_counter
    global tex_interval_counter
    
    
        # Initialize variables if not already defined
    if 'def_interval_counter' not in globals():
        def_interval_counter = 0
    if 'tex_interval_counter' not in globals():
        tex_interval_counter = 0
    
    # Check if it's time to change the value for y_label_def
    if def_interval_counter == 0:
        current_def_value = random.randint(1, 4)
        def_interval_counter = random.randint(6, 10)
    else:
        def_interval_counter -= 1
    
    # Check if it's time to change the value for y_label_tex
    if tex_interval_counter == 0:
        current_tex_value = random.randint(1, 4)
        tex_interval_counter = random.randint(6, 10)
    else:
        tex_interval_counter -= 1
    
    # Append the current values to the lists
    y_label_def.append(current_def_value)
    y_label_tex.append(current_tex_value)
    
    
    label_locations_def = [1, 2, 3, 4, 5]
    labels_def = ['Very Soft', 'Soft', 'Hard', 'Very Hard', ''] 
        
    label_locations_tex = [1, 2, 3, 4, 5]
    labels_tex = ['Very Smooth', 'Smooth', 'Rough', 'Very Rough', ''] 
         
         
    # Mapping dictionary for numerical labels to string labels
    def_labels_map = {1: 'Very Soft', 2: 'Soft', 3: 'Hard', 4: 'Very Hard', 5: ''} 
    tex_labels_map = {1: 'Very Smooth', 2: 'Smooth', 3: 'Rough', 4: 'Very Rough', 5: ''}

    x_values.append(counter)  
    counter += 1  
    lower_limit = max(0, len(x_values) - 100) 
    
    ''' Start plotting '''
    
    a1.set_xlim(lower_limit, len(x_values))
    a1.set_yticks(label_locations_def)
    a1.set_yticklabels(labels_def)
    a1.set_ylim(0, 5)
    title = "Deformation Values\nLast Value: "+ def_labels_map[y_label_def[-1]]
    a1.set_title(title)
    
    
    a2.set_xlim(lower_limit, len(x_values))
    a2.set_yticks(label_locations_tex)
    a2.set_yticklabels(labels_tex)
    a2.set_ylim(0, 5)
    title = "Texture Values\nLast Value: "+ tex_labels_map[y_label_tex[-1]]
    a2.set_title(title)
    
    plt.tight_layout()
    
    a1.plot(x_values, y_label_def, linestyle='-', color='darkcyan', label="Deformation Type")
    a2.plot(x_values, y_label_tex, linestyle='-', color='maroon', label="Texture Type")

    
 
class TactileFeedback(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        tk.Tk.iconbitmap(self, default='TDF logo.ico')
        tk.Tk.wm_title(self, "Tactile Feedback Display")
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save settings", command=lambda:popupmsg("Not supported just yet!"))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)
        
        typeChoice = tk.Menu(menubar, tearoff=1)
        typeChoice.add_command(label="Deformation",
                                    command=lambda: changeType("Deformation", "deformation"))
        typeChoice.add_command(label="Texture",
                                    command=lambda: changeType("Texture", "texture"))
        menubar.add_cascade(label="Type", menu=typeChoice)
        
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_cascade(label="Tutorial", command=tutorial)
        
        menubar.add_cascade(label="Help", menu=helpmenu)
        
            
        
        tk.Tk.config(self, menu=menubar)
        self.frames = {}
        for F in (StartPage, PageOne, TFD_Page): 
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)
        
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="""Tactile Feedback Display Application: """, font=LARGE_FONT_BOLD)
        label.pack(pady = 10, padx = 10)
        
        label1 = tk.Label(self, text= """ Used to detect defromation and texture 
        labels of a given object""", font=LARGE_FONT)
        label1.pack(padx = 10)
        
        
        # Button to switch pages
        button1 = ttk.Button(self, text = "Agree", 
                            command=lambda: controller.show_frame(TFD_Page) )
        button1.pack()
        button2 = ttk.Button(self, text = "Disagree", 
                            command=quit)
        button2.pack()
        
class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=" Page One!", font=LARGE_FONT)
        label.pack(pady = 10, padx = 10)
        
        # Button to switch pages
        button1 = ttk.Button(self, text="Back to Home", 
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

class TFD_Page(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph Page", font=LARGE_FONT)
        label.pack(pady = 10, padx = 10)
        
        # Button to switch pages
        button1 = ttk.Button(self, text="Back to Home", 
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()
       
        # Plot graph on grpah page
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add Navigation Bar to graph page
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
app = TactileFeedback()
app.geometry("1280x720")
ani = animation.FuncAnimation(f, animate, interval=20, cache_frame_data=False)

app.mainloop()
    
=======
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData_def.csv file using the TSC-LS algorithm. 
            After traning and testing, it returns the algorithms metrics such as accuracy, presision and more.
            The file performs cross validation for different TSC-LS parameters and returns the classification
            report. 
"""

import random, matplotlib, tkinter as tk, numpy as np, csv, matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style
from tkinter import ttk
matplotlib.use("TkAgg")

LARGE_FONT = ("Verdana", 12)
LARGE_FONT_BOLD = ("Verdana", 12, "bold")
NORM_FONT = ("Verdana", 10)
SMALL_FONT = ("Verdana", 8)

style.use("ggplot")

f = Figure()
a1 = f.add_subplot(211)
a2 = f.add_subplot(212)
f.subplots_adjust(hspace=0.5)

counter = 0
x_values = []  
y_label_def = [] 
y_label_tex = [] 


Type = "Deformation"
dataCounter = 9000
programName = "tfd"

def average_filter(column_values):
    average_value = np.mean(column_values)
    return average_value

def median_filter(column_values):
    median_value = np.median(column_values)
    return median_value

def read_last_n_rows(file_path, n):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        rows = list(csv_reader)
        if len(rows) < n:
            rows = [rows[-1]] * (n - len(rows)) + rows
        last_n_rows = rows[-n:]
    return last_n_rows

def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
    B1.pack()
    popup.mainloop()

def changeType(toWhat, pn):
    global Type
    global dataCounter
    global programName
    
    Type = toWhat
    programName = pn
    dataCounter = 9000

def tutorial():
        
    def page2():
        tut.destroy()
        tut2 = tk.Tk()
        
        def page3():
            tut2.destroy()
            tut3 = tk.Tk()
            
            tut3.wm_title("Part 3!")
            
            label = ttk.Label(tut3, text="Part 3", font=NORM_FONT)
            label.pack(side="top", fill="x", pady=10)
            B1 = ttk.Button(tut3, text="Done!", command=tut3.destroy)
            B1.pack()
            tut3.mainloop()      
            
            
        tut2.wm_title("Part 2!")
        
        label = ttk.Label(tut2, text="Part 3", font=NORM_FONT)
        label.pack(side="top", fill="x", pady=10)
        B1 = ttk.Button(tut2, text="Next", command=page3)
        B1.pack()
        tut2.mainloop() 
        
    tut = tk.Tk()
    tut.wm_title("Tutorial")
    label = ttk.Label(tut, text="What do you need help with?", font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
            
    B1 = ttk.Button(tut, text="Overview of the application", command=page2)
    B1.pack()
  
    B1 = ttk.Button(tut, text="How to save data", command=lambda:popupmsg("Not yet completed"))
    B1.pack()
    
    B1 = ttk.Button(tut, text="About...", command=lambda:popupmsg("Not yet completed"))
    B1.pack()  
    
    tut.mainloop()        
                  
            

def animate(i):
    
    global counter 
    global y_label_def
    global y_label_tex
    global current_def_value
    global current_tex_value
    global def_interval_counter
    global tex_interval_counter
    
    
        # Initialize variables if not already defined
    if 'def_interval_counter' not in globals():
        def_interval_counter = 0
    if 'tex_interval_counter' not in globals():
        tex_interval_counter = 0
    
    # Check if it's time to change the value for y_label_def
    if def_interval_counter == 0:
        current_def_value = random.randint(1, 4)
        def_interval_counter = random.randint(6, 10)
    else:
        def_interval_counter -= 1
    
    # Check if it's time to change the value for y_label_tex
    if tex_interval_counter == 0:
        current_tex_value = random.randint(1, 4)
        tex_interval_counter = random.randint(6, 10)
    else:
        tex_interval_counter -= 1
    
    # Append the current values to the lists
    y_label_def.append(current_def_value)
    y_label_tex.append(current_tex_value)
    
    
    label_locations_def = [1, 2, 3, 4, 5]
    labels_def = ['Very Soft', 'Soft', 'Hard', 'Very Hard', ''] 
        
    label_locations_tex = [1, 2, 3, 4, 5]
    labels_tex = ['Very Smooth', 'Smooth', 'Rough', 'Very Rough', ''] 
         
         
    # Mapping dictionary for numerical labels to string labels
    def_labels_map = {1: 'Very Soft', 2: 'Soft', 3: 'Hard', 4: 'Very Hard', 5: ''} 
    tex_labels_map = {1: 'Very Smooth', 2: 'Smooth', 3: 'Rough', 4: 'Very Rough', 5: ''}

    x_values.append(counter)  
    counter += 1  
    lower_limit = max(0, len(x_values) - 100) 
    
    ''' Start plotting '''
    
    a1.set_xlim(lower_limit, len(x_values))
    a1.set_yticks(label_locations_def)
    a1.set_yticklabels(labels_def)
    a1.set_ylim(0, 5)
    title = "Deformation Values\nLast Value: "+ def_labels_map[y_label_def[-1]]
    a1.set_title(title)
    
    
    a2.set_xlim(lower_limit, len(x_values))
    a2.set_yticks(label_locations_tex)
    a2.set_yticklabels(labels_tex)
    a2.set_ylim(0, 5)
    title = "Texture Values\nLast Value: "+ tex_labels_map[y_label_tex[-1]]
    a2.set_title(title)
    
    plt.tight_layout()
    
    a1.plot(x_values, y_label_def, linestyle='-', color='darkcyan', label="Deformation Type")
    a2.plot(x_values, y_label_tex, linestyle='-', color='maroon', label="Texture Type")

    
 
class TactileFeedback(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        tk.Tk.iconbitmap(self, default='TDF logo.ico')
        tk.Tk.wm_title(self, "Tactile Feedback Display")
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save settings", command=lambda:popupmsg("Not supported just yet!"))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)
        
        typeChoice = tk.Menu(menubar, tearoff=1)
        typeChoice.add_command(label="Deformation",
                                    command=lambda: changeType("Deformation", "deformation"))
        typeChoice.add_command(label="Texture",
                                    command=lambda: changeType("Texture", "texture"))
        menubar.add_cascade(label="Type", menu=typeChoice)
        
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_cascade(label="Tutorial", command=tutorial)
        
        menubar.add_cascade(label="Help", menu=helpmenu)
        
            
        
        tk.Tk.config(self, menu=menubar)
        self.frames = {}
        for F in (StartPage, PageOne, TFD_Page): 
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)
        
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="""Tactile Feedback Display Application: """, font=LARGE_FONT_BOLD)
        label.pack(pady = 10, padx = 10)
        
        label1 = tk.Label(self, text= """ Used to detect defromation and texture 
        labels of a given object""", font=LARGE_FONT)
        label1.pack(padx = 10)
        
        
        # Button to switch pages
        button1 = ttk.Button(self, text = "Agree", 
                            command=lambda: controller.show_frame(TFD_Page) )
        button1.pack()
        button2 = ttk.Button(self, text = "Disagree", 
                            command=quit)
        button2.pack()
        
class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=" Page One!", font=LARGE_FONT)
        label.pack(pady = 10, padx = 10)
        
        # Button to switch pages
        button1 = ttk.Button(self, text="Back to Home", 
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

class TFD_Page(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph Page", font=LARGE_FONT)
        label.pack(pady = 10, padx = 10)
        
        # Button to switch pages
        button1 = ttk.Button(self, text="Back to Home", 
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()
       
        # Plot graph on grpah page
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add Navigation Bar to graph page
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
app = TactileFeedback()
app.geometry("1280x720")
ani = animation.FuncAnimation(f, animate, interval=20, cache_frame_data=False)

app.mainloop()
    
>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
        