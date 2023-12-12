import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import random
import threading
import time

class LivePlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Data Plotter")
        # self.root.geometry("800x600")

        self.start_button = ttk.Button(root, text="Start Data Generation", command=self.toggle_data_generation)
        self.start_button.pack()

        # self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        # self.ax = self.figure.add_subplot(1,1,1)
        # self.figure.tight_layout(h_pad=2)

        self.figure, self.ax = plt.subplots(2, 1)
        self.figure.tight_layout(h_pad= 4)


        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack()

        self.data = []
        self.plotting = False
        self.hardness_levels = ['', 'Very Soft', 'Soft', 'Medium Soft', 'Hard', 'Very Hard', 'Very Very Hard', '']

    def toggle_data_generation(self):
        if self.plotting:
            self.plotting = False
            self.start_button.configure(text="Start Data Generation")
        else:
            self.plotting = True
            self.start_button.configure(text="Stop Data Generation")
            self.data = []  # Reset data

            # Start generating data and plotting
            self.data_thread = threading.Thread(target=self.generate_data)
            self.data_thread.start()

    def generate_data(self):
        while self.plotting:
            new_data_point = random.randint(1, 6)  # Generates values between 0 and 5
            self.data.append(new_data_point)

           # Update the plot
            self.ax[0].clear()
            self.ax[0].set_title('First Subplot')
            self.ax[0].plot(self.data, label='Live Data')
            self.ax[0].set_xlabel('Time')
            self.ax[0].set_ylabel('Hardness')  # Change y-axis label
            self.ax[0].set_title('Live Data Plot')
            self.ax[0].set_ylim(0, 6)  # Set y-axis limit
            self.ax[0].set_yticklabels(self.hardness_levels) # Set y-axis labels
            self.ax[0].legend()

            self.ax[1].clear()
            self.ax[1].set_title('Second Subplot')
            self.ax[1].plot(self.data, label='Live Data')
            self.ax[1].set_xlabel('Time')
            self.ax[1].set_ylabel('Hardness')  # Change y-axis label
            self.ax[1].set_title('Live Data Plot')
            self.ax[1].set_ylim(0, 6)  # Set y-axis limit
            self.ax[1].set_yticklabels(self.hardness_levels)   # Set y-axis labels
            self.ax[1].legend()
  

            self.canvas.draw()

            time.sleep(0.1)  # Simulate data generation interval

if __name__ == '__main__':
    root = tk.Tk()
    live_plotter = LivePlotter(root)
    root.mainloop()
