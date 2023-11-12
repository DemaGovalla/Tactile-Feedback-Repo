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
        self.root.geometry("800x600")

        self.start_button = ttk.Button(root, text="Start Data Generation", command=self.toggle_data_generation)
        self.start_button.pack()

        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack()

        self.data = []
        self.plotting = False

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
            new_data_point = random.randint(1, 100)
            self.data.append(new_data_point)

            # Update the plot
            self.ax.clear()
            self.ax.plot(self.data, label='Live Data')
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Value')
            self.ax.set_title('Live Data Plot')
            self.ax.legend()
            self.canvas.draw()

            time.sleep(0.1)  # Simulate data generation interval

if __name__ == '__main__':
    root = tk.Tk()
    live_plotter = LivePlotter(root)
    root.mainloop()
