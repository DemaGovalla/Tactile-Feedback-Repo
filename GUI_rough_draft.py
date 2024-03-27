<<<<<<< HEAD
"""
Module: GUI_rough_draft.py
Author: Dema N. Govalla
Date: December 11, 2023
Description: May need to delete this
"""

import numpy as np, pandas as pd, random, csv, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from matplotlib.animation import FuncAnimation
from torch import nn, optim
from TSC_LS import LearningShapelets
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading, time


def sample_ts_segments(X, shapelets_size, n_segments=10000):
    """
    Sample time series segments for k-Means.
    """
    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    segments = np.empty((n_segments, n_channels, shapelets_size))
    for i, k in enumerate(samples_i):
        s = random.randint(0, len_ts - shapelets_size)
        segments[i] = X[k, :, s:s+shapelets_size]
    return segments


def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=10000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters



class Tactile_Feedback_Plotter:
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




        series = pd.read_csv('combined_sensorData.csv')
        columns_to_combine = series.columns[:-1]
        combined_array = []


        # Loop through each column, extract values, and append to combined_array
        for column in columns_to_combine:
            new_array = series[column].to_numpy()
            combined_array.append(new_array)

        combined_array = np.column_stack(combined_array)

        y = series.iloc[:,-1].to_numpy() 

        label_map = {1: 0, 2: 1, 3: 2, 4: 3}
        y = np.array([label_map[label] for label in y])
        y = y[~np.isnan(y)]

        self.scaler = StandardScaler()
        X= self.scaler.fit_transform(combined_array)

        X = X.reshape(y.size,columns_to_combine.size,1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



        n_ts, n_channels, len_ts = X_train.shape
        loss_func = nn.CrossEntropyLoss()
        num_classes = len(set(y_train))
        # learn 2 shapelets of length 130
        shapelets_size_and_len = {1: 1}
        dist_measure = "euclidean"
        lr = 1e-2
        wd = 1e-3
        epsilon = 1e-7

        self.learning_shapelets = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len,
                                            in_channels=n_channels,
                                            num_classes=num_classes,
                                            loss_func=loss_func,
                                            to_cuda=False,
                                            verbose=1,
                                            dist_measure=dist_measure)

        for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len.items()):
            weights_block = get_weights_via_kmeans(X_train, shapelets_size, num_shapelets)
            self.learning_shapelets.set_shapelet_weights_of_block(i, weights_block)

        optimizer = optim.Adam(self.learning_shapelets.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
        self.learning_shapelets.set_optimizer(optimizer)

        self.losses = self.learning_shapelets.fit(X_train, y_train, epochs=2000, batch_size=256, shuffle=False, drop_last=False)

        self.output_file = 'random_data.csv'
        
        self.label = []
        self.x_label = []
        self.pred_x = 0

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
    def average_filter(self, column_values):
        average_value = np.mean(column_values)
        return average_value

    def median_filter(self, column_values):
        median_value = np.median(column_values)
        return median_value

    def generate_data(self):
        pred_x = 0
        while self.plotting:
          
            

            # global self.pred_x 
            with open(self.output_file, 'r') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)
                rows = list(csv_reader)

                second_column = np.array([row[2] for row in rows if len(row) > 2 and row[2]]).astype(float)  # Filtering out empty values
                third_column = np.array([row[3] for row in rows if len(row) > 3 and row[3]]).astype(float) # Filtering out empty values
                fourth_column = np.array([row[4] for row in rows if len(row) > 4 and row[4]]).astype(float)  # Filtering out empty values
                fifth_column = np.array([row[5] for row in rows if len(row) > 5 and row[5]]).astype(float)  # Filtering out empty values

                filtered_average_second_column = self.average_filter(second_column)
                filtered_average_third_column = self.average_filter(third_column)
                filtered_average_fourth_column = self.average_filter(fourth_column)
                filtered_average_fifth_column = self.average_filter(fifth_column)

                filtered_median_second_column = self.median_filter(second_column)
                filtered_median_third_column = self.median_filter(third_column)
                filtered_median_fourth_column = self.median_filter(fourth_column)
                filtered_median_fifth_column = self.median_filter(fifth_column)

            combined_filtered_values = np.array([
                filtered_average_second_column,
                filtered_average_third_column,
                filtered_average_fourth_column,
                filtered_average_fifth_column,
                filtered_median_second_column,
                filtered_median_third_column,
                filtered_median_fourth_column,
                filtered_median_fifth_column
            ])

            combined_filtered_values = combined_filtered_values.reshape(1, -1)
            
            X_test_scaled = self.scaler.transform(combined_filtered_values)
            X_test_scaled = X_test_scaled.reshape(1, combined_filtered_values.size, 1)

            prediction = self.learning_shapelets.predict(X_test_scaled) 

            prediction = prediction.ravel()
            prediction = prediction[0]
            print(prediction)

            self.label.append(prediction)
            self.x_label.append(pred_x)
        
            pred_x = pred_x + 1

            data_y = pd.Series(self.label)

            data_x = pd.Series(self.x_label)

            lower_limit = max(0, len(data_x) - 10) 

            plt.cla()
            self.ax.plot(data_x, data_y, label='Channel 1') 
            self.ax.set_ylabel("Value")   
            self.ax.set_xlabel("Time")    
            self.ax.set_title("Deformation Data")
            self.ax.set_xlim(lower_limit, len(data_x)) # Adjust x-axis limits dynamically
            self.ax.set_ylim(data_y.min() - 0.5, data_y.max() + 0.5)
            self.ax.set_yticklabels(['', 'Very Soft', 'Soft', 'Medium Soft', 'Hard', 'Very Hard', 'Very Very Hard'])  # Set y-axis labels
            # self.ax.set_legend(loc='upper left')
            # self.ax.set_tight_layout()
            self.canvas.draw()
            time.sleep(0.1)  # Simulate data generation interval


if __name__ == '__main__':
    root = tk.Tk()
    live_plotter = Tactile_Feedback_Plotter(root)
    root.mainloop()


    # def animate(self, i):
    #         global pred_x 
    #         with open(self.output_file, 'r') as file:
    #             csv_reader = csv.reader(file)
    #             rows = list(csv_reader)

    #             second_column = np.array([row[2] for row in rows if len(row) > 2 and row[2]]).astype(float)  # Filtering out empty values
    #             third_column = np.array([row[3] for row in rows if len(row) > 3 and row[3]]).astype(float) # Filtering out empty values
    #             fourth_column = np.array([row[4] for row in rows if len(row) > 4 and row[4]]).astype(float)  # Filtering out empty values
    #             fifth_column = np.array([row[5] for row in rows if len(row) > 5 and row[5]]).astype(float)  # Filtering out empty values

    #             filtered_average_second_column = self.average_filter(second_column)
    #             filtered_average_third_column = self.average_filter(third_column)
    #             filtered_average_fourth_column = self.average_filter(fourth_column)
    #             filtered_average_fifth_column = self.average_filter(fifth_column)

    #             filtered_median_second_column = self.median_filter(second_column)
    #             filtered_median_third_column = self.median_filter(third_column)
    #             filtered_median_fourth_column = self.median_filter(fourth_column)
    #             filtered_median_fifth_column = self.median_filter(fifth_column)

    #         combined_filtered_values = np.array([
    #             filtered_average_second_column,
    #             filtered_average_third_column,
    #             filtered_average_fourth_column,
    #             filtered_average_fifth_column,
    #             filtered_median_second_column,
    #             filtered_median_third_column,
    #             filtered_median_fourth_column,
    #             filtered_median_fifth_column
    #         ])

    #         combined_filtered_values = combined_filtered_values.reshape(1, -1)
            
    #         X_test_scaled = self.scaler.transform(combined_filtered_values)
    #         X_test_scaled = X_test_scaled.reshape(1, combined_filtered_values.size, 1)

    #         prediction = self.learning_shapelets.predict(X_test_scaled) 

    #         prediction = prediction.ravel()
    #         prediction = prediction[0]
    #         print(prediction)

    #         self.label.append(prediction)
    #         self.x_label.append(pred_x)
        
    #         pred_x = pred_x + 1

    #         data_y = pd.Series(self.label)

    #         data_x = pd.Series(self.x_label)

    #         lower_limit = max(0, len(data_x) - 10) 

    #         plt.cla()
    #         plt.plot(data_x, data_y, label='Channel 1') 
    #         plt.ylabel("Value")   
    #         plt.xlabel("Time")    
    #         plt.title("Deformation Data")
    #         plt.xlim(lower_limit, len(data_x)) # Adjust x-axis limits dynamically
    #         plt.ylim(data_y.min() - 0.5, data_y.max() + 0.5)
    #         plt.legend(loc='upper left')
    #         plt.tight_layout()
    # # fig, ax = plt.subplots()
    # # ani = FuncAnimation(fig, animate, interval=50)  # Update interval in milliseconds (1 second in this case)
=======
"""
Module: GUI_rough_draft.py
Author: Dema N. Govalla
Date: December 11, 2023
Description: May need to delete this
"""

import numpy as np, pandas as pd, random, csv, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from matplotlib.animation import FuncAnimation
from torch import nn, optim
from TSC_LS import LearningShapelets
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading, time


def sample_ts_segments(X, shapelets_size, n_segments=10000):
    """
    Sample time series segments for k-Means.
    """
    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    segments = np.empty((n_segments, n_channels, shapelets_size))
    for i, k in enumerate(samples_i):
        s = random.randint(0, len_ts - shapelets_size)
        segments[i] = X[k, :, s:s+shapelets_size]
    return segments


def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=10000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters



class Tactile_Feedback_Plotter:
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




        series = pd.read_csv('combined_sensorData.csv')
        columns_to_combine = series.columns[:-1]
        combined_array = []


        # Loop through each column, extract values, and append to combined_array
        for column in columns_to_combine:
            new_array = series[column].to_numpy()
            combined_array.append(new_array)

        combined_array = np.column_stack(combined_array)

        y = series.iloc[:,-1].to_numpy() 

        label_map = {1: 0, 2: 1, 3: 2, 4: 3}
        y = np.array([label_map[label] for label in y])
        y = y[~np.isnan(y)]

        self.scaler = StandardScaler()
        X= self.scaler.fit_transform(combined_array)

        X = X.reshape(y.size,columns_to_combine.size,1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



        n_ts, n_channels, len_ts = X_train.shape
        loss_func = nn.CrossEntropyLoss()
        num_classes = len(set(y_train))
        # learn 2 shapelets of length 130
        shapelets_size_and_len = {1: 1}
        dist_measure = "euclidean"
        lr = 1e-2
        wd = 1e-3
        epsilon = 1e-7

        self.learning_shapelets = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len,
                                            in_channels=n_channels,
                                            num_classes=num_classes,
                                            loss_func=loss_func,
                                            to_cuda=False,
                                            verbose=1,
                                            dist_measure=dist_measure)

        for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len.items()):
            weights_block = get_weights_via_kmeans(X_train, shapelets_size, num_shapelets)
            self.learning_shapelets.set_shapelet_weights_of_block(i, weights_block)

        optimizer = optim.Adam(self.learning_shapelets.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
        self.learning_shapelets.set_optimizer(optimizer)

        self.losses = self.learning_shapelets.fit(X_train, y_train, epochs=2000, batch_size=256, shuffle=False, drop_last=False)

        self.output_file = 'random_data.csv'
        
        self.label = []
        self.x_label = []
        self.pred_x = 0

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
    def average_filter(self, column_values):
        average_value = np.mean(column_values)
        return average_value

    def median_filter(self, column_values):
        median_value = np.median(column_values)
        return median_value

    def generate_data(self):
        pred_x = 0
        while self.plotting:
          
            

            # global self.pred_x 
            with open(self.output_file, 'r') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)
                rows = list(csv_reader)

                second_column = np.array([row[2] for row in rows if len(row) > 2 and row[2]]).astype(float)  # Filtering out empty values
                third_column = np.array([row[3] for row in rows if len(row) > 3 and row[3]]).astype(float) # Filtering out empty values
                fourth_column = np.array([row[4] for row in rows if len(row) > 4 and row[4]]).astype(float)  # Filtering out empty values
                fifth_column = np.array([row[5] for row in rows if len(row) > 5 and row[5]]).astype(float)  # Filtering out empty values

                filtered_average_second_column = self.average_filter(second_column)
                filtered_average_third_column = self.average_filter(third_column)
                filtered_average_fourth_column = self.average_filter(fourth_column)
                filtered_average_fifth_column = self.average_filter(fifth_column)

                filtered_median_second_column = self.median_filter(second_column)
                filtered_median_third_column = self.median_filter(third_column)
                filtered_median_fourth_column = self.median_filter(fourth_column)
                filtered_median_fifth_column = self.median_filter(fifth_column)

            combined_filtered_values = np.array([
                filtered_average_second_column,
                filtered_average_third_column,
                filtered_average_fourth_column,
                filtered_average_fifth_column,
                filtered_median_second_column,
                filtered_median_third_column,
                filtered_median_fourth_column,
                filtered_median_fifth_column
            ])

            combined_filtered_values = combined_filtered_values.reshape(1, -1)
            
            X_test_scaled = self.scaler.transform(combined_filtered_values)
            X_test_scaled = X_test_scaled.reshape(1, combined_filtered_values.size, 1)

            prediction = self.learning_shapelets.predict(X_test_scaled) 

            prediction = prediction.ravel()
            prediction = prediction[0]
            print(prediction)

            self.label.append(prediction)
            self.x_label.append(pred_x)
        
            pred_x = pred_x + 1

            data_y = pd.Series(self.label)

            data_x = pd.Series(self.x_label)

            lower_limit = max(0, len(data_x) - 10) 

            plt.cla()
            self.ax.plot(data_x, data_y, label='Channel 1') 
            self.ax.set_ylabel("Value")   
            self.ax.set_xlabel("Time")    
            self.ax.set_title("Deformation Data")
            self.ax.set_xlim(lower_limit, len(data_x)) # Adjust x-axis limits dynamically
            self.ax.set_ylim(data_y.min() - 0.5, data_y.max() + 0.5)
            self.ax.set_yticklabels(['', 'Very Soft', 'Soft', 'Medium Soft', 'Hard', 'Very Hard', 'Very Very Hard'])  # Set y-axis labels
            # self.ax.set_legend(loc='upper left')
            # self.ax.set_tight_layout()
            self.canvas.draw()
            time.sleep(0.1)  # Simulate data generation interval


if __name__ == '__main__':
    root = tk.Tk()
    live_plotter = Tactile_Feedback_Plotter(root)
    root.mainloop()


    # def animate(self, i):
    #         global pred_x 
    #         with open(self.output_file, 'r') as file:
    #             csv_reader = csv.reader(file)
    #             rows = list(csv_reader)

    #             second_column = np.array([row[2] for row in rows if len(row) > 2 and row[2]]).astype(float)  # Filtering out empty values
    #             third_column = np.array([row[3] for row in rows if len(row) > 3 and row[3]]).astype(float) # Filtering out empty values
    #             fourth_column = np.array([row[4] for row in rows if len(row) > 4 and row[4]]).astype(float)  # Filtering out empty values
    #             fifth_column = np.array([row[5] for row in rows if len(row) > 5 and row[5]]).astype(float)  # Filtering out empty values

    #             filtered_average_second_column = self.average_filter(second_column)
    #             filtered_average_third_column = self.average_filter(third_column)
    #             filtered_average_fourth_column = self.average_filter(fourth_column)
    #             filtered_average_fifth_column = self.average_filter(fifth_column)

    #             filtered_median_second_column = self.median_filter(second_column)
    #             filtered_median_third_column = self.median_filter(third_column)
    #             filtered_median_fourth_column = self.median_filter(fourth_column)
    #             filtered_median_fifth_column = self.median_filter(fifth_column)

    #         combined_filtered_values = np.array([
    #             filtered_average_second_column,
    #             filtered_average_third_column,
    #             filtered_average_fourth_column,
    #             filtered_average_fifth_column,
    #             filtered_median_second_column,
    #             filtered_median_third_column,
    #             filtered_median_fourth_column,
    #             filtered_median_fifth_column
    #         ])

    #         combined_filtered_values = combined_filtered_values.reshape(1, -1)
            
    #         X_test_scaled = self.scaler.transform(combined_filtered_values)
    #         X_test_scaled = X_test_scaled.reshape(1, combined_filtered_values.size, 1)

    #         prediction = self.learning_shapelets.predict(X_test_scaled) 

    #         prediction = prediction.ravel()
    #         prediction = prediction[0]
    #         print(prediction)

    #         self.label.append(prediction)
    #         self.x_label.append(pred_x)
        
    #         pred_x = pred_x + 1

    #         data_y = pd.Series(self.label)

    #         data_x = pd.Series(self.x_label)

    #         lower_limit = max(0, len(data_x) - 10) 

    #         plt.cla()
    #         plt.plot(data_x, data_y, label='Channel 1') 
    #         plt.ylabel("Value")   
    #         plt.xlabel("Time")    
    #         plt.title("Deformation Data")
    #         plt.xlim(lower_limit, len(data_x)) # Adjust x-axis limits dynamically
    #         plt.ylim(data_y.min() - 0.5, data_y.max() + 0.5)
    #         plt.legend(loc='upper left')
    #         plt.tight_layout()
    # # fig, ax = plt.subplots()
    # # ani = FuncAnimation(fig, animate, interval=50)  # Update interval in milliseconds (1 second in this case)
>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
    # # plt.show()