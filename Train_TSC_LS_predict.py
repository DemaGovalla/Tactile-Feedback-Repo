<<<<<<< HEAD
"""
Module: validate_classifier_TSC_LS.py
Author: Dema N. Govalla
Date: December 11, 2023
Description: I need the sensors to test this
"""

import joblib
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
from time import sleep, time

# np.set_printoptions(threshold=np.inf)




scaler_def = StandardScaler()
scaler_tex = StandardScaler()





loaded_model_def = joblib.load('learning_shapelets_def.pkl')
loaded_model_tex = joblib.load('learning_shapelets_tex.pkl')








def read_last_n_rows(file_path, n):
	with open(file_path, 'r') as file:
		csv_reader = csv.reader(file)
		header = next(csv_reader)
		rows = list(csv_reader)
		if len(rows) < n:
			rows = [rows[-1]] * (n - len(rows)) + rows
		last_n_rows = rows[-n:]
	return last_n_rows

def average_filter(column_values):
	average_value = np.mean(column_values)
	return average_value

def median_filter(column_values):
	median_value = np.median(column_values)
	return median_value












while True:
	
	last_n_rows = read_last_n_rows('arduino_live.csv', 20)

			
	Force = np.array([float(row[2]) for row in last_n_rows]).astype(float)
	Mag = np.array([float(row[3]) for row in last_n_rows]).astype(float)
	Accel = np.array([float(row[4]) for row in last_n_rows]).astype(float)
	Gyro = np.array([float(row[5]) for row in last_n_rows]).astype(float)

	Force_ave = average_filter(Force)
	Mag_ave = average_filter(Mag)
	Accel_ave = average_filter(Accel)
	Gyro_ave = average_filter(Gyro) 


	Force_med = median_filter(Force)
	Mag_med = median_filter(Mag)
	Accel_med = median_filter(Accel)
	Gyro_med = median_filter(Gyro) 


	# Combined array for defromation
	combined_filtered_values_def = np.array([
		Mag_ave,
		Mag_med,
		Force_ave,
		Force_med,
	])


	startTime1 = time()

	combined_filtered_values_def = combined_filtered_values_def.reshape(1, -1)
	# print("combined_filtered_values_def", combined_filtered_values_def)
	X_test_scaled_def = scaler_def.transform(combined_filtered_values_def)
	X_test_scaled_def = X_test_scaled_def.reshape(1, combined_filtered_values_def.size, 1)
 
 
	prediction_def = loaded_model_def.predict(X_test_scaled_def) 
	prediction_def = prediction_def.ravel()
	prediction_def = prediction_def[0]
 
 
	print("prediction_def", prediction_def)


	end_time1 = time()
	print("Total completion time is: ", end_time1 - startTime1)

	# Combined array for texture
	combined_filtered_values_tex = np.array([
		Force_ave,
		Force_med,
		Accel_ave,
		Accel_med,
		Gyro_ave,
		Gyro_med,

	])



	startTime2 = time()

	combined_filtered_values_tex = combined_filtered_values_tex.reshape(1, -1)
	# print("combined_filtered_values_tex", combined_filtered_values_tex)
	X_test_scaled_tex = scaler_tex.transform(combined_filtered_values_tex)
	X_test_scaled_tex = X_test_scaled_tex.reshape(1, combined_filtered_values_tex.size, 1)
 
 
	prediction_tex = loaded_model_tex.predict(X_test_scaled_tex) 
 
	prediction_tex = prediction_tex.ravel()
	prediction_tex = prediction_tex[0]
 
 
	print("prediction_tex", prediction_tex)
 
 
 

 
 
	# # print("X_test_scaled_tex", X_test_scaled_tex)
	# prediction_tex = nn_tex.predict(X_test_scaled_tex)
	# print("prediction_tex", prediction_tex)
	
	
	
	end_time2 = time()
	print("Total completion time is: ", end_time2 - startTime2)


=======
"""
Module: validate_classifier_TSC_LS.py
Author: Dema N. Govalla
Date: December 11, 2023
Description: I need the sensors to test this
"""

import joblib
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
from time import sleep, time

# np.set_printoptions(threshold=np.inf)




scaler_def = StandardScaler()
scaler_tex = StandardScaler()





loaded_model_def = joblib.load('learning_shapelets_def.pkl')
loaded_model_tex = joblib.load('learning_shapelets_tex.pkl')








def read_last_n_rows(file_path, n):
	with open(file_path, 'r') as file:
		csv_reader = csv.reader(file)
		header = next(csv_reader)
		rows = list(csv_reader)
		if len(rows) < n:
			rows = [rows[-1]] * (n - len(rows)) + rows
		last_n_rows = rows[-n:]
	return last_n_rows

def average_filter(column_values):
	average_value = np.mean(column_values)
	return average_value

def median_filter(column_values):
	median_value = np.median(column_values)
	return median_value












while True:
	
	last_n_rows = read_last_n_rows('arduino_live.csv', 20)

			
	Force = np.array([float(row[2]) for row in last_n_rows]).astype(float)
	Mag = np.array([float(row[3]) for row in last_n_rows]).astype(float)
	Accel = np.array([float(row[4]) for row in last_n_rows]).astype(float)
	Gyro = np.array([float(row[5]) for row in last_n_rows]).astype(float)

	Force_ave = average_filter(Force)
	Mag_ave = average_filter(Mag)
	Accel_ave = average_filter(Accel)
	Gyro_ave = average_filter(Gyro) 


	Force_med = median_filter(Force)
	Mag_med = median_filter(Mag)
	Accel_med = median_filter(Accel)
	Gyro_med = median_filter(Gyro) 


	# Combined array for defromation
	combined_filtered_values_def = np.array([
		Mag_ave,
		Mag_med,
		Force_ave,
		Force_med,
	])


	startTime1 = time()

	combined_filtered_values_def = combined_filtered_values_def.reshape(1, -1)
	# print("combined_filtered_values_def", combined_filtered_values_def)
	X_test_scaled_def = scaler_def.transform(combined_filtered_values_def)
	X_test_scaled_def = X_test_scaled_def.reshape(1, combined_filtered_values_def.size, 1)
 
 
	prediction_def = loaded_model_def.predict(X_test_scaled_def) 
	prediction_def = prediction_def.ravel()
	prediction_def = prediction_def[0]
 
 
	print("prediction_def", prediction_def)


	end_time1 = time()
	print("Total completion time is: ", end_time1 - startTime1)

	# Combined array for texture
	combined_filtered_values_tex = np.array([
		Force_ave,
		Force_med,
		Accel_ave,
		Accel_med,
		Gyro_ave,
		Gyro_med,

	])



	startTime2 = time()

	combined_filtered_values_tex = combined_filtered_values_tex.reshape(1, -1)
	# print("combined_filtered_values_tex", combined_filtered_values_tex)
	X_test_scaled_tex = scaler_tex.transform(combined_filtered_values_tex)
	X_test_scaled_tex = X_test_scaled_tex.reshape(1, combined_filtered_values_tex.size, 1)
 
 
	prediction_tex = loaded_model_tex.predict(X_test_scaled_tex) 
 
	prediction_tex = prediction_tex.ravel()
	prediction_tex = prediction_tex[0]
 
 
	print("prediction_tex", prediction_tex)
 
 
 

 
 
	# # print("X_test_scaled_tex", X_test_scaled_tex)
	# prediction_tex = nn_tex.predict(X_test_scaled_tex)
	# print("prediction_tex", prediction_tex)
	
	
	
	end_time2 = time()
	print("Total completion time is: ", end_time2 - startTime2)


>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
	sleep(1)