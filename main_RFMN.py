"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData.csv file using the RFMN algorithm. 
			After traning and testing, it returns the algorithms metrics such as accuracy, presision and more.
			The file performs cross validation for different RFMN parameters and returns the classification
			report.  
"""


import numpy as np, pandas as pd, os, multiprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef
from RFMN import ReflexFuzzyNeuroNetwork
from time import time

def testAlgo(X_test, y_test, nn, send_end): 

	y_predlr = nn.test(X_test, y_test)
	print("done with predictions")
	accuracy_score1 = accuracy_score(y_test, y_predlr)
	rounded_accuracy = round(accuracy_score1, 5)
	print("This is accuracy{rounded_accuracy}")
	send_end.send(rounded_accuracy)

def main():
	print('Number of CPUs in the system: {}'.format(os.cpu_count()))
	sensor_data = pd.read_csv('2 column_combined_sensorData.csv')
	sensor_data = sensor_data.iloc[:,0:]

	X = sensor_data.iloc[:, :-1].values
	y = sensor_data.iloc[:, sensor_data.shape[1]-1].values

	scaler_min_max = MinMaxScaler(feature_range=(0.001, .99))
	X_norm = scaler_min_max.fit_transform(X)

	X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.33, random_state=42)
	X_train = X_train.T
	y_train = y_train.T

	nn = ReflexFuzzyNeuroNetwork(gamma=5, theta=.5)
	nn.train(X_train, y_train)
	print("Model is trained")

	num_parts = 94
	split_size = len(X_test) // num_parts
	x_split = []
	y_split = []
	for i in range(num_parts):
		start_index = i * split_size
		end_index = (i + 1) * split_size
		x_part = X_test[start_index:end_index]
		y_part = y_test[start_index:end_index]
		x_split.append(x_part)
		y_split.append(y_part)
	x_split = np.array(x_split)
	y_split = np.array(y_split)

	transposed_arrays = []
	for i in range(num_parts):
		transposed_arrays.append(x_split[i].T)
	transposed_arrays = np.array(transposed_arrays)
	
	startTime = time()

	jobs = []
	pipe_list = []
	for i in range(0,2):
		# range(0, os.cpu_count())
		recv_end, send_end = multiprocessing.Pipe(False)
		p = multiprocessing.Process(target=testAlgo, args=(transposed_arrays[i], y_split[i], nn, send_end))
		jobs.append(p)
		pipe_list.append(recv_end)
		p.start()

	for proc in jobs:
		proc.join()
	result_list = [x.recv() for x in pipe_list]
	print(result_list)
	print(len(result_list))

	final_sum = sum(result_list)
	print("Final sum: ", {final_sum})

	final_sum_average = (final_sum / len(result_list))*100
	print("Final sum average: ",{final_sum_average}, " %")

	end_time=time()
	totalTime = end_time - startTime
	print("Total time in seconds: ", totalTime, " sec")
	print("Total time in minutes: ", totalTime/60, " min")

if __name__ == '__main__':
	main()