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




series = pd.read_csv('combined_sensorData.csv')
columns_to_combine = series.columns[:-1]
combined_array = []

print(columns_to_combine)


dist_measure_strings = "euclidean" 
shapelets_len_values = 1  

# Loop through each column, extract values, and append to combined_array
for column in columns_to_combine:
    new_array = series[column].to_numpy()
    combined_array.append(new_array)
    
print(columns_to_combine[:4].size)
print(columns_to_combine[2:8].size)


split_combined_array_def = combined_array[:4]


# print("\n\n split_combined_array_def",split_combined_array_def)


split_combined_array_tex = combined_array[2:8]
# print("\n\n split_combined_array_tex",split_combined_array_tex)


array_def = np.column_stack(split_combined_array_def)

array_tex = np.column_stack(split_combined_array_tex)



# print("\n\n array_def",array_def)
# print("\n\n array_tex",array_tex)






y = series.iloc[:,-1].to_numpy() 
label_map = {1: 0, 2: 1, 3: 2, 4: 3}
y = np.array([label_map[label] for label in y])
y = y[~np.isnan(y)]


y_def = y
y_tex = y
print(y_def.size)
print(y_tex)



scaler_def = StandardScaler()
scaler_tex = StandardScaler()


X_def = scaler_def.fit_transform(array_def)
X_def = X_def.reshape(y.size,columns_to_combine[:4].size,1) # fix the 4
print(X_def.shape)

X_train_def, X_test_def, y_train_def, y_test_def = train_test_split(X_def, y, test_size=1)

print(X_train_def.size)
print(X_test_def.size)




X_tex = scaler_tex.fit_transform(array_tex)
X_tex = X_tex.reshape(y.size,columns_to_combine[2:8].size,1) # fix the 4
print(X_tex.shape)


X_train_tex, X_test_tex, y_train_tex, y_test_tex = train_test_split(X_tex, y, test_size=1)

print(X_train_tex.size)
print(X_test_tex.size)



# deformation
n_ts, n_channels, len_ts = X_train_def.shape
loss_func = nn.CrossEntropyLoss()
num_classes = len(set(y_train_def))
# learn 2 shapelets of length 130
# shapelets_size_and_len = {1: 1}
shapelets_size_and_len_def = {1:shapelets_len_values}

# dist_measure = "euclidean"
dist_measure_def = dist_measure_strings

lr = 1e-2
wd = 1e-3
epsilon = 1e-7

learning_shapelets_def = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len_def,
                                    in_channels=n_channels,
                                    num_classes=num_classes,
                                    loss_func=loss_func,
                                    to_cuda=False,
                                    verbose=1,
                                    dist_measure=dist_measure_def)

for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len_def.items()):
    weights_block = get_weights_via_kmeans(X_train_def, shapelets_size, num_shapelets)
    learning_shapelets_def.set_shapelet_weights_of_block(i, weights_block)

optimizer = optim.Adam(learning_shapelets_def.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
learning_shapelets_def.set_optimizer(optimizer)

losses = learning_shapelets_def.fit(X_train_def, y_train_def, epochs=2000, batch_size=256, shuffle=False, drop_last=False)


# Save the model to a file
joblib.dump(learning_shapelets_def, 'learning_shapelets_def.pkl')
print("Model is trained")







# texture
n_ts, n_channels, len_ts = X_train_tex.shape
loss_func = nn.CrossEntropyLoss()
num_classes = len(set(y_train_tex))
# learn 2 shapelets of length 130
# shapelets_size_and_len = {1: 1}
shapelets_size_and_len_tex = {1:shapelets_len_values}

# dist_measure = "euclidean"
dist_measure_tex = dist_measure_strings

lr = 1e-2
wd = 1e-3
epsilon = 1e-7

learning_shapelets_tex = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len_tex,
                                    in_channels=n_channels,
                                    num_classes=num_classes,
                                    loss_func=loss_func,
                                    to_cuda=False,
                                    verbose=1,
                                    dist_measure=dist_measure_tex)

for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len_tex.items()):
    weights_block = get_weights_via_kmeans(X_train_tex, shapelets_size, num_shapelets)
    learning_shapelets_tex.set_shapelet_weights_of_block(i, weights_block)

optimizer = optim.Adam(learning_shapelets_tex.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
learning_shapelets_tex.set_optimizer(optimizer)

losses = learning_shapelets_tex.fit(X_train_tex, y_train_tex, epochs=2000, batch_size=256, shuffle=False, drop_last=False)


# Save the model to a file
joblib.dump(learning_shapelets_tex, 'learning_shapelets_tex.pkl')
print("Model is trained")



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




series = pd.read_csv('combined_sensorData.csv')
columns_to_combine = series.columns[:-1]
combined_array = []

print(columns_to_combine)


dist_measure_strings = "euclidean" 
shapelets_len_values = 1  

# Loop through each column, extract values, and append to combined_array
for column in columns_to_combine:
    new_array = series[column].to_numpy()
    combined_array.append(new_array)
    
print(columns_to_combine[:4].size)
print(columns_to_combine[2:8].size)


split_combined_array_def = combined_array[:4]


# print("\n\n split_combined_array_def",split_combined_array_def)


split_combined_array_tex = combined_array[2:8]
# print("\n\n split_combined_array_tex",split_combined_array_tex)


array_def = np.column_stack(split_combined_array_def)

array_tex = np.column_stack(split_combined_array_tex)



# print("\n\n array_def",array_def)
# print("\n\n array_tex",array_tex)






y = series.iloc[:,-1].to_numpy() 
label_map = {1: 0, 2: 1, 3: 2, 4: 3}
y = np.array([label_map[label] for label in y])
y = y[~np.isnan(y)]


y_def = y
y_tex = y
print(y_def.size)
print(y_tex)



scaler_def = StandardScaler()
scaler_tex = StandardScaler()


X_def = scaler_def.fit_transform(array_def)
X_def = X_def.reshape(y.size,columns_to_combine[:4].size,1) # fix the 4
print(X_def.shape)

X_train_def, X_test_def, y_train_def, y_test_def = train_test_split(X_def, y, test_size=1)

print(X_train_def.size)
print(X_test_def.size)




X_tex = scaler_tex.fit_transform(array_tex)
X_tex = X_tex.reshape(y.size,columns_to_combine[2:8].size,1) # fix the 4
print(X_tex.shape)


X_train_tex, X_test_tex, y_train_tex, y_test_tex = train_test_split(X_tex, y, test_size=1)

print(X_train_tex.size)
print(X_test_tex.size)



# deformation
n_ts, n_channels, len_ts = X_train_def.shape
loss_func = nn.CrossEntropyLoss()
num_classes = len(set(y_train_def))
# learn 2 shapelets of length 130
# shapelets_size_and_len = {1: 1}
shapelets_size_and_len_def = {1:shapelets_len_values}

# dist_measure = "euclidean"
dist_measure_def = dist_measure_strings

lr = 1e-2
wd = 1e-3
epsilon = 1e-7

learning_shapelets_def = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len_def,
                                    in_channels=n_channels,
                                    num_classes=num_classes,
                                    loss_func=loss_func,
                                    to_cuda=False,
                                    verbose=1,
                                    dist_measure=dist_measure_def)

for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len_def.items()):
    weights_block = get_weights_via_kmeans(X_train_def, shapelets_size, num_shapelets)
    learning_shapelets_def.set_shapelet_weights_of_block(i, weights_block)

optimizer = optim.Adam(learning_shapelets_def.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
learning_shapelets_def.set_optimizer(optimizer)

losses = learning_shapelets_def.fit(X_train_def, y_train_def, epochs=2000, batch_size=256, shuffle=False, drop_last=False)


# Save the model to a file
joblib.dump(learning_shapelets_def, 'learning_shapelets_def.pkl')
print("Model is trained")







# texture
n_ts, n_channels, len_ts = X_train_tex.shape
loss_func = nn.CrossEntropyLoss()
num_classes = len(set(y_train_tex))
# learn 2 shapelets of length 130
# shapelets_size_and_len = {1: 1}
shapelets_size_and_len_tex = {1:shapelets_len_values}

# dist_measure = "euclidean"
dist_measure_tex = dist_measure_strings

lr = 1e-2
wd = 1e-3
epsilon = 1e-7

learning_shapelets_tex = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len_tex,
                                    in_channels=n_channels,
                                    num_classes=num_classes,
                                    loss_func=loss_func,
                                    to_cuda=False,
                                    verbose=1,
                                    dist_measure=dist_measure_tex)

for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len_tex.items()):
    weights_block = get_weights_via_kmeans(X_train_tex, shapelets_size, num_shapelets)
    learning_shapelets_tex.set_shapelet_weights_of_block(i, weights_block)

optimizer = optim.Adam(learning_shapelets_tex.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
learning_shapelets_tex.set_optimizer(optimizer)

losses = learning_shapelets_tex.fit(X_train_tex, y_train_tex, epochs=2000, batch_size=256, shuffle=False, drop_last=False)


# Save the model to a file
joblib.dump(learning_shapelets_tex, 'learning_shapelets_tex.pkl')
print("Model is trained")



>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
