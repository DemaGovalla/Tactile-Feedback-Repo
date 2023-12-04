# Starts here
import numpy as np, pandas as pd, random, math, sys, csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from torch import nn, optim
from learning_shapelets import LearningShapelets
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef
from sklearn.utils.multiclass import unique_labels



'''
Train the data
'''
series = pd.read_csv('combined_sensorData.csv')


columns_to_combine = series.columns[:-1]
print(columns_to_combine)
print(columns_to_combine.size)

combined_array = []


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

# np.set_printoptions(threshold=np.inf)
print(combined_array)
print(type(combined_array))
print(combined_array.shape)

y = y[~np.isnan(y)]

# print("old X", combined_array, '\n'*300)


X = combined_array

scaler = StandardScaler()
X= scaler.fit_transform(X)
# print(X)

X = X.reshape(y.size,columns_to_combine.size,1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



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



n_ts, n_channels, len_ts = X_train.shape
loss_func = nn.CrossEntropyLoss()
num_classes = len(set(y_train))
# learn 2 shapelets of length 130
shapelets_size_and_len = {1: 2}
dist_measure = "euclidean"
lr = 1e-2
wd = 1e-3
epsilon = 1e-7


learning_shapelets = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len,
                                       in_channels=n_channels,
                                       num_classes=num_classes,
                                       loss_func=loss_func,
                                       to_cuda=False,
                                       verbose=1,
                                       dist_measure=dist_measure)


for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len.items()):
    weights_block = get_weights_via_kmeans(X_train, shapelets_size, num_shapelets)
    learning_shapelets.set_shapelet_weights_of_block(i, weights_block)


optimizer = optim.Adam(learning_shapelets.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
learning_shapelets.set_optimizer(optimizer)


losses = learning_shapelets.fit(X_train, y_train, epochs=2000, batch_size=256, shuffle=False, drop_last=False)




def eval_accuracy(model, X, Y):
    result = []
    predictions = model.predict(X)
    if len(predictions.shape) == 2:
        predictions = predictions.argmax(axis=1)
        print(predictions)
    print(type(predictions))
    print(predictions.shape)

    result.append(predictions)
    print(f"Accuracy: {(predictions == Y).sum() / Y.size}")
    return result



y_predlr = np.array(eval_accuracy(learning_shapelets, X_test, y_test)).reshape(-1)




# check results
print("confusion_matrix \n", confusion_matrix(y_test, y_predlr), "\n")
print("classification_report \n", classification_report(y_test, y_predlr), "\n")


print("Model is trained")





import random
import numpy as np
import pandas as pd
import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from statistics import median

output_file = 'output_file.csv'

def average_filter(column_values):
    # if len(column_values) <= 60:
    #     raise ValueError("Input column should contain 50 values")
    average_value = np.mean(column_values)
    return average_value

def median_filter(column_values):
    # if len(column_values) <= 60:
    #     raise ValueError("Input column should contain 50 values")
    
    # Calculate the median of the values in the column
    median_value = np.median(column_values)
    return median_value

label = []
x_label = []
pred_x = 0

def animate(i):

        global pred_x 

        with open(output_file, 'r') as file:
            csv_reader = csv.reader(file)
            rows = list(csv_reader)

            second_column = np.array([row[2] for row in rows if len(row) > 2 and row[2]]).astype(float)  # Filtering out empty values
            third_column = np.array([row[3] for row in rows if len(row) > 3 and row[3]]).astype(float) # Filtering out empty values
            fourth_column = np.array([row[4] for row in rows if len(row) > 4 and row[4]]).astype(float)  # Filtering out empty values
            fifth_column = np.array([row[5] for row in rows if len(row) > 5 and row[5]]).astype(float)  # Filtering out empty values

            filtered_average_second_column = average_filter(second_column)
            filtered_average_third_column = average_filter(third_column)
            filtered_average_fourth_column = average_filter(fourth_column)
            filtered_average_fifth_column = average_filter(fifth_column)

            filtered_median_second_column = median_filter(second_column)
            filtered_median_third_column = median_filter(third_column)
            filtered_median_fourth_column = median_filter(fourth_column)
            filtered_median_fifth_column = median_filter(fifth_column)


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

        # print(combined_filtered_values)

        combined_filtered_values = combined_filtered_values.reshape(1, -1)
        
        # print(combined_filtered_values)


        X_test_scaled = scaler.transform(combined_filtered_values)

        # Reshape the 1D array into a 2D array with 8 rows and 1 column
        X_test_scaled = X_test_scaled.reshape(1, combined_filtered_values.size, 1)

        prediction = learning_shapelets.predict(X_test_scaled) 


        print(prediction)

        prediction = prediction.ravel()
        prediction = prediction[0]
        print(prediction)


        # prediction = int(prediction[0][1])

    
        # print(prediction)

        
        label.append(prediction)
        x_label.append(pred_x)
      
        pred_x = pred_x + 1

        data_y = pd.Series(label)

        data_x = pd.Series(x_label)

        lower_limit = max(0, len(data_x) - 10) 

        plt.cla()
        plt.plot(data_x, data_y, label='Channel 1') 
        plt.ylabel("Value")   
        plt.xlabel("Time")    
        plt.title("Deformation Data")

        plt.xlim(lower_limit, len(data_x)) # Adjust x-axis limits dynamically
        # plt.ylim(-1, 4) # Adjust y-axis limits for better visualization
        plt.ylim(data_y.min() - 0.5, data_y.max() + 0.5)

        # ax.set_ylim([-1, 4])                              # Set Y axis limit of plot
        # plt.legend(loc='upper left')

        plt.tight_layout()
                                          
fig, ax = plt.subplots()


# Start the animation
ani = FuncAnimation(fig, animate, interval=50)  # Update interval in milliseconds (1 second in this case)

# Show the plot
plt.show()





# ful = np.array([])
# global pred_x
# x = data['x_value']
# y1 = data['Force']
# y2 = data['X_axis']
# y3 = data['Y_axis']
# y4 = data['Z_axis']
# len1 = y1.size
# len2 = y2.size
# len3 = y3.size
# len4 = y3.size
# new_data = np.array([y1[len1-1], y2[len2-1], y3[len3-1], y4[len4-1]])
# ful = np.append(ful, new_data)
# ful.append(y1[len1-1])
# ful.append(y2[len2-1])
# ful.append(y3[len3-1])
# ful.append(y4[len4-1])
# ful1 = ful.reshape(1, 4)
# ful2 = (ful-combined_array_X.min())/(combined_array_X.max()-combined_array_X.min())
# ful2 = ful2.ravel()

# norm_ful = (ful-X.min())/(X.max()-X.min())
# norm_ful = norm_ful.values
# norm_ful = (ful-X_norm.min())/(X_norm.max()-X_norm.min())
# norm_ful = norm_ful.values
# prediction = nn.predict(norm_df)

# prediction = nn.predict(ful)
# print("Here", prediction)

