
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

'''
Testing Data
Min Max scaling can only be performed on numeric data. e.g., float and int types are okay. no objects. 
'''
df = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\Computer Science\\ReflexFuzzyNetwork\\testdata.csv')

# print(df.head())
# print(df.tail())
# print(df.dtypes)

X = df.iloc[:, [2,4]] # I want all row ":", but only column number 2 and 4. 
y = df.iloc[:, 5] # I want all row ":", but only column number 5. 

# Split the data between train and test. 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 50)

# Perform minmax under the X_train. 
# print(X_train.head())
# print(X_train.tail())
print("This si X_train \n", X_train)

scaler = MinMaxScaler().fit(X_train) # Creating an object of class Minmax scalar and fiting in X_train
print("This is the fit \n", scaler)
print(scaler.data_min_)
print(scaler.data_max_)
print(X_train.describe()) # gives me all the important description for each colums. 
print(scaler.feature_range)
# valye will always lie between 0 and 1 after a transformation. This is for training data


# X_train_scale = scaler_X_train.transform(X_train)
# print("This is the transform \n", X_train_scale)

# scaler = MinMaxScaler

# This is for testing data. 


