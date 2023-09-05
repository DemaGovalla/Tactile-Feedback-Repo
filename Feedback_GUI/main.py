"""
Author of Code: Enrique Boswell Nueve IV.
Email: enriquenueve9@gmail.com
Written: June 2019

Description: Python-3.7 implementation of the paper "A General Reflex Fuzzy Min-Max Neural Network."
The General Reflex Fuzzy Min-Max Neural Network is a supervised clustering algorithim.

I DO NOT CLAIM TO BE THE CREATOR OF THIS ALGORITHIM! I have no affilation
with the creation of the paper this code is based on.
For information about the creators of this algorithim and the paper that it is based on,
please refer to the citation below.

@article{Nandedkar2007AGR,
  title={A General Reflex Fuzzy Min-Max Neural Network},
  author={Abhijeet V. Nandedkar and Prabir Kumar Biswas},
  journal={Engineering Letters},
  year={2007},
  volume={14},
  pages={195-205}
}
"""

# --- Import Modules --- #
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from GRMMFN import ReflexFuzzyNeuroNetwork
from Testgui import VTD_GUI


def main():
    # --- Import Iris data and split --- #
    data = pd.read_csv('iris_data_norm.csv')
    data = data.iloc[:,1:]
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    scaler_min_max = MinMaxScaler(feature_range=(0.01, .99))
    X = scaler_min_max.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    y_train, y_test = y_train.values, y_test.values
    X_train, X_test = X_train.T, X_test.T

    # --- Declare network --- #
    nn = ReflexFuzzyNeuroNetwork(gamma=2, theta=.1)

    # --- Run network --- #
    nn.train(X_train, y_train)

    # --- Test Network --- #
    nn.test(X_test,y_test)


main()
