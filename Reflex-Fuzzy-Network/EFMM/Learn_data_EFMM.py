
# In[95]:
# Use to learn and test the data and run the algorithm EFMM


# --- Import Modules --- #
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from EFMM import FuzzyMinMaxNN
import csv



data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\Arduino_train_test_labels.csv')


np.random.shuffle(data.values)
data = data.sample(frac=1) #shuffle dataframe sample


df = data[['Force','Range','CoR','Variance','STD','Work']]




normalized_df=(df-df.min())/(df.max()-df.min())



#choose 50% training and 50% testing sample
X = normalized_df.values
y = data['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)

y_train, y_test = y_train.reshape((-1,1)),y_test.reshape((-1,1))


print(y_train.shape)
print(y_test.shape)
print("X train shape \n", X_train.shape)
print("X test shape \n", X_test.shape)



X_train, X_test = X_train.tolist(),X_test.tolist()
y_train, y_test = y_train.tolist(),y_test.tolist()



fuzzy = FuzzyMinMaxNN(1,theta=0.1)
fuzzy.train(X_train,y_train,1)




import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax,a,b,color):
    width = abs(a[0] - b[0])
    height = abs(a[1] - b[1])
    ax.add_patch(patches.Rectangle(a, width, height, fill=False,edgecolor=color))

"""
    plot dataset
"""
fig1 = plt.figure()
ax = fig1.add_subplot(111, aspect='equal',alpha=0.7)

        
"""
    plot Hyperboxes
"""
for i in range(len(fuzzy.V)):
    if fuzzy.hyperbox_class[i]==[1]:
        draw_box(ax,fuzzy.V[i][:2],fuzzy.W[i][:2],color='g')
    elif fuzzy.hyperbox_class[i]==[2]:
        draw_box(ax,fuzzy.V[i][:2],fuzzy.W[i][:2],color='b')
    else:
        draw_box(ax,fuzzy.V[i][:2],fuzzy.W[i][:2],color='r')
    
for i in range(len(X_train)):
    if y_train[i] == [1]:
        ax.scatter(X_train[i][0],X_train[i][1] , marker='o', c='g')
    elif y_train[i] == [2]:
        ax.scatter(X_train[i][0],X_train[i][1] , marker='o', c='b')
    else:
        ax.scatter(X_train[i][0],X_train[i][1] , marker='o', c='r')
    
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Hyperboxes created during training')
plt.xlim([0,1])
plt.ylim([0,1])

plt.show()

                
def get_class(x):
        mylist = []
        for i in range(len(fuzzy.V)):
            mylist.append([fuzzy.fuzzy_membership(x,fuzzy.V[i],fuzzy.W[i])])
        result = np.multiply(mylist,fuzzy.U)
        mylist=[]
        for i in range(fuzzy.clasess):
            mylist.append(max(result[:,i]))
            
        #print(mylist)
        #print(mylist.index(max(mylist))+1,max(mylist))
        #print('pattern belongs to class {} with fuzzy membership : {}'.format(mylist.index(max(mylist))+1,max(mylist)))
        return [mylist.index(max(mylist))+1]
      

def score(train,train_labels):
    counter=0
    wronge=0
    for i in range(len(train)):
        if get_class(train[i]) == train_labels[i] :
            counter+=1
        else:
            wronge+=1
            
    print('No of misclassification : {}'.format(wronge))
    return (counter/len(train_labels))*100


print('Accuracy (train) : {} %'.format(score(X_train,y_train)))
print('Accuracy (test) : {} %'.format(score(X_test,y_test)))


def predict(x):
    mylist = []


    for i in range(len(fuzzy.V)):
        mylist.append([fuzzy.fuzzy_membership(x,fuzzy.V[i],fuzzy.W[i])])
    # print(fuzzy.U)
    result = np.multiply(mylist,fuzzy.U)

    test = []
    for i in range(fuzzy.clasess):
        # print('pattern {} belongs to class {} with fuzzy membership value : {}'.format(x,i+1,max(result[:,i])))
        # print(max(result[:,i]))
        test.append((max(result[:,i])))
        # print(test)
    
    class_omega = np.argmax(test) + 1
    return class_omega


import random
import numpy as np
import pandas as pd
from itertools import count
from datetime import datetime
import time
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.style.use('fivethirtyeight')

label = []
x_label = []
pred_x = 0
def animate(i):
        ful = []
        global pred_x
        data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\Arduino_live.csv')

        # x = data['x_value']
        y1 = data['Force']
        y2 = data['Range']
        y3 = data['CoR']
        y4 = data['Variance']
        y5 = data['STD']
        y6 = data['Work']



        len1 = y1.size
        len2 = y2.size
        len3 = y3.size
        len4 = y4.size
        len5 = y5.size
        len6 = y6.size

        ful.append(y1[len1-1])
        ful.append(y2[len2-1])
        ful.append(y3[len3-1])
        ful.append(y4[len4-1])
        ful.append(y5[len5-1])
        ful.append(y6[len6-1])

        
        norm_ful = (ful-X.min())/(X.max()-X.min())
        # norm_ful = norm_ful.values
        
        prediction = predict(norm_ful)
        
        

        label.append(prediction)
        x_label.append(pred_x)
      

        pred_x = pred_x + 1

        data_y = pd.Series(label)

        data_x = pd.Series(x_label)



        plt.cla()
        plt.plot(data_x, data_y, label='Channel 1')
        plt.xlim(i-30, i+10)
        # plt.ylim(y[i]-5, y[i]+5)
        plt.legend(loc='upper left')
        plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, interval = 50)

plt.tight_layout()
plt.show()