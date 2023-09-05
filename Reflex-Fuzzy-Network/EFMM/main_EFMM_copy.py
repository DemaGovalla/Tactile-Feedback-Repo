
# In[95]:


# --- Import Modules --- #
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from EFMM import FuzzyMinMaxNN


data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\irisdata.csv',                   
                            names=['PW','PL','SW','SL','Class'])



# data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\irisdata - 2 col.csv',                   
#                             names=['PW','PL','Class'])

# data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\mydata - 2 column - Copy.csv',                   
#                             names=['PW','PL','Class'])

print(data)



# In[96]:


# data['Class'] = data['Class'].replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [1,2,3])
data['Class'] = data['Class'].replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [1,2,3])


print(data['Class'])


# In[97]:


np.random.shuffle(data.values)
data = data.sample(frac=1) #shuffle dataframe sample

print(data)


# In[98]:


data.head()



# In[99]:


df = data[['PW','PL']]

print(df)


# In[100]:


print(df.head())


# In[101]:


import csv


normalized_df=(df-df.min())/(df.max()-df.min())
# print(df)
print(df.min())
print(df.max())
print(normalized_df)


# In[102]:


print(data['Class'].values)



# In[103]:


#choose 50% training and 50% testing sample
from sklearn.model_selection import train_test_split

X = normalized_df.values
y = data['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)

y_train, y_test = y_train.reshape((-1,1)),y_test.reshape((-1,1))

# print(y_train)
print(y_train.shape)

# print(y_test)
print(y_test.shape)

# print("This is X_train \n", X_train)
# print("This is y_train \n ", y_train)
print("X train shape \n", X_train.shape)
# print("Y train shape\n", y_train.shape)

# print("This is X_test \n", X_test)
# print("This is y_test \n ", y_test)
print("X test shape \n", X_test.shape)
# print("Y test shape\n", y_test.shape)

# train,test = normalized_df.values[:75,:4],normalized_df.values[75:,:4] 
# train_labels,test_labels = data['Class'].values[:75],data['Class'].values[75:]

# # print(train)

# # print("\n \n  new x_test", test)

# train_labels,test_labels = train_labels.reshape((-1,1)),test_labels.reshape((-1,1))

# print("\n \n \n Start here \n", train_labels)
# print(train_labels.shape)

# print(test_labels)
# print(test_labels.shape)


# In[104]:


print(X_train.shape, X_test.shape)


# In[105]:


# train_labels.shape,test_labels.shape

X_train, X_test = X_train.tolist(),X_test.tolist()
y_train, y_test = y_train.tolist(),y_test.tolist()



print(X_train)

print("\n \n \n", X_test)


# In[106]:


fuzzy = FuzzyMinMaxNN(1,theta=0.1)
fuzzy.train(X_train,y_train,1)


# In[107]:


len(fuzzy.V),len(fuzzy.W)

print("\n", fuzzy.V)
# print("\n", fuzzy.W)


# In[108]:


# ## In dimension 1 & 2

# In[107]:


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

                


# In[109]:


# ## In Dimension 3 & 4

# In[108]:


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# def draw_box(ax,a,b,color):
#     width = abs(a[0] - b[0])
#     height = abs(a[1] - b[1])
#     ax.add_patch(patches.Rectangle(a, width, height, fill=False,edgecolor=color))

# """
#     plot dataset
# """
# fig1 = plt.figure()
# ax = fig1.add_subplot(111, aspect='equal',alpha=0.7)

        
# """
#     plot Hyperboxes
# """
# for i in range(len(fuzzy.V)):
#     if fuzzy.hyperbox_class[i]==[1]:
#         draw_box(ax,fuzzy.V[i][2:],fuzzy.W[i][2:],color='g')
#     elif fuzzy.hyperbox_class[i]==[2]:
#         draw_box(ax,fuzzy.V[i][2:],fuzzy.W[i][2:],color='b')
#     else:
#         draw_box(ax,fuzzy.V[i][2:],fuzzy.W[i][2:],color='r')
    
# for i in range(len(X_train)):
#     if y_train[i] == [1]:
#         ax.scatter(X_train[i][2],X_train[i][3] , marker='o', c='g')
#     elif y_train[i] == [2]:
#         ax.scatter(X_train[i][2],X_train[i][3] , marker='o', c='b')
#     else:
#         ax.scatter(X_train[i][2],X_train[i][3] , marker='o', c='r')
    
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.title('Hyperboxes created during training')
# plt.xlim([0,1])
# plt.ylim([0,1])

# plt.show()


                


# In[110]:


# In[114]:


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
      


# In[111]:


# In[115]:


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


# In[112]:


# In[117]:


print('Accuracy (train) : {} %'.format(score(X_train,y_train)))


# In[113]:


print('Accuracy (test) : {} %'.format(score(X_test,y_test)))


# In[114]:


def predict(x):
    mylist = []

    # print(len(fuzzy.V))
    # print(len(fuzzy.W))
    # print(len(fuzzy.U))


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

# X = [[0.4,0.3],[0.6,0.25],[0.7,0.2]]

# for x in X:
#     print(predict(x))
#     print('='*80)

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
        data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\data.csv')

        # x = data['x_value']
        y1 = data['total_1']
        y2 = data['total_2']


        len1 = y1.size
        len2 = y2.size


        # yint1 = y1[len1-1]
        # yint2 = y2[len2-1]

        ful.append(y1[len1-1])
        ful.append(y2[len2-1])

        
        prediction = predict(ful)
        

        label.append(prediction)
        x_label.append(pred_x)
      

        pred_x = pred_x + 1

        data_y = pd.Series(label)

        data_x = pd.Series(x_label)

        # print(data_y)
        # print(type(data_y))
        # print(data_x)
        # print(type(data_x))



        plt.cla()
        plt.plot(data_x, data_y, label='Channel 1')

        plt.legend(loc='upper left')
        plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, interval = 50)

plt.tight_layout()
plt.show()