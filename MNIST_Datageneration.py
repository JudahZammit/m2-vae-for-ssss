#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
import keras
import numpy as np
import random
import os
import pandas as pd 
from pandas import read_csv


# In[2]:


(train_x,train_y) , (val_x,val_y) = mnist.load_data()
new_train_x = np.zeros((60000,28*28))
new_val_x = np.zeros((10000,28*28))
labeled_train_x = np.zeros((100,28*28))
labeled_train_y = np.zeros((100,10))
new_val_y = np.zeros((10000,10))
new_train_y = np.zeros((60000,10))
classes = 10


# In[3]:


for x in range(len(train_x)):
    new_train_x[x] = (train_x[x].flatten() / 255) -.5
    new_train_y[x] = keras.utils.to_categorical(train_y[x],num_classes = classes)
for x in range(len(val_x)):
    new_val_x[x] = (val_x[x].flatten() /255) -.5
    new_val_y[x] = keras.utils.to_categorical(val_y[x],num_classes = classes)
train_x = new_train_x
train_y = new_train_y
val_x = new_val_x
val_y = new_val_y


# In[4]:


index = np.arange(0, len(train_x), 1).tolist()
random.shuffle(index)


# In[5]:


zero = 0
one = 0
two = 0
three = 0
four = 0
five = 0
six = 0
seven = 0
eight = 0
nine = 0


# In[6]:


for x in index:
    if(np.array_equal(train_y[x],[1,0,0,0,0,0,0,0,0,0])):
        labeled_train_y[zero] = [1,0,0,0,0,0,0,0,0,0]
        labeled_train_x[zero] = train_x[x]
        train_x = np.delete(train_x,x,0)
        train_y = np.delete(train_y,x,0)
        zero += 1
    if(zero == 10):
        break
for x in index:
    if(np.array_equal(train_y[x],[0,1,0,0,0,0,0,0,0,0])):
        labeled_train_y[1*10 + one] = [0,1,0,0,0,0,0,0,0,0]
        labeled_train_x[1*10 + one] = train_x[x]
        train_x = np.delete(train_x,x,0)
        train_y = np.delete(train_y,x,0)
        one += 1
    if(one == 10):
        break
for x in index:
    if(np.array_equal(train_y[x],[0,0,1,0,0,0,0,0,0,0])):
        labeled_train_y[2*10 + two] = [0,0,1,0,0,0,0,0,0,0]
        labeled_train_x[2*10 + two] = train_x[x]
        train_x = np.delete(train_x,x,0)
        train_y = np.delete(train_y,x,0)
        two += 1
    if(two == 10):
        break
for x in index:
    if(np.array_equal(train_y[x],[0,0,0,1,0,0,0,0,0,0])):
        labeled_train_y[3*10 + three] = [0,0,0,1,0,0,0,0,0,0]
        labeled_train_x[3*10 + three] = train_x[x]
        train_x = np.delete(train_x,x,0)
        train_y = np.delete(train_y,x,0)
        three += 1
    if(three == 10):
        break
for x in index:
    if(np.array_equal(train_y[x],[0,0,0,0,1,0,0,0,0,0])):
        labeled_train_y[4*10 + four] = [0,0,0,0,1,0,0,0,0,0]
        labeled_train_x[4*10 + four] = train_x[x]
        train_x = np.delete(train_x,x,0)
        train_y = np.delete(train_y,x,0)
        four += 1
    if(four == 10):
        break
for x in index:
    if(np.array_equal(train_y[x],[0,0,0,0,0,1,0,0,0,0])):
        labeled_train_y[5*10 + five] = [0,0,0,0,0,1,0,0,0,0]
        labeled_train_x[5*10 + five] = train_x[x]
        train_x = np.delete(train_x,x,0)
        train_y = np.delete(train_y,x,0)
        five += 1
    if(five == 10):
        break
for x in index:
    if(np.array_equal(train_y[x],[0,0,0,0,0,0,1,0,0,0])):
        labeled_train_y[6*10 + six] = [0,0,0,0,0,0,1,0,0,0]
        labeled_train_x[6*10 + six] = train_x[x]
        train_x = np.delete(train_x,x,0)
        train_y = np.delete(train_y,x,0)
        six += 1
    if(six == 10):
        break
for x in index:
    if(np.array_equal(train_y[x],[0,0,0,0,0,0,0,1,0,0])):
        labeled_train_y[7*10 + seven] =  [0,0,0,0,0,0,0,1,0,0]
        labeled_train_x[7*10 + seven] = train_x[x]
        train_x = np.delete(train_x,x,0)
        train_y = np.delete(train_y,x,0)
        seven += 1
    if(seven == 10):
        break
for x in index:
    if(np.array_equal(train_y[x],[0,0,0,0,0,0,0,0,1,0])):
        labeled_train_y[8*10 + eight] = [0,0,0,0,0,0,0,0,1,0]
        labeled_train_x[8*10 + eight] = train_x[x]
        train_x = np.delete(train_x,x,0)
        train_y = np.delete(train_y,x,0)
        eight += 1
    if(eight == 10):
        break
for x in index:
    if(np.array_equal(train_y[x],[0,0,0,0,0,0,0,0,0,1])):
        labeled_train_y[9*10 + nine] = [0,0,0,0,0,0,0,0,0,1]
        labeled_train_x[9*10 + nine] = train_x[x]
        train_x = np.delete(train_x,x,0)
        train_y = np.delete(train_y,x,0)
        nine += 1
    if(nine == 10):
        break


# In[7]:


train_x.shape


# In[8]:


pd.DataFrame(train_x).to_csv("./MNIST/59900_100_balenced/train_x.csv", index=None)
pd.DataFrame(train_y).to_csv("./MNIST/59900_100_balenced/train_y.csv", index=None)
pd.DataFrame(val_x).to_csv("./MNIST/59900_100_balenced/val_x.csv", index=None)
pd.DataFrame(val_y).to_csv("./MNIST/59900_100_balenced/val_y.csv", index=None)
pd.DataFrame(labeled_train_x).to_csv("./MNIST/59900_100_balenced/labeled_train_x.csv", index=None)
pd.DataFrame(labeled_train_y).to_csv("./MNIST/59900_100_balenced/labeled_train_y.csv", index=None)


# In[9]:


df = read_csv("./MNIST/59900_100_balenced/val_y.csv")


# In[10]:


np_x = df.to_numpy()


# In[11]:


np_x[0]

