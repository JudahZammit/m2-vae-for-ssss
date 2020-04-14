#!/usr/bin/env python
# coding: utf-8

# In[1]:


# A simple MLP for MNIST Classification


# In[2]:


import gc
import keras
from keras.layers import Lambda, Input, Dense, LeakyReLU, Concatenate,Dropout,RepeatVector,Reshape,Flatten
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy, mse
from keras.utils import plot_model
from keras.optimizers import Adam
from keras import backend as K
from keras.datasets import mnist
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize as norm
from collections import Counter
from keras import callbacks

import random
from pandas import read_csv
from sklearn.preprocessing import Binarizer

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#Shape of image: i.e. the image will be RESIZE X RESIZE
RESIZE = 28

# Number of latent z's
LATENT_DIM = 300

# Number of neurons in each hidden layer
INTERMEDIATE_DIM = 1000

# Number of classes
CLASSES = 10

BATCH_SIZE = 100


# In[4]:


#Shape of flattened image
input_shape=(RESIZE*RESIZE,)

img_input = Input(shape=input_shape)

#Implements a q(y|x) NN with two hidden units that outputs the probability of each img being a certain label
y = Dense(INTERMEDIATE_DIM)(img_input)
y =LeakyReLU(alpha = .03)(y)
y = Dense(INTERMEDIATE_DIM)(y)
y = LeakyReLU(alpha = .03)(y)
y = Dense(CLASSES, activation = 'softmax')(y)


# In[5]:


model = Model(img_input,y)
model.compile(loss = 'categorical_crossentropy',metrics = ['categorical_accuracy'],optimizer = keras.optimizers.Adam(lr=.001,clipnorm=1.,clipvalue= .5))
model.summary()


# In[6]:


# Generates the data used for training
class TrainGenerator(keras.utils.Sequence):
    
    # Loads in unlabeled images(file paths) and repeats the labeled images until they're
    # are more labeled ones then unlabeled ones
    def __init__(self,batch_size = 64):
        
        self.unlabeled_images = read_csv("./MNIST/59900_100_balenced/train_x.csv").to_numpy()
        self.labeled_images = read_csv("./MNIST/59900_100_balenced/labeled_train_x.csv").to_numpy()
        self.labels = read_csv("./MNIST/59900_100_balenced/labeled_train_y.csv").to_numpy()
        
        self.labeled_index = np.arange(0, len(self.labeled_images), 1).tolist()
        self.unlabeled_index = np.arange(0, len(self.unlabeled_images), 1).tolist()
        random.shuffle(self.labeled_index)
        random.shuffle(self.unlabeled_index)
        
        lis = np.arange(0, len(self.labeled_images), 1).tolist()
        while len(self.labeled_index) <= len(self.unlabeled_index):
            random.shuffle(lis)
            self.labeled_index.extend(lis)
        
        self.batch_size = batch_size
        self.X = np.zeros((self.batch_size, RESIZE*RESIZE), dtype='float32')
        self.Y = np.zeros((self.batch_size,CLASSES), dtype='float32')
        
    # Number of epochs is number of unlabeled images divided by the batch size
    def __len__(self):
        return  len(self.labeled_index) // self.batch_size 
        
    # Fetches batch treating image as a matrix of the parameters 
    # to independent bernoulli distributed random variables, which are
    # then sampled from to create a dynamic discretization of the data.
    # Also dummy encodes the label.
    def __getitem__(self, i):
        n = 0
        for x in self.labeled_index[i*self.batch_size:(i+1)*self.batch_size]:
            
            image = self.labeled_images[x] + .5
            label = self.labels[x]
            rand = np.random.ranf(image.shape)
            image = np.greater(image,rand).astype(int)

            self.X[n] = image
            self.Y[n] = label
            n = n + 1

        return self.X , self.Y
        
    def on_epoch_end(self):
        random.shuffle(self.unlabeled_index)
        
        self.labeled_index = np.arange(0, len(self.labeled_images), 1).tolist()
        random.shuffle(self.labeled_index)
        lis = np.arange(0, len(self.labeled_images), 1).tolist()
        while len(self.labeled_index) <= len(self.unlabeled_index):
            random.shuffle(lis)
            self.labeled_index.extend(lis)


# In[7]:


# Generates the data used for validation
class ValGenerator(keras.utils.Sequence):
    
    # Loads in the labeled images
    def __init__(self,batch_size = 64):
        
        self.labeled_images = read_csv("./MNIST/59900_100_balenced/val_x.csv").to_numpy()
        self.labels = read_csv("./MNIST/59900_100_balenced/val_y.csv").to_numpy()
        
        self.labeled_index = np.arange(0, len(self.labeled_images), 1).tolist()
        
        self.batch_size = batch_size
        self.X = np.zeros((self.batch_size, RESIZE*RESIZE), dtype='float32')
        self.Y = np.zeros((self.batch_size,CLASSES), dtype='float32')
        
    def __len__(self):
        return  len(self.labeled_index) // self.batch_size
        
    def __getitem__(self, i):
        n = 0
        for x in self.labeled_index[i*self.batch_size : (i+1)*self.batch_size]:
            
            image = self.labeled_images[x] + .5
            rand = np.random.ranf(image.shape)
            image = np.greater(image,rand).astype(int)
            label = self.labels[x]

            self.X[n] = image
            self.Y[n] = label
            n = n + 1
            
        return self.X, self.Y
        
    def on_epoch_end(self):
        pass


# In[8]:


# Create generators
train_gen = TrainGenerator(BATCH_SIZE)
val_gen = ValGenerator(BATCH_SIZE)


# In[9]:


# Saves the model best weights to a file 
checkpoint = ModelCheckpoint(
    'NEW_MNIST_Baseline.h5', 
    monitor='val_categorical_accuracy', 
    verbose=0, 
    save_best_only=True, 
    save_weights_only=False,
    mode='max',
    period = 1
)

# Reduces the learning rate when the model has stoped learning
reduce_lr = ReduceLROnPlateau(monitor='loss',patience = 3 ,factor = .5,verbose = 1)

# Trains the model for 10 epochs
history = model.fit_generator(
    generator = train_gen,
    validation_data=val_gen,
    callbacks=[checkpoint,reduce_lr],
    use_multiprocessing=False,
    workers=1,
    epochs=10 ,
    max_queue_size = 10
)

