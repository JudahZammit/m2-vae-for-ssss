#!/usr/bin/env python
# coding: utf-8

# In[1]:


#An M2 VAE for semi supervised modelling on MNIST


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

# Number of monte  carlo samples (see paper)
MC_SAMPLES = 1

BATCH_SIZE = 200


# In[4]:


def M2():
    #Shape of flattened image
    input_shape=(RESIZE*RESIZE,)
    #inputs
    img_input = Input(shape=input_shape)
    #labels with empty second half
    y_full = Input(shape=(CLASSES,))
    #labels with empty labels removed
    y_input,y_val = Lambda(lambda x:  tf.split(x,num_or_size_splits=2,axis=0))(y_full)
    
    def gaussian_sampling(args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        z_mean_repeat = RepeatVector(MC_SAMPLES)(z_mean)
        z_log_var_repeat = RepeatVector(MC_SAMPLES)(z_log_var)
        epsilon = K.random_normal(shape=K.shape(z_mean_repeat))
        z_sample = z_mean_repeat + K.exp(0.5 * z_log_var_repeat) * epsilon
        return z_sample
    
    
    #Implements a q(y|x) NN with two hidden units that outputs the probability of each img being a certain label
    q_y__x_layer1 = Dense(INTERMEDIATE_DIM)(img_input)
    q_y__x_layer1_act =LeakyReLU(alpha = .03)(q_y__x_layer1)
    q_y__x_layer2 = Dense(INTERMEDIATE_DIM)((q_y__x_layer1_act))
    q_y__x_layer2_act = LeakyReLU(alpha = .03)(q_y__x_layer2)
    q_y__x_output = Dense(CLASSES, activation = 'softmax',name = 'q_y__x')(q_y__x_layer2_act)
    
    # Seperates out the predictions that we have labels for and those that we do not
    y_sup,y_un= Lambda(lambda x:  tf.split(x,num_or_size_splits=2,axis=0))(q_y__x_output) 
    # For the integrating out approach, we repeat the input matrix x, and construct a target (bs * n_y) x n_y
    # Example of input and target matrix for a 3 class problem and batch_size=2. 2D tensors of the form
    #               x_repeat                     t_repeat
    #  [[x[0,0], x[0,1], ..., x[0,n_x]]         [[1, 0, 0]
    #   [x[1,0], x[1,1], ..., x[1,n_x]]          [1, 0, 0]
    #   [x[0,0], x[0,1], ..., x[0,n_x]]          [0, 1, 0]
    #   [x[1,0], x[1,1], ..., x[1,n_x]]          [0, 1, 0]
    #   [x[0,0], x[0,1], ..., x[0,n_x]]          [0, 0, 1]
    #   [x[1,0], x[1,1], ..., x[1,n_x]]]         [0, 0, 1]]
    one_hot = Lambda( lambda x: K.constant(np.eye(CLASSES, dtype=int)))(img_input)
    #if garbage values change tile to repeat
    dummy_y = Lambda( lambda x: K.tile(x, [(BATCH_SIZE//2),1] ))(one_hot)
    
    y = Concatenate(axis=0)([y_input,dummy_y])
    
    # turn x,y,z into x,x,x,y,y,y,z,z,z with the number of repeats being the number of classes
    img_sup,img_un =Lambda(lambda x:  tf.split(x,num_or_size_splits=2,axis=0))(img_input) 
    rep_img_un = Lambda(lambda x: K.repeat_elements(x,rep=CLASSES,axis = 0))(img_un)
    rep_img_input = Concatenate(axis=0)([img_sup,rep_img_un])
    
    #Implements a q(z|y,x) NN with two hidden units that outputs the parameters to a gaussian distribution
    # for labeled data and for unlabeled outputs parameters for each possible y
    q_z__y_x_concat = Concatenate()([rep_img_input,y])
    q_z__y_x_layer1 = Dense(INTERMEDIATE_DIM)(q_z__y_x_concat)
    q_z__y_x_layer1_act = LeakyReLU(alpha = .03)(q_z__y_x_layer1)
    q_z__y_x_layer2 = Dense(INTERMEDIATE_DIM)(q_z__y_x_layer1_act)
    q_z__y_x_layer2_act = LeakyReLU(alpha = .03)(q_z__y_x_layer2)
    q_z__y_x_mean = Dense(LATENT_DIM,name = 'q_z__y_x_mean')(q_z__y_x_layer2_act)
    rep_q_z__y_x_mean = RepeatVector(MC_SAMPLES)(q_z__y_x_mean)
    q_z__y_x_log_var = Dense(LATENT_DIM,name = 'q_z__y_x_log_var')(q_z__y_x_layer2_act)
    rep_q_z__y_x_log_var = RepeatVector(MC_SAMPLES)(q_z__y_x_log_var)
    q_z__y_x_output = Lambda(gaussian_sampling,name = 'q_z__y_x')([q_z__y_x_mean,q_z__y_x_log_var])

    # Implements a p(x|y,z) NN with two hidden units that outputs the parameters to a bernoulli distribution
    # for labeled data and for unlabeled outputs parameters for each possible y
    p_x__y_z_concat = Concatenate()([y ,Flatten()(q_z__y_x_output)])
    p_x__y_z_layer1 = Dense(INTERMEDIATE_DIM)(p_x__y_z_concat)
    p_x__y_z_layer1_act = LeakyReLU(alpha = .03)(p_x__y_z_layer1)
    p_x__y_z_layer2 = Dense(INTERMEDIATE_DIM)(p_x__y_z_layer1_act)
    p_x__y_z_layer2_act = LeakyReLU(alpha = .03)(p_x__y_z_layer2)
    p_x__y_z_mean = Dense(RESIZE*RESIZE,activation = 'sigmoid',name = 'p_x__y_z_mean')(p_x__y_z_layer2_act)
    #p_x__y_z_log_var = Dense(resize*resize,name = 'p_x__a_y_z_log_var')(p_x__a_y_z_layer2_act)
    #p_x__y_z_output = Lambda(gaussian_sampling,name = 'p_x__a_y_z')([p_x__a_y_z_mean, p_x__a_y_z_log_var]) 
    
    def gaussian_ll(args):
        # Calculates the log liklihood of a point x under a gaussian distribution parameterized by mu and log_var
        x , mu, log_var = args
        
        c = -.5 * math.log(2*math.pi)
        density = c - log_var/2 - ((x - mu)/(2*K.exp(log_var) + 1e-8))*(x - mu)

        return K.sum(density,axis = -1)
    
    def unit_gaussian_ll(args):
        # Calculates the log liklihood of a point x under a unit gaussian distribution
        x = args
        
        c = -.5 * math.log(2*math.pi)
        density = c - (x)**2/2

        return K.sum(density,axis = -1)

        
    def log_pz(y_true,y_pred):
        # Calculates the log liklihood that the sampled 'z' is under the unit gaussian distributions 
        #, then weights the unsupervised samples according to how likely their asscociated y value was.
        # (as predicted by p(y|x))
        flat_y_un = K.reshape(y_un,shape = [-1])
        ones = K.ones(shape = (BATCH_SIZE//2))
        weights = K.concatenate([ones,flat_y_un],0)
        loss_per_point = weights*K.mean(unit_gaussian_ll(q_z__y_x_output),axis = 1)
        split = tf.split(loss_per_point, num_or_size_splits=CLASSES+1 ,axis=0)
        sup_loss = split[0]
        un = K.concatenate(split[1:])
        un_loss = K.sum(K.reshape(un,[BATCH_SIZE//2,CLASSES]),axis = 1)
        loss = K.concatenate([sup_loss,un_loss])
        return loss
        
        
    def log_qz(y_true,y_pred):
        # Calculates the log liklihood that the sampled 'z' is under the gaussian distributions predicted by
        # q(z|y,x), then weights the unsupervised sampled according to how likely their asscociated y value was
        #(as predicted by p(y|x))
        flat_y_un = K.reshape(y_un,shape = [-1])
        ones = K.ones(shape = (BATCH_SIZE//2))
        weights = K.concatenate([ones,flat_y_un],0)
        loss_per_point = weights*K.mean(gaussian_ll([q_z__y_x_output,rep_q_z__y_x_mean,rep_q_z__y_x_log_var]),axis = 1)
        split = tf.split(loss_per_point, num_or_size_splits=CLASSES+1 ,axis=0)
        sup_loss = split[0]
        un = K.concatenate(split[1:])
        un_loss = K.sum(K.reshape(un,[BATCH_SIZE//2,CLASSES]),axis = 1)
        loss = K.concatenate([sup_loss,un_loss])

        return loss
    
    def log_py(y_true,y_pred):
        # Calculates the log liklihood that the all possible 'y' is under y's true distribution WHICH
        # IS ASSUMED TO BE BALANCED CATAGORICLE, then weights the unsupervised sampled according to 
        #how likely their asscociated y value was (as predicted by p(y|x)).
        flat_y_un = K.reshape(y_un,shape = [-1])
        ones = K.ones(shape = (BATCH_SIZE//2))
        weights = K.concatenate([ones,flat_y_un],0)
        expected = K.ones_like(q_y__x_output)/CLASSES
        concat = K.concatenate([y_input,y_un])
        loss_per_point = K.categorical_crossentropy(expected,q_y__x_output)
        return -loss_per_point
        
    
    def log_px(y_true,y_pred):
        # Calculates the log liklihood that the true images is under the gaussian distributions predicted by
        # p(x|a,y,z), then weights the unsupervised sampled according to how likely their asscociated y value was
        #(as predicted by p(y|a,x))
        flat_y_un = K.reshape(y_un,shape = [-1])
        ones = K.ones(shape = ((BATCH_SIZE//2)))
        weights = K.concatenate([ones,flat_y_un],0)
        loss_per_point = -weights*keras.losses.binary_crossentropy(rep_img_input,p_x__y_z_mean)
        split = tf.split(loss_per_point, num_or_size_splits=CLASSES+1 ,axis=0)
        sup_loss = split[0]
        un = K.concatenate(split[1:])
        un_loss = K.sum(K.reshape(un,[BATCH_SIZE//2,CLASSES]),axis = 1)
        loss = K.concatenate([sup_loss,un_loss])
        return loss
    
    def y_ent(y_true,y_pred):
        # Caluclates the entropy of the unsupervised predicted y values
        
        flat_y_un = K.reshape(y_un,shape = [-1])
        zero = K.zeros(shape = ((BATCH_SIZE//2)))
        un = flat_y_un*K.log(flat_y_un)
        un_loss = K.sum(K.reshape(un,[BATCH_SIZE//2,CLASSES]),axis = 1)

        loss = K.concatenate([zero,un_loss])
        return -loss
       
    def acc(y_true,y_pred):
        # Calculates the raw accuracy of our y prediction for the images that we have labels for

        return K.mean(keras.metrics.categorical_accuracy(y_input,y_sup))
    
    def y_class(y_true,y_pred):
        # Calculates a supervised loss for the y predictions for the images that we have labels for
        zero = K.zeros(shape = (BATCH_SIZE//2))
        sup_loss = K.categorical_crossentropy(y_input,y_sup)

        loss = K.concatenate([sup_loss,zero])
        return loss
    
    
    def qy_loss(y_true,y_pred):
        return K.mean(1*y_ent(y_true,y_pred) + -1*log_py(y_true,y_pred) + 10*y_class(y_true,y_pred))
    
    def qz_loss(y_true,y_pred):
        return K.mean(log_qz(y_true,y_pred) + -1*log_pz(y_true,y_pred))
    
    def px_loss(y_true,y_pred):
        return K.mean(-log_px(y_true,y_pred))
    
        
    losses = {'q_y__x': qy_loss,'q_z__y_x': qz_loss, 'p_x__y_z_mean':px_loss}
    
    model = Model([img_input,y_full],[p_x__y_z_mean,q_y__x_output,q_z__y_x_output]
                  , name = 'VAE')
    model.compile(loss = losses,metrics = {'q_y__x':acc},optimizer = keras.optimizers.Adam(lr=.001,clipnorm=1.,clipvalue= .5))
    
    return model  


# In[5]:


model = M2()
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
        
        self.batch_size = int(batch_size/2)
        self.X = np.zeros((self.batch_size*2, RESIZE*RESIZE), dtype='float32')
        self.Y = np.zeros((self.batch_size*2,CLASSES), dtype='float32')
        
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
            
        for x in self.unlabeled_index[i*self.batch_size:(i+1)*self.batch_size]:
            
            image = self.unlabeled_images[x] + .5
            rand = np.random.ranf(image.shape)
            image = np.greater(image,rand).astype(int)
            self.X[n] = image
            n = n + 1

        return [self.X , self.Y] , [self.Y,self.Y,self.Y]
        
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
        self.X = np.zeros((self.batch_size*2, RESIZE*RESIZE), dtype='float32')
        self.Y = np.zeros((self.batch_size*2,CLASSES), dtype='float32')
        
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
            
        return [self.X, self.Y], [self.Y,self.Y,self.Y]
        
    def on_epoch_end(self):
        pass


# In[8]:


# Create generators
train_gen = TrainGenerator(BATCH_SIZE)
val_gen = ValGenerator(int(BATCH_SIZE/2))


# In[9]:


# Saves the model best weights to a file 
checkpoint = ModelCheckpoint(
    'NEW_MNIST_M2_VAE.h5', 
    monitor='val_q_y__x_acc', 
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

