# -*- coding: utf-8 -*-

'''
The 'Data_Preprocessing' module contains functions to work on data such as loading and 
scaling dataset and also to create a non-iid form of dataset.

      Copyright 2021 Ahmad Taheri. All Rights Reserved.
      
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np
import random
import tensorflow as tf

# #### import and prepare data ####
def load_detaset():
    num_of_classes = 10
    input_shape = (28, 28, 1)

    # laod dataset 
   #(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    return X_train, Y_train , X_test, Y_test

# ###  Scaling data ###
def Scaling_data(X_train, Y_train , X_test, Y_test):
   # (X_train, Y_train) = zip(train_data)
   # (X_test, Y_test) = zip(test_data)
    num_of_classes = np.asscalar(np.max(Y_train))+1
    X_train = X_train.astype("float32") / 255
    Y_train = tf.keras.utils.to_categorical(Y_train, num_of_classes)

    X_test = X_test.astype("float32") / 255
    Y_test =  tf.keras.utils.to_categorical(Y_test, num_of_classes)

    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    
    return X_train, Y_train , X_test, Y_test

# ###### preparing disributed data  ######
def Data_partitioning(data, num_of_clients,client_names,max_sample_size):
 
    # shuffling the data (randomization)  
    random.shuffle(data)
    
    np.random.shuffle(client_names)

    # the minimum sample size is 0.2% of total samples
    min_sample_size = len(data)*2//1000 
    Sizes = [min_sample_size for i in range(0,num_of_clients)] 

    R_samps = len(data) - (min_sample_size * num_of_clients )
    #z = np.random.randint(0,num_of_clients)
    z = 0
    while R_samps > (len(data)*0.001):
       clientID = z # np.random.randint(z,num_of_clients)
       samp_size = int(np.round( R_samps * max_sample_size * np.random.rand()))
       Sizes[clientID] += samp_size     
       R_samps = R_samps - samp_size
       z += 1
       if z == 100 :
         z = 0

    #
    s = 0    
    Dist_Data = [ 0 for i in range(num_of_clients)]
    for i in range(0,num_of_clients):
     Dist_Data[i] = data[s:s + Sizes[i]]
     s += Sizes[i]
   
    '''
    size = len(data)//num_of_clients
    Dist_Data = [data[i:i + size] for i in range(0, size*num_of_clients, size)]
    '''
    
    Client_Data = {client_names[i] : Dist_Data[i] for i in range(num_of_clients)} 
    return Client_Data

