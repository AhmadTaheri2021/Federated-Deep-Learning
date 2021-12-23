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
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os


# #### import and prepare data ####
def load_dataset(global_config):
    dataset_ = global_config['dataset_']
    num_of_classes = 10
    input_shape = (28, 28, 1)
    print(' loading  {}  Dataset...'.format(dataset_))
    current_dir = os.getcwd()
    # laod dataset 
    if(dataset_ == 'MNIST'):
       (X_train, Y_train), (X_test, Y_test) = mnist.load_data(path=current_dir+'/dataset/mnist.npz')

       #(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
       num_of_classes = 10
       input_shape = (28, 28, 1)
    
    if(dataset_ == 'FMNIST'):
       (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()   
       num_of_classes = 10
       input_shape = (28, 28, 1)

    if(dataset_ == 'CFAR10'):
       (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()   
       num_of_classes = 10
       input_shape = (32, 32, 3)
    '''
    if(dataset_ == 'CUSTOM'):
       print('You are using personal dataset, so, it is essential to modify the function load_dataset() in Data_Preprocessing module.')     
       #load_data(path=current_dir+'/dataset/YourDatasetName.???')
    '''


    global_config.update({'num_of_classes' : num_of_classes})
    global_config.update({'input_shape' : input_shape})
    return X_train, Y_train , X_test, Y_test, global_config

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
def Data_partitioning(X_train,Y_train, global_config,show_dist=True):
  
    data = list(zip(X_train, Y_train))
    num_of_classes = int(np.asscalar(np.max(Y_train))+1)
    num_of_clients = int(global_config['num_of_clients'])
    client_names = global_config['client_names']
    dist_type = int(global_config['dist_type'])
    alpha = float(global_config['dist_alpha'])
    
    # shuffling the data (randomization)  
    #random.shuffle(data)
    np.random.shuffle(data)
    #
    samp_size = len(data)

    # ######################
    classes = dict()
    count = 0
    y_tr = Y_train   
    if(np.shape(y_tr)[1] > 1):  
      y_tr = np.argmax(Y_train,axis=1)

    for cl in range(num_of_classes):
        class_ = list()
        for i in range(len(y_tr)):
           if( y_tr[i] == cl ):
             class_.append(i)
             count +=1
        class_name = 'class_{}'.format(cl)     
        classes.update({class_name : class_})  

    #  
    np.random.shuffle(client_names)
    # --------------------------------------------
    # IID
    if(dist_type == 0):
       alpha = 0
       print('... IID Dataset...')
   
    # -------------------------------------------- 
    # non-IID-ie 
    if(dist_type == 0 or dist_type == 1):
      
      if(dist_type == 1):
        print('... non-IID-ie Dataset...')
      # the minimum sample size is 0.2% of total samples
      min_sample_size = int(len(data)*2//1000 )
      Sizes = [min_sample_size for i in range(0,num_of_clients)] 

      R_samps = len(data) - (min_sample_size * num_of_clients )
      #z = np.random.randint(0,num_of_clients)
     
     # z = 0
     # while R_samps > (len(data)*0.001):
     #    clientID = z # np.random.randint(z,num_of_clients)
     #    samp_size = int(np.round( R_samps * max_sample_size * np.random.rand()))
     #    Sizes[clientID] += samp_size     
     #    R_samps = R_samps - samp_size
     #    z += 1
     #    if z == 100 :
     #       z = 0

      d = tf.random.normal(shape=(1,num_of_clients),stddev=alpha)
      #print(d)
      dist = (tf.keras.activations.softmax(d) * R_samps +  min_sample_size)
     
      #print(dist)
      dist = np.reshape(dist,(num_of_clients,))
      #dist = dist +  min_sample_size

      #x = np.linspace(1,num_of_clients,num=num_of_clients)
      #plt.bar(x,dist)
      #plt.show()
    #
      s = 0    
      Dist_Data = [ 0 for i in range(num_of_clients)]
      for i in range(0,num_of_clients):
         Dist_Data[i] = data[s:s + int(dist[i])]
         s += int(dist[i])

      q_dist = dist
   # --------------------------------------------
    #non-IID-ii 
    if(dist_type == 2): #(c >= 0.1):
      print('... non-IID-ii Dataset...')
    #  c = 0.0001
      alpha = np.max([alpha,0.001])
      x = tf.random.normal(shape=(num_of_classes, num_of_clients),stddev=1)

      exp_ = np.zeros((num_of_classes,num_of_clients))
      dist = np.zeros((num_of_classes,num_of_clients))
      s = 0
      for i in range(num_of_classes):
        for j in range(num_of_clients):
           exp_[i,j] = np.math.exp(x[i,j])
    
      s = tf.reduce_sum(exp_)
 
      for i in range(num_of_classes):
        for j in range(num_of_clients):
           dist[i][j] = exp_[i][j] / s 

      for i in range(num_of_classes):
        for j in range(num_of_clients):
            exp_[i,j] = np.math.exp(dist[i,j]* (num_of_clients*(alpha*10)))

                   
      for i in range(num_of_classes ):
         s = tf.reduce_sum(exp_[i,:])
         class_name = 'class_{}'.format(i)
         temp_samp = classes[class_name] 
         rem_samp = len(temp_samp)
      #  print(s)
         for j in range(num_of_clients):            
            dist[i][j] =  int(np.round(exp_[i][j] / s  * len(temp_samp)))
            if(dist[i][j] > rem_samp):
               dist[i][j] = rem_samp
            rem_samp = rem_samp - dist[i][j]  

      Dist_Data = [ 0 for i in range(num_of_clients)]
      for j in range(num_of_clients):
         for i in range(num_of_classes):
           samp_clss = int(dist[i][j])
           class_name = 'class_{}'.format(i)
           temp_samp = classes[class_name] 
           Dist_Data[j] = [data[temp_samp.pop()] for z in range(samp_clss)]
           classes.update({class_name : temp_samp})
    # ----------------------------------------
      q_dist = [np.sum(dist[:,i]) for i in range(num_of_clients)]
    

    # non-IID-ei
    if( dist_type == 3):
       print('... non-IID-ei Dataset...')
       dist = np.zeros((num_of_classes,num_of_clients))
       clss = int(np.max([(num_of_classes * (1-alpha)) , 1]))
       print('classes : {}'.format(clss))
       Dist_Data = [ 0 for i in range(num_of_clients)]
       '''
       samp_clss = len(data) // num_of_clients // clss
       print('samp_clss : {}'.format(samp_clss))
       rem_samp = [ len(classes['class_{}'.format(i)])   for i in range(num_of_classes)]
       
       for i in range(num_of_clients):
          temp_sum = 0
          for j in range(num_of_classes): 
             if(rem_samp[j] >=  samp_clss and temp_sum < (samp_clss * clss)):
                dist[j][i] = samp_clss
                rem_samp[j] = rem_samp[j] - dist[j][i] 
                temp_sum = temp_sum + samp_clss 
                class_name = 'class_{}'.format(j)
                temp_samp = classes[class_name] 
                Dist_Data[i] = [data[temp_samp.pop()] for z in range(samp_clss)]
                classes.update({class_name : temp_samp})
       '''
# -------
       samp_clss = int((len(X_train)// num_of_clients // num_of_classes )) 
       rem_samp = [ len(classes['class_{}'.format(i)])   for i in range(num_of_classes)]
       clsses = [0 for k in range(clss)]
       client_clsses = dict()
       clients_id = [k for k in range(num_of_clients)]
       np.random.shuffle(clients_id)
       for i in clients_id:
          temp_sum = 0
          clss_ids = [k for k in range(num_of_classes)]
          clsses = [0 for k in range(clss)]
          for j in range(clss):
             r = np.argmax(rem_samp)  #np.random.randint(0,len(clss_ids))
             id = r#clss_ids.pop(r)
             clsses[j] = id
             if(rem_samp[id] >=  samp_clss and temp_sum < (samp_clss * clss)):
               dist[id][i] = samp_clss
               rem_samp[id] = rem_samp[id] - dist[id][i] 
               temp_sum = temp_sum + samp_clss
               class_name = 'class_{}'.format(id)
               temp_samp = classes[class_name] 
               Dist_Data[i] = [data[temp_samp.pop()] for z in range(samp_clss)]
               classes.update({class_name : temp_samp})
            
          client_clsses.update({client_names[i] : clsses })

       samp_clss = int(((len(X_train)// num_of_clients) - samp_clss * clss) / clss)# int((len(X_train)// num_of_clients // num_of_classes ) * (1 / clss))
       for i in range(num_of_clients):
          temp_sum = 0
          clsses = client_clsses[client_names[i]]
          for j in clsses:
             if(rem_samp[j] >=  samp_clss and temp_sum < (samp_clss * clss)):
                rem_samp[j] = rem_samp[j] - samp_clss 
                dist[j][i] += samp_clss
                temp_sum = temp_sum + samp_clss
                class_name = 'class_{}'.format(j)
                temp_samp = classes[class_name] 
                Dist_Data[i] = [data[temp_samp.pop()] for z in range(samp_clss)]
                classes.update({class_name : temp_samp}) 

       q_dist = [np.sum(dist[:,i]) for i in range(num_of_clients)]

    '''
    if(show_dist):
       print(np.min(q_dist))
       print(np.max(q_dist))
       x = np.linspace(1,num_of_clients,num=num_of_clients)
       plt.bar(x,q_dist)
       plt.show()
    '''   

    # 
    Client_Data = {client_names[i] : Dist_Data[i] for i in range(num_of_clients)} 
    return Client_Data
