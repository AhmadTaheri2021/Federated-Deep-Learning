# -*- coding: utf-8 -*-

'''
The following code is the server side implementation and consists of several functions such as
config_(), create_model(), Aggregation() and evaluation(). 

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
from sklearn.metrics import accuracy_score
import tensorflow as tf
import Client_side

# #### global settings configuration  #####
def config_(args,num_of_clients=100,
            Max_Round=10,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            bs=32,
            lr=0.01,
            dist_type=1,
            alpha=0.7,
            participation_rate=0.1):
   
    ''' 
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, 
                    momentum=0.9)     
    '''
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr) 
    #optimizer = tf.keras.optimizers.Adam() #(learning_rate=lr)
         
    # a list of client names
    client_names = ['client_{}'.format(id) for id in range(1,num_of_clients+1)]
    
    global_config = {'dataset_' : 'FMNIST', #'CFAR10'
                      'Model_type' : 'CNN',
                      'num_of_clients' : num_of_clients,
                      'Max_Round' :  Max_Round,
                      'loss' : loss ,
                      'metrics' :  metrics,
                      'batch_size' : bs ,
                      'learning_rate' : lr,
                      'optimizer' : optimizer,
                      'client_names' : client_names,
                      'dist_type' : dist_type,
                      'dist_alpha' : alpha,
                      'participation_rate' : participation_rate}


    
    if(args.customized):
       # print('searching {} ...'.format(args.file))    
        try:   
          df_config = pd.read_csv(args.file)
          params = df_config["param"]
          vals = df_config["val"]

          print('Loading settings from {}'.format(args.file))

          for i in range(len(df_config)):
             if(params[i] == 'metrics'):
               vals[i] = [vals[i]]
             global_config.update({params[i] : vals[i]})
     
          
        except:
           print('Error: file {} not found'.format(args.file))     
           print("Default settings will be used")    

     
    print('--- Configuration settings ----')
   # print(global_config['ds_name'])  
                   
    return global_config

# ################################################
# ################################################

def Pre_Comm(global_model):
   '''
     this function is launched before communication (before sending global model).
     'model' : global model. 
   '''
   
   cmp_mode = global_model
   return cmp_mode

# ################################################

def Post_Comm(local_model):
   '''
     this function is launched after communication (after receiving local model).
     'cmp_model' : global model. 
   '''
   
   mode = local_model
   return mode

# ################################################

def Communication(sub_clients,
                global_weights,
                data_,
                global_config):
    '''
     this function is used to communicate with clients.
     
    '''
    # list of local model weights 
    local_weights = dict()
   # ### call clients ###
    for client in sub_clients:   
         # Preprocessing before communication 
         g_model = Pre_Comm(global_weights)
   
         #send global model & receive local modeel
         rec_local_model = Client_side.Communication(client,
                           g_model,
                           data_[client],
                           global_config)
         #
   
         #after communication
         local_model = Post_Comm(rec_local_model)

         # keep local weights        
         local_weights.update({client: local_model})
         #free up memory after each round
         tf.keras.backend.clear_session()
        
   
    # Aggregation
    average_weights = Aggregation(local_weights,data_,sub_clients)
    
    return average_weights


# ##################################################################### 
# The following section is the structure of model used in this example. 
# Note that it is possible to modify the structure or use any other model.
#   #######
def creat_model(input_shape, classes):
    model = tf.keras.Sequential(
            [
               tf.keras.Input(shape=input_shape),
               tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
               tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
               tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
               tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
               tf.keras.layers.Flatten(),
               tf.keras.layers.Dropout(0.5),
               tf.keras.layers.Dense(classes, activation="softmax"),
            ]
        )
    return model
# ###############################################


# ####  ####
def Aggregation(weights,partitioned_data,sub_clients):
  
    client_names = sub_clients
   
    # calculate the total sample size across contributed clients
    # client_config = clients[client_name]
    sample_sizes = [ len(partitioned_data[client_name]) for client_name in sub_clients]
    total_sample_size = sum(sample_sizes)

    weight = weights[sub_clients[0]]
    aggregated_weight =  [np.zeros(np.shape(weight[i])) for i in range(len(weight))]   
    # 
    for client in sub_clients:
      weight = weights[client]
     # contrib_counter[client] +=1
      local_sample_size = len(partitioned_data[client])
      scaler = local_sample_size/total_sample_size 
 
      num_layer = len(weight)
      for i in range(num_layer):
          aggregated_weight[i] = aggregated_weight[i] + (scaler * weight[i])

    return aggregated_weight
#################################################

    # ####  #######
    #evaluate the global model
log_ = dict()    

def evaluate(round_,global_model, X_test,Y_test):    
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
   # X_test,Y_test = zip(*test_data)
    logits = global_model.predict(X_test)
    loss_ = cce(Y_test, logits)
    Accuracy_ = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('Round : {} | Accuracy: {} | loss: {:.3}'.format(round_, Accuracy_, loss_))
   
    result_ = [round_, Accuracy_,float(loss_)]
    log_.update({'Round_{}'.format(round_) : result_}) 
    return log_   

