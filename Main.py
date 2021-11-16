# -*- coding: utf-8 -*-
"""
In 'Main.py' the global loop is implemented. 

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
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
#-----------------------------------------
import Client_side
import Server_side
import Data_Preprocessing
import Utils
#------------------------------------------


#Implementation settings 
global_config = Server_side.config_()

# import and prepare data
# laod dataset 
X_train, Y_train , X_test, Y_test = Data_Preprocessing.load_detaset()

# Scaling data
X_train, Y_train , X_test, Y_test = Data_Preprocessing.Scaling_data(X_train, Y_train , X_test, Y_test)

train_data = list(zip(X_train, Y_train))
test_data = list(zip(X_test, Y_test))
#  
# a list of client names
client_names = global_config['client_names']

# data partitioning 
max_sample_size = 0.1 # the maximum sample size which can be held by a client is set to 10%

partitioned_data = Data_Preprocessing.Data_partitioning(train_data, 
                                      global_config['num_of_clients'],
                                      client_names,max_sample_size)

# illustrate Data Distribution
Utils.Show_dist(partitioned_data)

# ####################################################
# ### Global Model Initialization ###
global_model = Server_side.creat_model(global_config['input_shape'], 10)
global_config.update({'model' : global_model})

# #####################################################

# -----------------------------------------#
#             global loop                    
# -----------------------------------------#
for round in range(global_config['Max_Round']):
              
    # list of local model weights 
    local_weights = dict()
    
    # randomize clients sequence per round
    random.shuffle(client_names)
  
    # 
    participants_size = int(np.floor(global_config['num_of_clients'] * 0.1)) # percent 
    sub_clients = [ client_names[clientID]  for clientID in range(participants_size)]
    
    # 
    global_weights = global_model.get_weights()
    # ### call clients ###
    for client in sub_clients:   
        local_weight = Client_side.call_client(client,
                                               global_weights,
                                               partitioned_data[client],
                                               global_config)
        # keep local weights        
        local_weights.update({client: local_weight})
        #free up memory after each round
        tf.keras.backend.clear_session()
        
    # Aggregation
    average_weights = Server_side.Aggregation(local_weights,partitioned_data,sub_clients)
    #update global model 
    global_model.set_weights(average_weights)
  
    #evaluate the global model
    Server_side.evaluate(round,global_model, X_test,Y_test)
