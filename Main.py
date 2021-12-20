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
import matplotlib.pyplot as plt
import tensorflow as tf
import keras 
import argparse 
#-----------------------------------------
import Client_side
import Server_side
import Data_Preprocessing
import Utils
#------------------------------------------ 

# ###########################################
# this section is to run 'Main.py' from command line and set implementation settings 
parser = argparse.ArgumentParser(
	description = "FedSim is a Federated Deep Learning simulator.",
	prog = "Main.py",
	epilog = "more details on https://github.com/AhmadTaheri2021/Federated-Deep-Learning"
)


parser.add_argument(
	"-c", "--customized",
	help = "to use customized configuration settings. then use [-f] to set the filname",
	action= 'store_true'

)

parser.add_argument(
	"-f", "--file",
	help = "to load configuration settings from a csv file, set the file name. for example '-f yourfilename.csv'",
	type = str,
        default = "config.csv"
)

# -----
args = parser.parse_args()
# ##########################################

#Implementation settings 
'''
global_config = Server_side.config_(args,
                                    num_of_clients=100,
                                    Max_Round=5,
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'],
                                    bs=32,
                                    lr=0.01,
                                    dist_type=1,
                                    alpha=0.7,
				    participation_rate=0.1)
'''
global_config = Server_side.config_(args)

# import and prepare data
# laod dataset 
X_train, Y_train , X_test, Y_test, global_config = Data_Preprocessing.load_detaset(global_config)

# Scaling data
X_train, Y_train , X_test, Y_test = Data_Preprocessing.Scaling_data(X_train, Y_train , X_test, Y_test)

train_data = list(zip(X_train, Y_train))
test_data = list(zip(X_test, Y_test))
#  
# a list of client names
client_names = global_config['client_names']

# data partitioning 
partitioned_data = Data_Preprocessing.Data_partitioning(X_train,Y_train, global_config)


# illustrate Data Distribution
#Utils.Show_dist(partitioned_data)

# ####################################################
# ### Global Model Initialization ###
global_model = Server_side.creat_model(global_config['input_shape'], global_config['num_of_classes'])
global_config.update({'model' : global_model})

# #####################################################

# -----------------------------------------#
#             global loop                    
# -----------------------------------------#
Max_Rounds = int(global_config['Max_Round'])
num_clients = int(global_config['num_of_clients'])
print('----------- FedSim is running... --------------')

for round in range(Max_Rounds):        
      
    # randomize clients sequence per round
    np.random.shuffle(client_names)
  
    # the percentage of participated clients 
    participation_rate = float(global_config['participation_rate']) # percent 
    participants_size = int(num_clients * participation_rate)
    sub_clients = [ client_names[clientID]  for clientID in range(participants_size)]
    
    # 
    global_weights = global_model.get_weights()
    # ### Server clients Communication ###
   
    average_weights = Server_side.Communication(sub_clients,
                                               global_weights,
                                               partitioned_data,
                                               global_config)
      
    #update global model 
    global_model.set_weights(average_weights)
  
    #evaluate the global model
    results_ = Server_side.evaluate(round,Max_Rounds,global_model, X_test,Y_test)

# write results in log_.csv
Utils.Log_(results_)
print('Reaults have been saved in log_.csv')