# -*- coding: utf-8 -*-

# ------------------------------------------------------# 
# The following function acts as a client. So that, 
# the global model is downloaded from the server and 
# retrained based on its local data. finally, 
# client uploads the local model.
# ------------------------------------------------------# 
'''
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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf

# #### 
def call_client(client,global_weights,local_dataset,global_config):
       
       # create local model according to global model structure
        local_model = global_config['model']
        
        # ###### update local learning rate ####################
        
        # ######################################################
        local_model.compile(optimizer=global_config['optimizer'],
                            metrics=global_config['metrics'],
                             loss=global_config['loss'])
        
        #set the received global weights as the local weights
        local_model.set_weights(global_weights)
        
        #fit local model 
        bs = global_config['batch_size'] 
        x_tr,y_tr = zip(*local_dataset)
        dataset = tf.data.Dataset.from_tensor_slices((list(x_tr), list(y_tr))).batch(bs)
        local_model.fit(dataset, epochs=1, verbose=0)

        weight = local_model.get_weights()
             
        #save the last local models
        #Hist_local_weight_list.update({client : weight})
        
        local_weights = weight.copy() 

        return local_weights
