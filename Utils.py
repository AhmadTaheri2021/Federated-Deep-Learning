# -*- coding: utf-8 -*-
"""
 Federated Deep Learning. 

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

def Show_dist(partitioned_data):
    
    dist =[ np.shape(partitioned_data[Cname])[0] for Cname in partitioned_data]
    np.random.shuffle(dist)
    print('The total sample size is {}'.format( np.sum(dist)))
    #print(dist)
   # x = [i for i in range(0,len(dist))]
    x = np.linspace(1,len(dist),len(dist))
    plt.bar(x,dist)
    plt.show()

# #######################
def Save_Model(model):
    # Save model weights
    model.save_weights('./checkpoints/CNN_checkpoint')

# #######################
def Load_Model(model):
   # Restore model weights
    model.load_weights('./checkpoints/CNN_checkpoint')
    return model