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



# ###############################################
def Log_(info_):
    '''This function is used to save reports into a csv file.
       'info_' : a dictionary which contains parametters name and their values
    '''
    # open the file in the write mode
    f = open('./Log.csv', 'w')
    
    # create the csv writer
    writer = csv.writer(f)
    
    keys_ = list(info_.keys())
    vals_ = list(info_.values())
    for i in range(len(info)):
       row = [keys_[i],vals_[i]]
       # write a row to the csv file
       writer.writerow(row)

    # close the file
    f.close()
# ########################
def comm_cost_calc_(info_):
   '''
      this function calculates the communication cost.
      'info_' : a dictionary  
   '''
   results = dict()
   
   return results
# ########################

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
