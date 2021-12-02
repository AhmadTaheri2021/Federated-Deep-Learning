# *Federated-Deep-Learning Simulator (FedSim)*
A Simulator for Federated Deep Learning. 

## *Introduction:*
Federated learning is an approach (paradigm) to train a central model (global model) based on distributed data [1]. In this approach, instead of collecting data from distributed resources (clients) and creating a central dataset, in an iterative process each client downloads the global model and retrains it according to the local dataset. Then local models will be sent back to the server to be aggregated to obtain the global model. A schematic view of Federated Learning approach is illustrated in (fig. 1). 

#### *Advantages:* 
 - Privacy preservation (the main aim of this approach).
 - A huge amount of distributed and valuable data (to deal with the challenge of data in ML).
 
#### *Challenges:*
 - The performance of federated trained models is usually worse than those trained in the centralized learning mode.
 - The communication cost (models may have million number of weights)
 - Heterogeneous and imbalanced data distribution on clients (non-IID dataset)

![image](https://user-images.githubusercontent.com/92728743/141955974-0b7e2165-3cfd-47db-aff0-0e53f12449c5.png)

      fig. 1. A schematic view of Federated Learning approach

## *Descriptions of implementation :*
In this project, a Federated Deep Learning Simulator (FedSim) is introduced. FedSim provides a simple and flexible platform to implement federated learning algorithms and strategies, such as aggregation methods, communication methods, and compression methods. it is also provided to implement different strategies for local optimization algorithms (customized learning rate, batch size and so on).
Furthermore, FedSim can be used to investigate the performance of FL algorithms with different cases of data distributions such as:
-  IID : equal sample size and equal number of samples of all classes
-  non-IID ie : imbalanced number of samples and equal number of samples from all of classes.
-  non-IID ee : equal number of samples and equal number of samples from a subset of classes. 
-  non-IID ei : equal number of samples and imbalanced number of samples from a subset of classes.
-  non-IID ii : imbalanced number of samples and imbalanced number of samples from a subset of classes (fig. 3).

By default in FedSim a Convolutional Neural Network (CNN) model is utilized for classification task. So that, the 'Fashion MNIST' dataset is used in (non-iid-ie) form (fig. 2).  in this section contents of the project will be described in detail. The following list represents modules of the project. 
 
 ### *Contents :* 
 1. Server_side.py 
 2. Client_side.py
 3. Data_Preprocessing.py
 4. Utils.py
 5. config.csv
 6. Main.py
 
 
##### *The ‘Server_side’ module:*
The module “*Serveer_side*” contains four functions such as:  'config_()' initializes the configuration parameters. 'Create_model()' is utilized to create model. 'Aggregation()' receives and aggregates local models. And also, 'Evaluation()' is used to evaluate the global model after each round. 
##### *The ‘Client_side’ module:*
The module “*Client_side*” contains the 'Call_Client()' function which simulates the client-side processes such as: receiving global model and retrains it to create a local model.
##### *The ‘Data_Preprocessing’ module:*
The module “*Data_Preprocessing*” contains three functions such as:  'load_detaset()' loads dataset and splits to train and test parts. 'Scaling_data()' is utilized to scale uploaded dataset. 'Data_partitioning()' is used to prepare distributed dataset.
##### *The ‘Main’ module:*
The module “*Main*” involves the main loop of process.

## *How to execute FedSim*
In order to execute FedSim you can use the following instruction :

       $ python Main.py 
   
Note, in this case, FedSim will run with default settings. So, to customize implementation settings you should use the following form. 

       $ python Main.py -c 
 
Then, FedSim will set the parameters according to the 'config.csv' file. to change the implementation settings, you should customize the 'config.csv' or you should use your personal config file. the following instruction must be used:

   
       $ python Main.py -c -f yourfilename.csv


 

 ![image](https://user-images.githubusercontent.com/92728743/141702184-354611e7-ba6e-408e-9174-ab7a41f967ba.png)


 
       fig. 2. illustrates (the non-iid-ie) dataset (horizontal axis = clients and vertical axis = number of samples)

![image](https://user-images.githubusercontent.com/92728743/144510612-b8e55051-f4d2-4cf3-8434-c43cb018c67c.png)

       fig. 3. illustrates (the non-iid-ii) dataset (horizontal axis = clients and vertical axis = the number of samples from each class)

### *References:*
[1] H. Brendan McMahan and Daniel Ramage. Federated learning: Collaborative machine learning without centralized training data. https://research.googleblog.com/2017/04/federated-learning-collaborative.html, 2017.
