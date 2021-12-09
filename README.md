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

      Fig. 1. A schematic view of Federated Learning approach

## *Descriptions of implementation :*
In this project, a Federated Deep Learning Simulator (FedSim) is introduced. FedSim provides a simple and flexible platform to implement federated learning algorithms and strategies, such as aggregation methods, communication methods, and compression methods. it is also provided to implement different strategies for local optimization algorithms (customized learning rate, batch size and so on).
Furthermore, FedSim can be used to investigate the performance of FL algorithms with different cases of data distributions such as:
-  IID : equal sample size and equal number of samples of all classes
-  non-IID ie : imbalanced number of samples and equal number of samples from all of classes.
-  non-IID ee : equal number of samples and equal number of samples from a subset of classes. 
-  non-IID ei : equal number of samples and imbalanced number of samples from a subset of classes.
-  non-IID ii : imbalanced number of samples and imbalanced number of samples from a subset of classes (fig. 4).

By default, in FedSim a Convolutional Neural Network (CNN) model is utilized for classification task. And also, the 'Fashion MNIST' dataset is used in (non-iid-ie) form (fig. 3). 
### *FedSim Components :*
in this section contents of the project will be described in detail. The following list represents modules of the project. 
 
 ### *Contents :* 
 1. Server_side.py 
 2. Client_side.py
 3. Data_Preprocessing.py
 4. Utils.py
 5. config.csv
 6. requirements.txt
 7. Main.py
 
 
 ![image](https://user-images.githubusercontent.com/92728743/145492314-0f2eecb3-7517-4169-8610-d9a202fda991.png)
 
     Fig. 2. A schematic view of FedSim components 
 
 
##### *The ‘Server_side’ module:*
The module “*Serveer_side*” contains four functions such as:  'config_()' initializes the configuration parameters. 'Create_model()' is utilized to create model. 'Aggregation()' receives and aggregates local models. And also, 'Evaluation()' is used to evaluate the global model after each round. 
##### *The ‘Client_side’ module:*
The module “*Client_side*” contains the 'Call_Client()' function which simulates the client-side processes such as: receiving global model and retrains it to create a local model.
##### *The ‘Data_Preprocessing’ module:*
The module “*Data_Preprocessing*” contains three functions such as:  'load_detaset()' loads dataset and splits to train and test parts. 'Scaling_data()' is utilized to scale uploaded dataset. 'Data_partitioning()' is used to prepare distributed dataset.
##### *The ‘Main’ module:*
The module “*Main*” involves the main loop of process.

## *How to execute FedSim*
Note: Before executing FedSim make sure that all requirements such as packages and libraries have been installed. To  do that, you can use the following instructions:

       $ pip install -r instruction.txt
In order to execute FedSim you can use the following instruction :

       $ python Main.py 
   
Note, in this case, FedSim will run with default settings. So, to customize implementation settings you should use the following form. 

       $ python Main.py -c 
 
Then, FedSim will set the parameters according to the 'config.csv' file. to change the implementation settings, you should customize the 'config.csv' or you should use your personal config file. the following instruction must be used:

   
       $ python Main.py -c -f yourfilename.csv
       
Table 1. represents the content of config.csv (note that if you remove parameters from config.csv their default values will be used.)       
![image](https://user-images.githubusercontent.com/92728743/144516344-6f156e50-c8e3-4202-ae66-f0f8031216a0.png)

 

 ![image](https://user-images.githubusercontent.com/92728743/141702184-354611e7-ba6e-408e-9174-ab7a41f967ba.png)


 
       fig. 3. illustrates (the non-iid-ie) dataset (horizontal axis = clients and vertical axis = number of samples)

![image](https://user-images.githubusercontent.com/92728743/144513113-e99c8c61-63c6-4a4e-8d52-c67cc3708f5b.png)


       fig. 4. illustrates (the non-iid-ii) dataset (horizontal axis = clients and vertical axis = the number of samples from each class)

### *References:*
[1] H. Brendan McMahan and Daniel Ramage. Federated learning: Collaborative machine learning without centralized training data. https://research.googleblog.com/2017/04/federated-learning-collaborative.html, 2017.
