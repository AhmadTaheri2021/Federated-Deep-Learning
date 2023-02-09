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

To develop and investigate new strategies, algorithms and methods for federated learning, it is essential to use a federated learning simulator. In this project, a simple and flexible Federated Deep Learning Simulator (FedSim) is designed. FedSim provides a simple platform to implement federated learning algorithms and strategies, such as aggregation methods, communication methods, compression methods and privacy preserving strategies. It is also provided to implement different strategies for local optimization algorithms (customized learning rate, batch size and so on). Furthermore, FedSim can be used to investigate the performance of Federated Learning algorithms with different cases of data distributions.

The rest of the article is organized as follows: Section 2, explains the components of FedSim. The section 3 describes different data distributions provided in FedSim. In section 4, the usage of the simulator and its parameters would be explained.     
![image](https://user-images.githubusercontent.com/92728743/141955974-0b7e2165-3cfd-47db-aff0-0e53f12449c5.png)

      Fig. 1. A schematic view of Federated Learning approach

## *Section 2. The Components of FedSim :*
In this section, components and contents of FedSim will be described in detail. FedSim contains five modules: Server_side, Client_side, Data_Preprocessing, Utils and Main. The workflow diagram of FedSim components is illustrated in (Fig. 2). The following list represents contents of the project:
### *Contents :* 
 1. dataset (folder)
 2. Server_side.py 
 3. Client_side.py
 4. Data_Preprocessing.py
 5. Utils.py
 6. config.csv
 7. requirements.txt
 8. Main.py
 
 
 ![image](https://user-images.githubusercontent.com/92728743/145492314-0f2eecb3-7517-4169-8610-d9a202fda991.png)
 
     Fig. 2. illustrates the workflow diagram of FedSim components 
 

##### *The ‘dataset’ folder:*
If you want to use your dataset, you should put it in “*dataset*” folder and set the parameter value of dataset in config.csv to 'CUSTOM'. Note, in this case, it is essential to modify the function load_dataset() in Data_Preprocessing module.

##### *The ‘Server_side’ module:*
The module “*Serveer_side*” contains following functions:  'config_()' initializes the configuration parameters. 'Create_model()' is utilized to create model. function 'Communication()' simulates the process of sending global model to clients and receiving local models from clients. Function 'Aggregation()' receives and aggregates local models. Function 'Evaluation()' is used to evaluate the global model after each round. And also, functions 'Pre_comm()' and 'Post_comm()' would be launched before sending global model to a client and after receiving each local model, respectively.

##### *The ‘Client_side’ module:*
The module “*Client_side*” contains the following functions: function 'Communication()' simulates the process of sending local model to server and receiving global model from server. 'local_fit()' function retrains global model to create a local model. And also, functions 'Pre_comm()' and 'Post_comm()' would be launched before sending local model to server and after receiving global model, respectively.
##### *The ‘Data_Preprocessing’ module:*
The module “*Data_Preprocessing*” contains three functions such as:  'load_dataset()' loads dataset and splits to train and test parts. 'Scaling_data()' is utilized to scale uploaded dataset. 'Data_partitioning()' is used to prepare distributed dataset.
##### *The ‘Main’ module:*
The module “*Main*” involves the main loop of process.

## *Section 3. Data distributions:*
In order to provide a realistic simulation platform, FedSim provides miscellaneous form of data distributions such as independent and identically distributed (IID) and non-IID datasets such as:
-  IID : balanced sample size and balanced number of classes. All clients have same sample size from all     type of classes. 
-  non-IID ib : imbalanced sample size and balanced number of classes. Each client has different number of samples and balanced number of samples from all classes.
-  non-IID bi : balanced sample size and imbalanced number of samples from a subset of classes. In this case, each local dataset contains a subset of classes (For example, just 30%  of classes).
-  non-IID ii : imbalanced sample size and imbalanced number of classes. In this case, each client has different sample size and also, the number of classes are different and imbalanced. For example, a client may have 30% of all type of classes with different number of sample from each class. (fig. 4).



## *Section 4. The Usage of FedSim*
The flexibility and user-friendly are the main aims in this project. Thus, it is easy to implement new algorithms in FedSim and test their performance with different conditions. In this way, to enhance the flexibility, a set of parameters are considered in FedSim (Table. 1). In addition, to make FedSim more user-friendly, it is provided to import all parameters from a csv type file. So, users can easily run and test their algorithms according to different settings. By default, in FedSim a Convolutional Neural Network (CNN) model is utilized for classification task. And also, the 'MNIST' dataset is used in (non-iid-ib) form (fig. 3).

As mentioned before, FedSim utilizes a set of parameters that would be categorized in 3 groups, such as: (a) data partitioning and distribution parameters. (b) model and optimizer parameters. (c) infrastructure parameters. These parameters and their descriptions are explained in (Table. 1). 

Table. 1. represents FedSim parameters and their descriptions  

![image](https://user-images.githubusercontent.com/92728743/145910605-028774ab-0253-4a6f-ba62-bd55861e8e9a.png)


#### *How to execute FedSim*
Note: Before executing FedSim make sure that all requirements such as packages and libraries have been installed. To do that, you can use the following instructions:

       $ pip install -r requirements.txt
In order to execute FedSim you can use the following instruction :

       $ python Main.py 
   
Note, in this case, FedSim will run with default settings. So, to customize implementation settings you should use the following form. 

       $ python Main.py -c 
 
Then, FedSim will set the parameters according to the 'config.csv' file. to change the implementation settings, you should customize the 'config.csv' or you should use your personal config file. In this case, the following instruction must be used:

   
       $ python Main.py -c -f yourfilename.csv
       
Table 2. represents the content of config.csv (note that if you remove parameters from config.csv their default values will be used.)       
![image](https://user-images.githubusercontent.com/92728743/145713386-7d14e6ef-af02-4bd2-b2f0-dc579f824fda.png)


 

 ![image](https://user-images.githubusercontent.com/92728743/141702184-354611e7-ba6e-408e-9174-ab7a41f967ba.png)


 
       fig. 3. illustrates (the non-iid-ib) dataset (horizontal axis = clients and vertical axis = number of samples)

![image](https://user-images.githubusercontent.com/92728743/144513113-e99c8c61-63c6-4a4e-8d52-c67cc3708f5b.png)


       fig. 4. illustrates (the non-iid-ii) dataset (horizontal axis = clients and vertical axis = the number of samples from each class)

### *Acknowledgement:*
This simulator was developed under supervision of Dr. Keyvan RahimiZadeh.

### *References:*
[1] H. Brendan McMahan and Daniel Ramage. Federated learning: Collaborative machine learning without centralized training data. https://research.googleblog.com/2017/04/federated-learning-collaborative.html, 2017.
