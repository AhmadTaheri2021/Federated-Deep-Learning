# *Federated-Deep-Learning*
An implementation of Federated Deep Learning for classification task. 

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
 In this project a Federated Convolutional Neural Network (CNN) model is utilized for classification task. So that, the 'Fashion MNIST' dataset is used in non-iid form (fig. 2).  in this section contents of the project will be described in detail. The following
 
 ### *Contents :* 
 1. Server_side.py 
 2. Client_side.py
 3. Data_Preprocessing.py
 4. Utils.py
 5. Main.py
 
 (NOTE, In order to run the project, it is just needed to execut 'Main.py' file.) 


  
 ![image](https://user-images.githubusercontent.com/92728743/141702184-354611e7-ba6e-408e-9174-ab7a41f967ba.png)


 
       fig. 2. illustrates the non-iid dataset 


### *References:*
[1] H. Brendan McMahan and Daniel Ramage. Federated learning: Collaborative machine learning without centralized training data. https://research.googleblog.com/2017/04/federated-learning-collaborative.html, 2017.
