project/
|- src/
|  |-federated/
|    |-client.py
     |-server_base.py
|  |-models/
     |-base_model.py
|  |-utils/
     |-datapreparation.py
     |-PCA_Kmeans.py
     |-assign_to_client.py
     |-dataset.py
   |-readme.md
|- experiments/
   |-ditto.py
   |-train_fedavg.py
|- results/
|- data/
   |-federated_data/
   |-ml-100k/

step1: use python files in utils to do some data preparation and create user_preferences_count.csv. Consider that there are to many film catagories and one movie can have several related catagories, use PCA_Kmeans.py to decrease the catagory dimension from 19 to 8 and then cluster into 10 clusters and each becomes a client in fl. In data/federated_data/, there are csv for every client. Use dataset.py to transform csv to tensors.

step2: model is basic matixfactorization
step3: in src/federated/client.py there is a specific class for ditto 
step4: run ditto in experiments
step5: results files is used to save training results
