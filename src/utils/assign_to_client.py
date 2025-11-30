import pandas as pd
import os

original_ratings='data/ml-100k/u.data'
cluster_data='data/user_cluster_mapping.csv'
output_dir='data/federated_data'
NUM_CLIENTS=10
#step1 read cluster results
cluster_mapping=pd.read_csv(cluster_data)
#step2 read ratings
ratings=pd.read_csv(original_ratings, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
#step3 merge
ratings_with_client=ratings.merge(cluster_mapping[['user_id', 'cluster']], on='user_id', how='inner')

n_lost=len(ratings)-len(ratings_with_client)
if n_lost>0:
    print(f"lose {n_lost} ratings")
#
stats=[]
for client_id in range(NUM_CLIENTS):
    client_data=ratings_with_client[ratings_with_client['cluster']==client_id]
    if len(client_data)==0:
        print(f"client{client_id} no data")
        continue
    client_data = client_data[['user_id', 'item_id', 'rating', 'timestamp']]
    output_path = f'{output_dir}/client_{client_id}.csv'
    client_data.to_csv(output_path, index=False)
    
    
