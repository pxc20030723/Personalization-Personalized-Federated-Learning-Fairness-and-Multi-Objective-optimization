# train_fedavg.py
import sys
sys.path.insert(0, '/Users/xinchepeng/Documents/Github_projects/18667_project')

import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from src.utils.dataset import create_dataloaders
from src.models.base_model import MatrixFactorization
from src.federated.client import Client
from src.federated.server_base import Server

def main():

    NUM_CLIENTS = 10
    NUM_ROUNDS = 30   # 20
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 32    # 64
    LEARNING_RATE = 0.001
    EMBEDDING_DIM = 64
    Local_LEARNING_RATE = 0.005
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # DATA_DIR = '/Users/xinchepeng/Documents/Github_projects/18667_project/data/federated_data'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'federated_data')
    
    print("="*70)
    print("Federated learning train---FedAvg")
    print("="*70)
    print(f"device: {DEVICE}")
    print(f"number of clients: {NUM_CLIENTS}")
    print(f"communication rounds: {NUM_ROUNDS}")
    print(f"local epochs: {LOCAL_EPOCHS}")
    print(f"Batch: {BATCH_SIZE}")
    print(f"lr: {LEARNING_RATE}")
    print(f"Embedding: {EMBEDDING_DIM}")
    
    
    print("\n[1/5] creat global map")
    
    all_users = set()
    all_items = set()
    
    for client_id in range(NUM_CLIENTS):
        df = pd.read_csv(f'{DATA_DIR}/client_{client_id}.csv')
        all_users.update(df['user_id'].unique())
        all_items.update(df['item_id'].unique())
    
    global_user_map = {uid: idx for idx, uid in enumerate(sorted(all_users))}
    global_item_map = {iid: idx for idx, iid in enumerate(sorted(all_items))}
    
    NUM_USERS = len(global_user_map)
    NUM_ITEMS = len(global_item_map)
    
    print(f"  global users: {NUM_USERS}")
    print(f"  global items: {NUM_ITEMS}")
    

    print("\n[2/5] create clients.")
    
    clients = []
    
    for client_id in range(NUM_CLIENTS):
        df = pd.read_csv(f'{DATA_DIR}/client_{client_id}.csv')
        
       
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        
        train_loader, test_loader, _, _ = create_dataloaders(
            train_df, test_df,
            batch_size=BATCH_SIZE,
            user_id_map=global_user_map,
            item_id_map=global_item_map
        )
        
        
        client = Client(
            client_id=client_id,
            train_loader=train_loader,
            test_loader=test_loader,
            device=DEVICE
        )
        
        clients.append(client)
        print(f"  Client {client_id}: {len(train_df)} train, {len(test_df)} test")
    
  
    print("\n[3/5] initialize global model")
    
    global_model = MatrixFactorization(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM
    )
    
    server = Server(global_model, device=DEVICE)
    
    num_params = sum(p.numel() for p in global_model.parameters())
    
    
   
    print("\n[4/5] begin training")
    print("="*70)
    
    history = {
        'round': [],
        'avg_train_loss': [],
        'avg_test_rmse': [],
        'min_test_rmse': [],
        'max_test_rmse': []
    }
    
    for round_idx in range(NUM_ROUNDS):
        print(f"\n【Round {round_idx + 1}/{NUM_ROUNDS}】")
        
       
        global_model_copy = server.get_global_model()
        
        
        client_models = []
        client_weights = []
        round_train_losses = []
        round_test_rmses = []
        
        for client in clients:
            
            client.receive_model(global_model_copy)
            
            
            train_loss = client.train(epochs=LOCAL_EPOCHS, lr=LEARNING_RATE)
            round_train_losses.append(train_loss)
            
            
            metrics = client.evaluate()
            round_test_rmses.append(metrics['rmse'])
            
            #
            client_models.append(client.get_local_model_params())
            client_weights.append(client.get_num_samples())
        
        # server aggerate
        server.aggregate(client_models, client_weights)
        
        
        avg_train_loss = sum(round_train_losses) / len(round_train_losses)
        avg_test_rmse = sum(round_test_rmses) / len(round_test_rmses)
        min_test_rmse = min(round_test_rmses)
        max_test_rmse = max(round_test_rmses)
        
        print(f"  average train loss: {avg_train_loss:.4f}")
        print(f"  average test rmse: {avg_test_rmse:.4f} (min={min_test_rmse:.4f}, max={max_test_rmse:.4f})")
        
        
        history['round'].append(round_idx + 1)
        history['avg_train_loss'].append(avg_train_loss)
        history['avg_test_rmse'].append(avg_test_rmse)
        history['min_test_rmse'].append(min_test_rmse)
        history['max_test_rmse'].append(max_test_rmse)
        
        
        if (round_idx + 1) % 10 == 0:
            print(f"\n  client test RMSE:")
            for i, rmse in enumerate(round_test_rmses):
                print(f"    Client {i}: {rmse:.4f}")
    
  
    print("\n[5/5] save result...")
    print("="*70)
    
   
    output_dir = '/Users/xinchepeng/Documents/Github_projects/18667_project/results'
    os.makedirs(output_dir, exist_ok=True)
    
    
    model_path = f'{output_dir}/global_model_fedavg.pth'
    torch.save(server.global_model.state_dict(), model_path)
    
    
    
    history_df = pd.DataFrame(history)
    history_path = f'{output_dir}/training_history_fedavg.csv'
    history_df.to_csv(history_path, index=False)
    print(f"✓ training history: {history_path}")
    
    #
    config_path = f'{output_dir}/config_fedavg.txt'
    with open(config_path, 'w') as f:
        f.write(f"NUM_CLIENTS: {NUM_CLIENTS}\n")
        f.write(f"NUM_ROUNDS: {NUM_ROUNDS}\n")
        f.write(f"LOCAL_STEPS: {LOCAL_EPOCHS}\n")
        f.write(f"Batch SIZE: {BATCH_SIZE}\n")
        f.write(f"LEARNING RATE: {LEARNING_RATE}\n")
        f.write(f"EMBEDDING: {EMBEDDING_DIM}\n")
        f.write(f"NUM_USERS: {NUM_USERS}\n")
        f.write(f"NUM_ITEMS: {NUM_ITEMS}\n")
        f.write(f"MODEL PARAMETER AMOUNT: {num_params:,}\n")
    print(f"✓ SAVE SUCCESSFULLY: {config_path}")
    

    print("\n" + "="*70)
    print("FINISH!")
    print("="*70)
    best_round = history['avg_test_rmse'].index(min(history['avg_test_rmse'])) + 1
    best_rmse = min(history['avg_test_rmse'])
    print(f"BEST AVERAGE TEST RMSE: {best_rmse:.4f} (Round {best_round})")
    print(f"FINAL AVERAGFE TEST RMSE: {history['avg_test_rmse'][-1]:.4f}")

if __name__ == '__main__':
    main()