# src/federated/test_server.py
import sys
sys.path.insert(0, '/Users/xinchepeng/Documents/Github_projects/18667_project')

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.dataset import create_dataloaders
from src.models.base_model import MatrixFactorization
from src.federated.client import Client
from src.federated.server_base import Server

print("="*70)
print("测试 Server 类")
print("="*70)

# ============ 1. 创建全局映射 ============
print("\n[1/6] 创建全局user/item映射...")

# 收集所有客户端的user和item
all_users = set()
all_items = set()

for client_id in range(2):
    df = pd.read_csv(f'/Users/xinchepeng/Documents/Github_projects/18667_project/data/federated_data/client_{client_id}.csv')
    all_users.update(df['user_id'].unique())
    all_items.update(df['item_id'].unique())

# 创建全局映射
global_user_map = {uid: idx for idx, uid in enumerate(sorted(all_users))}
global_item_map = {iid: idx for idx, iid in enumerate(sorted(all_items))}

NUM_USERS = len(global_user_map)
NUM_ITEMS = len(global_item_map)

print(f"  ✓ 全局用户数: {NUM_USERS}")
print(f"  ✓ 全局物品数: {NUM_ITEMS}")

# ============ 2. 准备客户端数据（使用全局映射）============
print("\n[2/6] 准备客户端数据...")

clients = []
for client_id in range(2):
    df = pd.read_csv(f'/Users/xinchepeng/Documents/Github_projects/18667_project/data/federated_data/client_{client_id}.csv')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # ✅ 关键：使用全局映射
    train_loader, test_loader, _, _ = create_dataloaders(
        train_df, test_df, 
        batch_size=256,
        user_id_map=global_user_map,  # ← 使用全局映射
        item_id_map=global_item_map   # ← 使用全局映射
    )
    
    client = Client(client_id, train_loader, test_loader, device='cpu')
    clients.append(client)
    print(f"  ✓ Client {client_id}: {len(train_df)} train samples")

# ============ 3. 创建全局模型和服务器 ============
print("\n[3/6] 创建全局模型和服务器...")

global_model = MatrixFactorization(
    num_users=NUM_USERS,  # ← 使用全局用户数
    num_items=NUM_ITEMS,  # ← 使用全局物品数
    embedding_dim=64
)

server = Server(global_model, device='cpu')
print(f"  ✓ 服务器创建成功")
print(f"  ✓ 模型user_embedding: {global_model.user_embedding.num_embeddings}")
print(f"  ✓ 模型item_embedding: {global_model.item_embedding.num_embeddings}")

# ============ 4. 模拟一轮联邦学习 ============
print("\n[4/6] 模拟联邦学习...")

# 下发全局模型
print("\n  下发全局模型到客户端...")
for client in clients:
    client.receive_model(server.get_global_model())
    print(f"    ✓ Client {client.client_id} 接收模型")

# 客户端本地训练
print("\n  客户端本地训练...")
client_models = []
client_weights = []

for client in clients:
    train_loss = client.train(epochs=3, lr=0.001)
    print(f"    Client {client.client_id}: 训练损失={train_loss:.4f}")
    
    client_models.append(client.get_local_model_params())
    client_weights.append(client.get_num_samples())

# ============ 5. 服务器聚合 ============
print("\n[5/6] 服务器聚合...")
print(f"  客户端权重: {client_weights}")

server.aggregate(client_models, client_weights)
print(f"  ✓ 聚合完成")

# ============ 6. 评估全局模型 ============
print("\n[6/6] 评估聚合后的全局模型...")

for client in clients:
    # 用聚合后的全局模型评估
    client.receive_model(server.get_global_model())
    metrics = client.evaluate()
    print(f"  Client {client.client_id}: RMSE={metrics['rmse']:.4f}")

print("\n" + "="*70)
print("✅ Server 测试通过！")
print("="*70)