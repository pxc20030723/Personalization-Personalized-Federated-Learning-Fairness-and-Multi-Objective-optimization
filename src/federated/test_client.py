# test_client.py
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

project_root = '/Users/xinchepeng/Documents/Github_projects/18667_project'
sys.path.insert(0, project_root)

from src.utils.dataset import create_dataloaders
from src.models.base_model import MatrixFactorization
from src.federated.client import Client

print("="*70)
print("测试 Client 类")
print("="*70)

# ============ 1. 准备数据 ============
print("\n[1/5] 准备数据...")
df = pd.read_csv('data/federated_data/client_0.csv')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_loader, test_loader, user_map, item_map = create_dataloaders(
    train_df, test_df, batch_size=256
)

print(f"  ✓ 训练样本: {len(train_df)}")
print(f"  ✓ 测试样本: {len(test_df)}")
print(f"  ✓ 用户数: {len(user_map)}")
print(f"  ✓ 物品数: {len(item_map)}")

# ============ 2. 创建客户端 ============
print("\n[2/5] 创建客户端...")
client = Client(
    client_id=0,
    train_loader=train_loader,
    test_loader=test_loader,
    device='cpu'
)
print(f"  ✓ Client {client.client_id} 创建成功")

# ============ 3. 创建模型 ============
print("\n[3/5] 创建模型...")
model = MatrixFactorization(
    num_users=len(user_map),
    num_items=len(item_map),
    embedding_dim=64
)
num_params = sum(p.numel() for p in model.parameters())
print(f"  ✓ 模型参数量: {num_params:,}")

# ============ 4. 下发模型并训练前评估 ============
print("\n[4/5] 下发模型...")
client.receive_model(model)
print(f"  ✓ 模型已下发")

print("\n  训练前评估:")
metrics_before = client.evaluate()
print(f"    损失: {metrics_before['loss']:.4f}")
print(f"    RMSE: {metrics_before['rmse']:.4f}")

# ============ 5. 本地训练 ============
print("\n[5/5] 本地训练...")
train_loss = client.train(epochs=5, lr=0.001)
print(f"  ✓ 训练完成")
print(f"    训练损失: {train_loss:.4f}")

print("\n  训练后评估:")
metrics_after = client.evaluate()
print(f"    损失: {metrics_after['loss']:.4f}")
print(f"    RMSE: {metrics_after['rmse']:.4f}")

# ============ 6. 检查参数上传 ============
print("\n检查参数上传:")
model_params = client.get_local_model_params()
print(f"  ✓ 参数数量: {len(model_params)}")
print(f"  ✓ 训练样本数: {client.get_num_samples()}")

print("\n" + "="*70)
print("✅ Client 测试通过！")
print("="*70)