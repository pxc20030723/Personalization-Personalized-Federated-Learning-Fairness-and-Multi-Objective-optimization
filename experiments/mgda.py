# train_ditto_mgda.py
import sys

sys.path.insert(0, '/Users/xinchepeng/Documents/Github_projects/18667_project')

import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

from src.utils.dataset import create_dataloaders
from src.models.base_model import MatrixFactorization
from src.federated.client import ClientDitto
from src.federated.server_base import Server

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, 'results', 'lambda_search')


def main():
    # ============ 配置 ============
    NUM_CLIENTS = 10
    NUM_ROUNDS = 30
    LOCAL_EPOCHS_GLOBAL = 3
    LOCAL_EPOCHS_PERSONALIZED = 3
    BATCH_SIZE = 32
    Global_LEARNING_RATE = 0.01
    Local_LEARNING_RATE = 0.005

    EMBEDDING_DIM = 64
    # Ditto 正则的 lambda
    with open(os.path.join(RESULT_DIR, 'best_lambda_mgda.txt')) as f:
        LAMBDA = float(f.read())
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'federated_data')

    print("=" * 70)
    print("Ditto + MGDA Training - Personalized FL")
    print("=" * 70)
    print(f"device: {DEVICE}")
    print(f"number of clients: {NUM_CLIENTS}")
    print(f"communication rounds: {NUM_ROUNDS}")
    print(f"global local epochs: {LOCAL_EPOCHS_GLOBAL}")
    print(f"batch: {BATCH_SIZE}")
    print(f"personalized local epochs: {LOCAL_EPOCHS_PERSONALIZED}")
    print(f"lambda : {LAMBDA}")

    # ============ 1. 构建全局 user / item 映射 ============
    print("\n[1/5] create global map")

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

    # ============ 2. 创建 Ditto 客户端 ============
    print("\n[2/5] create ditto clients")

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

        client = ClientDitto(
            client_id=client_id,
            train_loader=train_loader,
            test_loader=test_loader,
            device=DEVICE,
            lam=LAMBDA
        )

        clients.append(client)
        print(f"  Client {client_id}: {len(train_df)} train, {len(test_df)} test")

    # ============ 3. 初始化全局模型 ============
    print("\n[3/5] initialize global model")

    global_model = MatrixFactorization(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.2
    )

    server = Server(global_model, device=DEVICE)

    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"  number of parameters: {num_params}")

    # ============ 4. 开始 Ditto + MGDA 训练 ============
    print("\n[4/5] start training ditto + MGDA")
    print("=" * 70)

    history = {
        'round': [],
        'avg_global_train_loss': [],
        'avg_personalized_train_loss': [],
        'avg_global_test_rmse': [],
        'avg_personalized_test_rmse': []
    }

    for round_idx in range(NUM_ROUNDS):
        print(f"\n【Round {round_idx + 1}/{NUM_ROUNDS}】")

        global_model_copy = server.get_global_model()

        client_models = []
        client_grads = []

        global_train_losses = []
        personalized_train_losses = []
        global_test_rmses = []
        personalized_test_rmses = []

        for cid, client in enumerate(clients):
            client.receive_global_model(global_model_copy)

            # 先训练全局模型
            global_loss = client.train_global_model(
                epochs=LOCAL_EPOCHS_GLOBAL,
                lr=Global_LEARNING_RATE
            )
            global_train_losses.append(global_loss)

            # Ditto 个性化模型
            personalized_loss = client.train_personalized_model(
                epochs=LOCAL_EPOCHS_PERSONALIZED,
                lr=Local_LEARNING_RATE
            )
            personalized_train_losses.append(personalized_loss)

            # 评估
            global_metrics = client.evaluate(use_personalized=False)
            personalized_metrics = client.evaluate(use_personalized=True)

            global_test_rmses.append(global_metrics['rmse'])
            personalized_test_rmses.append(personalized_metrics['rmse'])

            # 聚合需要的模型与梯度
            client_models.append(client.get_global_model_params())
            grad_vec = client.compute_grad_one_batch()  # 一小批上的梯度向量
            client_grads.append(grad_vec)

        # ---- MGDA 聚合（不使用 q-FFL 权重，FedAvg）----
        # alphas=None → 纯 MGDA
        server.aggregate_mgda(client_models, client_grads, alphas=None)

        avg_global_loss = sum(global_train_losses) / len(global_train_losses)
        avg_personalized_loss = sum(personalized_train_losses) / len(personalized_train_losses)
        avg_global_rmse = sum(global_test_rmses) / len(global_test_rmses)
        avg_personalized_rmse = sum(personalized_test_rmses) / len(personalized_test_rmses)

        print(f"  global model   - train loss: {avg_global_loss:.4f}, test RMSE: {avg_global_rmse:.4f}")
        print(f"  personalized model - train loss: {avg_personalized_loss:.4f}, test RMSE: {avg_personalized_rmse:.4f}")
        print(f"  improvement: {avg_global_rmse - avg_personalized_rmse:.4f}")

        history['round'].append(round_idx + 1)
        history['avg_global_train_loss'].append(avg_global_loss)
        history['avg_personalized_train_loss'].append(avg_personalized_loss)
        history['avg_global_test_rmse'].append(avg_global_rmse)
        history['avg_personalized_test_rmse'].append(avg_personalized_rmse)

        if (round_idx + 1) % 10 == 0:
            for i in range(NUM_CLIENTS):
                improvement = global_test_rmses[i] - personalized_test_rmses[i]
                print(f"    Client {i}: global_rmse={global_test_rmses[i]:.4f}, "
                      f"personalized_rmse={personalized_test_rmses[i]:.4f}, "
                      f"improvements={improvement:.4f}")

    # ============ 5. 保存结果 ============
    print("\n[5/5] save result")
    print("=" * 70)

    output_dir = os.path.join(BASE_DIR, 'results', 'mgda')
    os.makedirs(output_dir, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_path = f'{output_dir}/training_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f"✓ history saved in : {history_path}")

    # 保存配置
    config_path = f'{output_dir}/config.txt'
    with open(config_path, 'w') as f:
        f.write(f"算法: Ditto + MGDA\n")
        f.write(f"客户端数: {NUM_CLIENTS}\n")
        f.write(f"训练轮数: {NUM_ROUNDS}\n")
        f.write(f"全局模型本地轮数: {LOCAL_EPOCHS_GLOBAL}\n")
        f.write(f"个性化模型本地轮数: {LOCAL_EPOCHS_PERSONALIZED}\n")
        f.write(f"Lambda: {LAMBDA}\n")
        f.write(f"用户数: {NUM_USERS}\n")
        f.write(f"物品数: {NUM_ITEMS}\n")
    print(f"✓ 配置已保存: {config_path}")

    # 打印最终结果
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)

    best_global_rmse = min(history['avg_global_test_rmse'])
    best_personalized_rmse = min(history['avg_personalized_test_rmse'])

    print(f"\n全局模型:")
    print(f"  最佳RMSE: {best_global_rmse:.4f}")
    print(f"  最终RMSE: {history['avg_global_test_rmse'][-1]:.4f}")

    print(f"\n个性化模型:")
    print(f"  最佳RMSE: {best_personalized_rmse:.4f}")
    print(f"  最终RMSE: {history['avg_personalized_test_rmse'][-1]:.4f}")

    print(f"\nDitto改进: {best_global_rmse - best_personalized_rmse:.4f}")


if __name__ == '__main__':
    main()