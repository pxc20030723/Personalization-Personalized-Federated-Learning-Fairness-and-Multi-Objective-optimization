# tune_ditto_lambda.py
"""
scan lambda, opt output, support multiple aggregation modes:
  - fedavg  -> Server.aggregate
  - qffl    -> Server.aggregate_qffl
  - mgda    -> Server.aggregate_mgda
  - perfl   -> Server.aggregate_qffl_mgda
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.dataset import create_dataloaders
from src.models.base_model import MatrixFactorization
from src.federated.client import ClientDitto
from src.federated.server_base import Server

# ========= 可改常量 =========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'federated_data')
RESULT_DIR = os.path.join(BASE_DIR, 'results', 'lambda_search')
NUM_CLIENTS = 10
NUM_ROUNDS_SEARCH = 10      # <30
LOCAL_EPOCHS_GLOBAL = 3
LOCAL_EPOCHS_PERSONALIZED = 5
BATCH_SIZE = 256
EMBEDDING_DIM = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAMBDA_LIST = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]
Q_FAIR = 0.2  # 给 q-FFL / perFL 用
# ============================

os.makedirs(RESULT_DIR, exist_ok=True)


def build_global_maps():
    """构建全局 user/item 映射"""
    all_users, all_items = set(), set()
    for cid in range(NUM_CLIENTS):
        df = pd.read_csv(f'{DATA_DIR}/client_{cid}.csv')
        all_users.update(df['user_id'])
        all_items.update(df['item_id'])
    global_user_map = {uid: idx for idx, uid in enumerate(sorted(all_users))}
    global_item_map = {iid: idx for idx, iid in enumerate(sorted(all_items))}
    return global_user_map, global_item_map


def build_clients(global_user_map, global_item_map):
    """预先把每个客户端的数据和 dataloader 建好，后面循环 lambda 时复用"""
    clients = []
    for cid in range(NUM_CLIENTS):
        df = pd.read_csv(f'{DATA_DIR}/client_{cid}.csv')
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        tr_loader, va_loader, _, _ = create_dataloaders(
            train_df, val_df, BATCH_SIZE, global_user_map, global_item_map
        )
        clients.append({
            'id': cid,
            'train_df': train_df,
            'val_df': val_df,
            'tr_loader': tr_loader,
            'va_loader': va_loader
        })
    return clients


def aggregate_once(mode, server, client_instances, q_fair):
    """
    针对当前 mode，在一轮通信中进行一次聚合：
      - fedavg: Server.aggregate(models, weights)
      - qffl:   Server.aggregate_qffl(models, losses, q)
      - mgda:   Server.aggregate_mgda(models, grads, alphas=None)
      - perfl:  Server.aggregate_qffl_mgda(models, losses, grads, q)
    """
    global_copy = server.get_global_model()
    models, weights = [], []
    losses, grads = [], []

    for cl in client_instances:
        cl.receive_global_model(global_copy)
        # 全局模型训练
        global_loss = cl.train_global_model(
            epochs=LOCAL_EPOCHS_GLOBAL,
            lr=0.005
        )
        # 个性化模型训练
        cl.train_personalized_model(
            epochs=LOCAL_EPOCHS_PERSONALIZED,
            lr=0.008
        )

        models.append(cl.get_global_model_params())
        weights.append(cl.get_num_samples())
        losses.append(global_loss)

        # MGDA / perFL 需要梯度
        if mode in ('mgda', 'perfl'):
            grads.append(cl.compute_grad_one_batch())

    if mode == 'fedavg':
        server.aggregate(models, weights)
    elif mode == 'qffl':
        server.aggregate_qffl(models, losses, q=q_fair)
    elif mode == 'mgda':
        server.aggregate_mgda(models, grads, alphas=None)
    elif mode == 'perfl':
        server.aggregate_qffl_mgda(models, losses, grads, q=q_fair)
    else:
        raise ValueError(f'Unknown mode: {mode}')


def tune_lambda(mode: str):
    """主流程：在给定聚合模式下扫描 lambda，输出 best_lambda_{mode}.txt"""
    mode = mode.lower()
    assert mode in ('fedavg', 'qffl', 'mgda', 'perfl')

    print(f'>>> Tuning Ditto lambda for mode = {mode}')

    # 1. 全局映射
    global_user_map, global_item_map = build_global_maps()
    NUM_USERS, NUM_ITEMS = len(global_user_map), len(global_item_map)

    # 2. 客户端数据缓存
    client_meta = build_clients(global_user_map, global_item_map)

    # 3. 不同 lambda 顺序跑
    results = []          # 每行：lambda, client_id, rmse
    best_per_client = {i: {'rmse': 1e9, 'lambda': None} for i in range(NUM_CLIENTS)}

    for lam in LAMBDA_LIST:
        print(f'\n===== mode = {mode}, lambda = {lam} =====')

        # 重建模型 & 服务器
        global_model = MatrixFactorization(
            NUM_USERS, NUM_ITEMS, EMBEDDING_DIM, dropout=0.2
        ).to(DEVICE)
        server = Server(global_model, device=DEVICE)

        # 重建带当前 lambda 的 ClientDitto
        client_instances = []
        for c in client_meta:
            cl = ClientDitto(
                client_id=c['id'],
                train_loader=c['tr_loader'],
                test_loader=c['va_loader'],
                device=DEVICE,
                lam=lam
            )
            client_instances.append(cl)

        # 跑 NUM_ROUNDS_SEARCH 轮
        for rnd in range(NUM_ROUNDS_SEARCH):
            aggregate_once(mode, server, client_instances, Q_FAIR)

        # 评估个性化模型在验证集上的 RMSE
        for cl in client_instances:
            rmse = cl.evaluate(use_personalized=True)['rmse']
            results.append({'lambda': lam, 'client': cl.client_id, 'rmse': rmse})
            if rmse < best_per_client[cl.client_id]['rmse']:
                best_per_client[cl.client_id]['rmse'] = rmse
                best_per_client[cl.client_id]['lambda'] = lam

    # 4. 保存
    prefix = f'lambda_search_{mode}'

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULT_DIR, f'{prefix}_lambda_vs_rmse.csv'), index=False)

    best_df = pd.DataFrame(
        [{'client': i,
          'best_lambda': best_per_client[i]['lambda'],
          'best_rmse': best_per_client[i]['rmse']}
         for i in range(NUM_CLIENTS)]
    )
    best_df.to_csv(os.path.join(RESULT_DIR, f'{prefix}_best_lambda_per_client.csv'), index=False)

    # 全局统计
    avg_lambda = best_df['best_lambda'].mean()
    print(f'\n>>> mode = {mode}, 平均最优 λ ≈ {avg_lambda:.3f}')
    out_txt = os.path.join(RESULT_DIR, f'best_lambda_{mode}.txt')
    with open(out_txt, 'w') as f:
        f.write(str(avg_lambda))
    print(f'>>> 已写入 {out_txt}，该模式下的正式训练可直接读取。')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        default='fedavg',
        choices=['fedavg', 'qffl', 'mgda', 'perfl'],
        help="aggregation mode for tuning lambda"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    tune_lambda(args.mode)