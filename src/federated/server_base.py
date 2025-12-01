# src/federated/server.py
import torch
import copy
from scipy.optimize import minimize  # CPU QP 求解器
import numpy as np

class Server:
    
    def __init__(self, model, device='cpu'):
        """
        
        Args:
            model: global
            device: 'cpu' or 'cuda'
        """
        self.global_model = model.to(device)
        self.device = device
        
    def aggregate(self, client_models, client_weights=None):
        """
        FedAvg aggregator
        
        Args:
            client_models: List of client model state_dicts
            client_weights: List of weights (e.g., num_samples)
                          If None, simple average
        """
        if client_weights is None:
            
            client_weights = [1.0] * len(client_models)
        
        # normalize the weight
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        
        global_dict = self.global_model.state_dict()
        
        for key in global_dict.keys():
            
            global_dict[key] = sum(
                client_models[i][key] * client_weights[i]
                for i in range(len(client_models))
            )
        
        self.global_model.load_state_dict(global_dict)

    def aggregate_qffl(self, client_states, client_losses, q=0.2):
        """
        q-FedAvg (q-FFL) 聚合
        参数
        ----
        client_models : list[OrderedDict]  10 个 state_dict
        client_losses : list[float]        10 个本地最终损失 F_i(w_i)
        q : float                          公平系数，q=0 退化为 FedAvg
        """
        n = len(client_states)
        losses = torch.tensor(client_losses, dtype=torch.float32)
        alphas = torch.pow(losses, q)   #(losses + 1e-8).pow(q)
        alphas /= alphas.sum()

        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for idx in range(n):
                global_dict[key] += client_states[idx][key] * alphas[idx]
        self.global_model.load_state_dict(global_dict)

    # -----------------------------
    # MGDA 权重求解（被 aggregate_mgda / aggregate_qffl_mgda 共用）
    # -----------------------------
    def mgda_weights(self, grads, alphas=None):
        """
        grads: list[tensor]  每个元素 shape (d,) 已在一维
        alphas: tensor(K,)  若 None → 退化为纯 MGDA；若给定 → α-MGDA
        return: tensor(K,) 最终权重 w_i
        """
        K = len(grads)
        if alphas is None:
            alphas = torch.ones(K) / K
        # 加权梯度矩阵
        G = torch.stack([alphas[i] * grads[i] for i in range(K)])  # K×d
        Gram = G @ G.T                                             # K×K

        def obj(b): return 0.5 * b @ Gram.numpy() @ b
        def jac(b): return Gram.numpy() @ b

        cons = ({'type': 'eq', 'fun': lambda b: b.sum() - 1.0},
                {'type': 'ineq', 'fun': lambda b: b})
        x0 = np.ones(K) / K
        res = minimize(obj, x0, jac=jac, constraints=cons, method='SLSQP')
        beta = torch.from_numpy(res.x).float()
        w = alphas * beta
        w /= w.sum()
        return w

    # -----------------------------
    # ③ 纯 MGDA 聚合（无 q-FFL 权重）
    # -----------------------------
    def aggregate_mgda(self, client_states, client_grads, alphas=None):
        """
        纯 MGDA 聚合（可被 q-FFL+MGDA 复用）
        参数
        ----
        client_states : list[OrderedDict]  各客户端 state_dict
        client_grads  : list[tensor]       一维梯度向量
        alphas : tensor(K,)  若为 None → 退化为纯 MGDA；若给定 → α-MGDA
        """
        w = self.mgda_weights(client_grads, alphas)   # 复用已有函数
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key])
            for i in range(len(client_states)):
                global_dict[key] += client_states[i][key] * w[i]
        self.global_model.load_state_dict(global_dict)

    # -----------------------------
    # 辅助：一次性返回内存 FedAvg / q-FFL 模型（不污染主流程）
    # -----------------------------
    def _memory_fedavg_qffl(self, global_copy, client_models, client_weights, global_losses):
        """返回 (w_fedavg, w_qffl) 两份内存模型，用于纯对比实验"""
        # FedAvg 分支
        sv_fed = Server(copy.deepcopy(global_copy), device=self.device)
        sv_fed.aggregate(client_models, client_weights)
        w_fed = sv_fed.get_global_model()

        # q-FFL 分支
        sv_qffl = Server(copy.deepcopy(global_copy), device=self.device)
        sv_qffl.aggregate_qffl(client_models, global_losses, q=0.2)
        w_qffl = sv_qffl.get_global_model()

        return w_fed, w_qffl

    def aggregate_qffl_mgda(self, client_states, client_losses, client_grads, q=0.2):
        """
        q-FFL + MGDA 聚合（纯 CPU）
        参数
        ----
        client_states : list[OrderedDict]  每个客户端的 model.state_dict()
        client_losses : list[float]        本地最终损失 F_i
        client_grads  : list[torch.Tensor] 每个客户端的一维梯度向量 g_i
        q : float                          q-FFL 公平系数
        """
        n = len(client_states)
        device = self.device

        # 1. q-FFL 权重 α_i
        losses = torch.tensor(client_losses, dtype=torch.float32) + 1e-8
        alphas = losses.pow(q)
        alphas /= alphas.sum()  # shape (n,)

        # 2. 组装 Gram 矩阵  G_{ij} = (α_i g_i) ⋅ (α_j g_j)
        grad_vec = torch.stack([alphas[i] * client_grads[i] for i in range(n)])  # (n, d)
        Gram = grad_vec @ grad_vec.T  # (n, n)

        # 3. 解最小范数凸组合：min β^T Gram β  s.t. Σβ=1, β≥0
        def objective(b):
            return 0.5 * b @ Gram.numpy() @ b  # 0.5 是为了梯度一致

        def grad_obj(b):
            return Gram.numpy() @ b

        cons = ({'type': 'eq', 'fun': lambda b: b.sum() - 1.0},
                {'type': 'ineq', 'fun': lambda b: b})  # β≥0
        x0 = np.ones(n) / n
        res = minimize(objective, x0, jac=grad_obj, constraints=cons, method='SLSQP')
        beta = torch.from_numpy(res.x).float()  # (n,)

        # 4. 最终权重  w_i = α_i β_i
        w = alphas * beta
        w /= w.sum()

        # 5. 加权聚合参数
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key])
            for i in range(n):
                global_dict[key] += client_states[i][key] * w[i]
        self.global_model.load_state_dict(global_dict)
    
    def get_global_model(self):
        
        return copy.deepcopy(self.global_model)
    
    def evaluate_global(self, test_loader):
        """
        
        Args:
            test_loader: 测试数据的DataLoader
        
        Returns:
            dict: {'loss': , 'rmse': }
        """
        self.global_model.eval()
        
        criterion = torch.nn.MSELoss()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch['user'].to(self.device)
                item_ids = batch['item'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                predictions = self.global_model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                
                total_loss += loss.item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        
        import numpy as np
        rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_targets)) ** 2))
        
        return {
            'loss': avg_loss,
            'rmse': rmse
        }