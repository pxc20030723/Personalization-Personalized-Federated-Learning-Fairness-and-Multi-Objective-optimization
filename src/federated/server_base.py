# src/federated/server.py
import torch
import copy

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