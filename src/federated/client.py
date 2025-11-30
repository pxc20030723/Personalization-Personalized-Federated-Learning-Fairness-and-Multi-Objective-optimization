import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import numpy as np



class Client:
    def __init__(self, client_id, train_loader, test_loader, device='cpu'):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model = None
        self.criterion = nn.MSELoss()

    def receive_model(self, model):
        self.model = copy.deepcopy(model).to(self.device)
    def train(self, epochs=1, lr=0.001):
        if self.model is None:
            raise ValueError("model not set yet run set_model() first")
        
        self.model.train()
        
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch in self.train_loader:
                
                user_ids = batch['user'].to(self.device)
                item_ids = batch['item'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(self.train_loader)
        
        avg_loss = total_loss / epochs
        
        return avg_loss

    def evaluate(self):
        """
        Returns:
            dict: {'loss': 测试损失, 'rmse': RMSE}
        """
        if self.model is None:
            raise ValueError("model not set yet run set_model() first")
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batch in self.test_loader:
                user_ids = batch['user'].to(self.device)
                item_ids = batch['item'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                # 前向传播
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                
                total_loss += loss.item()
                
                # 收集预测和真实值
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / len(self.test_loader)
        import numpy as np
        rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_targets)) ** 2))
        
        return {
            'loss': avg_loss,
            'rmse': rmse
        }

    def get_local_model_params(self):
        return copy.deepcopy(self.model.state_dict())
          
    def get_num_samples(self):
     
        return len(self.train_loader.dataset)


class ClientDitto:

    def __init__(self, client_id, train_loader, test_loader, device='cpu', lam=0.1):
        self.client_id=client_id
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.lam = lam

        self.global_model=None
        self.personalized_model=None
        self.criterion=nn.MSELoss()
    
    def receive_global_model(self, model):
        self.global_model=copy.deepcopy(model).to(self.device)
        #initialize personalized model in first time
        if self.personalized_model is None:
            self.personalized_model=copy.deepcopy(model).to(self.device)
    def train_global_model(self, epochs=1, lr=0.001):
        """
        return avg_loss
        """
        self.global_model.train()
        optimizer=optim.AdamW(self.global_model.parameters(), lr=lr, weight_decay=1e-5)
        total_loss=0
        for epoch in range(epochs):
            epoch_loss=0
            for batch in self.train_loader:
                user_ids = batch['user'].to(self.device)
                item_ids = batch['item'].to(self.device)
                ratings = batch['rating'].to(self.device)
                predictions = self.global_model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss+=loss.item()
            total_loss+=epoch_loss/len(self.train_loader)
        return total_loss/ epochs
    
    def train_personalized_model(self, epochs=1, lr=0.001):
        self.personalized_model.train()
        optimizer = optim.AdamW(self.personalized_model.parameters(), lr=lr,weight_decay=1e-4)

        global_params={
            name: param.clone().detach() 
            for name, param in self.global_model.named_parameters()
        }
        total_loss=0
        for epoch in range(epochs):
            epoch_loss=0
            for batch in self.train_loader:
                user_ids = batch['user'].to(self.device)
                item_ids = batch['item'].to(self.device)
                ratings = batch['rating'].to(self.device)
                predictions = self.personalized_model(user_ids, item_ids)
                data_loss = self.criterion(predictions, ratings)

                #normalization loss
                reg_loss=0
                for name, param in self.personalized_model.named_parameters():
                    reg_loss += torch.sum((param - global_params[name]) ** 2)
                reg_loss = (self.lam / 2) * reg_loss

                loss=data_loss+reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(self.train_loader)
        
        return total_loss / epochs

    def evaluate(self, use_personalized=True):

            """
        Args:
            use_personalized: True->personalized model Flase->global model
        
        Returns:
            dict: {'loss': test loss, 'rmse': RMSE}
            """
            model = self.personalized_model if use_personalized else self.global_model
            model.eval()
            total_loss = 0
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for batch in self.test_loader:

                    user_ids = batch['user'].to(self.device)
                    item_ids = batch['item'].to(self.device)
                    ratings = batch['rating'].to(self.device)
                
                    predictions = model(user_ids, item_ids)
                    loss = self.criterion(predictions, ratings)
                
                    total_loss += loss.item()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(ratings.cpu().numpy())
            avg_loss = total_loss / len(self.test_loader)
            rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_targets)) ** 2))
        
            return {'loss': avg_loss, 'rmse': rmse}

    def get_global_model_params(self):
       
        return copy.deepcopy(self.global_model.state_dict())
    
    def get_num_samples(self):
       
        return len(self.train_loader.dataset)       


















