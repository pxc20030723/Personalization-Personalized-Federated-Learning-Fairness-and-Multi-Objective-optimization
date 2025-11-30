import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class RatingsDataset(Dataset):
    def __init__(self, rating_df, user_id_map=None, item_id_map=None):
        """
        Args:
         rating_df: dataframe  (every client has a csv file can be transformed)
         user_id_map:dict, {original_user_id: continuous index }
         item_id_map:
        """
        self.ratings=rating_df.copy()

        if user_id_map is None:
            unique_users=sorted(self.ratings['user_id'].unique())
            self.user_id_map={uid: idx for idx,uid in enumerate(unique_users)}
        else:
            self.user_id_map=user_id_map
        
        if item_id_map is None:
            unique_items = sorted(self.ratings['item_id'].unique())
            self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        else:
            self.item_id_map = item_id_map

        self.ratings['user_idx'] = self.ratings['user_id'].map(self.user_id_map)
        self.ratings['item_idx'] = self.ratings['item_id'].map(self.item_id_map)
        self.ratings = self.ratings.dropna(subset=['user_idx', 'item_idx']) #filter


        self.users=torch.LongTensor(self.ratings['user_idx'].values)
        self.items = torch.LongTensor(self.ratings['item_idx'].values)
        self.ratings_values = torch.FloatTensor(self.ratings['rating'].values)
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self,idx):
        """
        Args:
        idx (0  len-1)
        Returns:
        dict:{'user':tensor, 'item':tensor, 'ratings':tensor}
    
        """
        return {'user':self.users[idx], 'item': self.items[idx], 'rating': self.ratings_values[idx]
}
    @property
    def num_users(self):
        return len(self.user_id_map)
    
    @property
    def num_items(self):
        return len(self.item_id_map)
    

def create_dataloaders(train_df, test_df, batch_size=32, user_id_map=None, item_id_map=None):
    train_dataset = RatingsDataset(train_df, user_id_map, item_id_map)
    test_dataset = RatingsDataset(
        test_df, 
        user_id_map=train_dataset.user_id_map,
        item_id_map=train_dataset.item_id_map
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader, train_dataset.user_id_map, train_dataset.item_id_map
