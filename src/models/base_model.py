import torch
import torch.nn as nn

class MatrixFactorization(nn.Module): #inherit base class nn.Module
    """
    inherit model.parameters(), model.to('cuda'), model.state_dic(), model.load_state_dic()
    """
    def __init__(self, num_users, num_items, embedding_dim=64, dropout=0.0):
        super(MatrixFactorization, self).__init__()#Call the constructor of the parent class (nn.Module)


        self.user_embedding=nn.Embedding(num_users, embedding_dim)
        self.item_embedding=nn.Embedding(num_items, embedding_dim)
        self.dropout=nn.Dropout(dropout) if dropout >0 else None

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)


    def forward(self, user_ids, item_ids): #output=model(input)  Automatically invoke forward method
        """
        Args:
            user_ids: (batch_size,) LongTensor
            item_ids: (batch_size,) LongTensor
        
        Returns:
            predictions: (batch_size,) FloatTensor
        """
        # 获取embedding
        user_embeds = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        item_embeds = self.item_embedding(item_ids)  # (batch_size, embedding_dim)
        user_embeds=self.dropout(user_embeds)
        item_embeds=self.dropout(item_embeds)
        # 点积预测评分
        predictions = (user_embeds * item_embeds).sum(dim=1)  # (batch_size,)
        
        return predictions


