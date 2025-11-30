import torch
import torch.nn as nn

class MatrixFactorization(nn.Module): #inherit base class nn.Module
    """
    inherit model.parameters(), model.to('cuda'), model.state_dic(), model.load_state_dic()
    """
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(MatrixFactorization, self).__init__()#Call the constructor of the parent class (nn.Module)


        self.user_embedding=nn.Embedding(num_users, embedding_dim)
        self.item_embedding=nn.Embedding(num_items, embedding_dim)

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
        
        # 点积预测评分
        predictions = (user_embeds * item_embeds).sum(dim=1)  # (batch_size,)
        
        return predictions


# test model
import torch

# 创建模型
model = MatrixFactorization(num_users=943, num_items=1682, embedding_dim=64)

# 测试前向传播
user_ids = torch.LongTensor([0, 1, 2, 3, 4])
item_ids = torch.LongTensor([10, 20, 30, 40, 50])

predictions = model(user_ids, item_ids)

print(f"输入user_ids: {user_ids}")
print(f"输入item_ids: {item_ids}")
print(f"预测评分: {predictions}")
print(f"预测shape: {predictions.shape}")  # 应该是 torch.Size([5])