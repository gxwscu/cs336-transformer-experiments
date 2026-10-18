import torch 
from torch import einsum
class embedding (torch.nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
            super().__init__()
            self.w=torch.nn.Parameter(torch.empty(num_embeddings,embedding_dim))
            mean,std=0,0.02
            torch.nn.init.trunc_normal_(self.w, mean=mean, std=std,a=mean-3*std,b=mean+3*std)
    def forward(self,token_IDs):
          return self.w[token_IDs]