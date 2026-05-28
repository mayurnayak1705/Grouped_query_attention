import torch 
import torch.nn as nn
import torch.nn.functional as f


class MHA(nn.Module):
    def __init__(self, d_in, d_q, d_v, d_heads):
        super().__init__()
        self.d_in = d_in
        self.d_q = d_q
        self.d_v = d_v
        self.heads = d_heads

        self.w_q = torch.nn.Parameter(torch.rand(d_in, d_q))
        self.w_k = torch.nn.Parameter(torch.rand(d_in, d_q))
        self.w_v = torch.nn.Parameter(torch.rand(d_in, d_v))
        self.W_o = torch.nn.Parameter(torch.rand(d_v, d_in))

    def forward(self, x):
        B, T, _ = x.shape

        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v

        d_head_q = self.d_q // self.heads
        d_head_v = self.d_v // self.heads

        Q = q.view(B, T, self.heads, d_head_q).transpose(1, 2)  # (B, H, T, d_head_q)
        K = k.view(B, T, self.heads, d_head_q).transpose(1, 2)  # (B, H, T, d_head_q)
        V = v.view(B, T, self.heads, d_head_v).transpose(1, 2)  # (B, H, T, d_head_v)

        scores = Q @ K.transpose(-2, -1)                        # (B, H, T, T)
        scores = scores / (d_head_q ** 0.5)                     
        attn = f.softmax(scores, dim=-1)                       

        values = attn @ V                                      
        values = values.transpose(1, 2).contiguous()           
        values = values.view(B, T, self.d_v)                   

        out = values @ self.W_o                                
        return out


class Group_query_attention(nn.module):
    def __init__(self, d_in, d_model, d_heads, d_groups):
        self.d_in = d_in
        self.d_model = d_model
        self.d_heads = d_heads
        self.d_groups = d_groups
        
        assert d_model%d_heads == 0
        assert d_heads%d_groups == 0
        head_dim = self.d_model // self.d_heads

        self.W_q = torch.nn.Parameter(torch.rand(self.d_in, self.d_model))
        self.W_k = torch.nn.Parameter(torch.rand(self.d_in, head_dim * self.d_groups))
        self.W_v = torch.nn.Parameter(torch.rand(self.d_in, head_dim * self.d_groups))
        self.W_o = torch.nn.Parameter(torch.rand(self.d_model, self.d_in))

    def forward(self, x):
        B, T,  _ = x.shape

        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v

        Q = q.view(B,T, self.d_heads, self.d_model//self.d_heads).transpose(1,2)
        K = k.view(B,T, self.d_groups, self.d_model//self.d_heads).transpose(1,2)
        V = v.view(B,T, self.d_groups, self.d_model//self.d_heads).transpose(1,2)

        #The repeat is usually LOGICAL, not physical
        # In real optimized implementations:
        # we do NOT actually store duplicated KV tensors in memory,
        # we do NOT compute separate K/V projections for every query head.
        
        repeat = self.d_heads // self.d_groups
        K = K.repeat_interleave(repeat, dim=1)
        V = V.repeat_interleave(repeat, dim=1)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / ((self.d_model // self.d_heads) ** 0.5)
        attn = f.softmax(scores, dim=-1)  

        values = attn @ V
        values = values.transpose(1, 2).contiguous()           
        values = values.view(B, T, self.d_model)                 

        out = values @ self.W_o                                
        return out




 #just pass the groups as 1 
class Multi_query_attention(nn.module):
    def __init__(self, d_in, d_model, d_heads, d_groups = 1):
        self.d_in = d_in
        self.d_model = d_model
        self.d_heads = d_heads
        self.d_groups = d_groups
        
        assert d_model%d_heads == 0
        assert d_heads%d_groups == 0
        head_dim = self.d_model // self.d_heads

        self.W_q = torch.nn.Parameter(torch.rand(self.d_in, self.d_model))
        self.W_k = torch.nn.Parameter(torch.rand(self.d_in, head_dim * self.d_groups))
        self.W_v = torch.nn.Parameter(torch.rand(self.d_in, head_dim * self.d_groups))
        self.W_o = torch.nn.Parameter(torch.rand(self.d_model, self.d_in))

    def forward(self, x):
        B, T,  _ = x.shape

        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v

        Q = q.view(B,T, self.d_heads, self.d_model//self.d_heads).transpose(1,2)
        K = k.view(B,T, self.d_groups, self.d_model//self.d_heads).transpose(1,2)
        V = v.view(B,T, self.d_groups, self.d_model//self.d_heads).transpose(1,2)

        #The repeat is usually LOGICAL, not physical
        # In real optimized implementations:
        # we do NOT actually store duplicated KV tensors in memory,
        # we do NOT compute separate K/V projections for every query head.
        
        repeat = self.d_heads // self.d_groups
        K = K.repeat_interleave(repeat, dim=1)
        V = V.repeat_interleave(repeat, dim=1)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / ((self.d_model // self.d_heads) ** 0.5)
        attn = f.softmax(scores, dim=-1)  

        values = attn @ V
        values = values.transpose(1, 2).contiguous()           
        values = values.view(B, T, self.d_model)                 

        out = values @ self.W_o                                
        return out
