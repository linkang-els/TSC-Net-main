import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





class MHSA_Intra(nn.Module):
    """
    compute intra-segment attention
    """
    def __init__(self, dim_in, heads, num_pos, pos_enc_type='relative', use_pos=True):
        super(MHSA_Intra, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = self.dim_in
        self.heads = heads
        self.dim_head = self.dim_inner // self.heads
        self.num_pos = num_pos

        self.scale = self.dim_head ** -0.5

        self.conv_query = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_key = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_value = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_out = nn.Conv1d(
            self.dim_inner, self.dim_in, kernel_size=1, stride=1, padding=0
        )
        self.bn = nn.BatchNorm1d(
            num_features=self.dim_in, eps=1e-5, momentum=0.1
        )
        self.bn.weight.data.zero_()
        self.bn.bias.data.zero_()
    
    def forward(self, input, intra_attn_mask = None):
        B, C, T = input.shape
        query = self.conv_query(input).view(B, self.heads, self.dim_head, T).permute(0, 1, 3, 2).contiguous() #(B, h, T, dim_head)
        key = self.conv_key(input).view(B, self.heads, self.dim_head, T) #(B, h, dim_head, T)
        value = self.conv_value(input).view(B, self.heads, self.dim_head, T).permute(0, 1, 3, 2).contiguous() #(B, h, T, dim_head)

        query *= self.scale
        sim = torch.matmul(query, key) #(B, h, T, T)
        attn = F.softmax(sim, dim=-1) #(B, h, T, T)
        attn = torch.nan_to_num(attn, nan=0.0)
        output = torch.matmul(attn, value) #(B, h, T, dim_head)

        output = output.permute(0, 1, 3, 2).contiguous().view(B, C, T) #(B, C, T)
        output = input + self.bn(self.conv_out(output))
        return output