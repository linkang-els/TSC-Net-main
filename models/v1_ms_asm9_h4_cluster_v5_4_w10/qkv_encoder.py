import numpy as np
import torch
from torch import nn
import torch.nn.functional as F




class ScaledDotProductGeometryAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.5, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductGeometryAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model  # 2048
#        print("*******************************")
#        print("d_model: ", d_model)
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.scale = self.h ** -0.5

        self.init_weights()

        self.bn = nn.BatchNorm1d(d_model, eps=1e-5, momentum=0.1)
        self.bn.weight.data.zero_()
        self.bn.bias.data.zero_()

        self.comment = comment
        # if True:
            # self.head_conv = nn.Conv2d(h, h, kernel_size=3, stride=1, padding=1)
            # self.head_norm = nn.InstanceNorm2d(h)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        q *= self.scale
        sim = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)


        attn = F.softmax(sim, dim=-1) # # (b_s, h, nq, nq)

        attn = torch.nan_to_num(attn, nan=0.0)

        attn = self.dropout(attn)


        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        # 层正则化
        x1 = self.bn(out.permute(0, 2, 1))

        return x1



class MultiHeadGeometryAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k=64, d_v=64, h=8, dropout=.5, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadGeometryAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering

        self.attention = ScaledDotProductGeometryAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)

        self.dropout = nn.Dropout(p=dropout)
#        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        # self-attention层
        out = self.attention(queries, keys, values, attention_mask)
        out = self.dropout(out)
        # 残差连接
        out = queries + out.permute(0, 2, 1) # self.bn(out)

        return out

class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=2048, d_ff=2048, dropout=.5):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
#        self.layer_norm = nn.LayerNorm(d_model)

        self.bn = nn.BatchNorm1d(d_model, eps=1e-5, momentum=0.1)
        self.bn.weight.data.zero_()
        self.bn.bias.data.zero_()

    def forward(self, input):

        out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
        out = self.dropout(out)
        # 层正则化
        x1 = self.bn(out.permute(0, 2, 1))
        out = x1.permute(0, 2, 1)  
        # 残差连接
        out = input + out
        return out
    
# 处理网格特征
class GridEncoderLayer(nn.Module):
    def __init__(self, d_model=2048, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.5):
        super(GridEncoderLayer, self).__init__()
        self.mhatt = MultiHeadGeometryAttention(d_model, d_k, d_v, h, dropout)
        self.dropout = nn.Dropout(dropout)
#        self.layer_norm = nn.LayerNorm(d_model)
        self.bn = nn.BatchNorm1d(d_model, eps=1e-5, momentum=0.1)
        self.bn.weight.data.zero_()
        self.bn.bias.data.zero_()      
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, queries, keys, values, attention_mask):
        queries = queries.permute(0, 2, 1)
        #keys = keys.permute(0, 2, 1)
        #values = values.permute(0, 2, 1)

        # 多头注意力
        att = self.mhatt(queries, keys, values, attention_mask)
        # 层正则化
        x1 = self.bn(att.permute(0, 2, 1))
        att = x1.permute(0, 2, 1) 
        # 残差
        att = queries + self.dropout(att)
        # 前馈层                                                
        ff = self.pwff(att)
        out = ff.permute(0, 2, 1)
        return out 