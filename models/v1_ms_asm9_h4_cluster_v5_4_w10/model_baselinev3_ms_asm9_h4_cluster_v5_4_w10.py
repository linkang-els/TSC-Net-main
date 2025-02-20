import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import utils.wsad_utils as utils
from torch.nn import init
from multiprocessing.dummy import Pool as ThreadPool
import copy
import h5py
#导入聚类的encoder
from .qkv_encoder import GridEncoderLayer
#导入时间增强分支的encoder
from .asm_transfomer import MHSA_Intra

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)

        
                       
    
class TSC_Net(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim=2048
        mid_dim=1024
        self.dataset_name = args['opt'].dataset_name
        dropout_ratio=args['opt'].dropout_ratio
        reduce_ratio=args['opt'].reduce_ratio
        batch_size=args['opt'].batch_size
        max_seqlen = args['opt'].max_seqlen
        #双分支融合权重
        self.fuse_weight = args['opt'].fuse_weight


        #time_enhance分支
        self.time_Attn = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.Dropout(0.5),
                                   nn.Conv1d(512, 512, 3, padding=1),
                                   nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                   nn.Dropout(0.5),
                                   nn.Sigmoid())                
        self.time_classifier = nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.7),   
                    nn.Conv1d(embed_dim, n_class+1, 1),
                    nn.LeakyReLU(0.2),                          
                    )
        #cluster分支        
        self.cluster_Attn = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.Dropout(0.5),
                                   nn.Conv1d(512, 512, 3, padding=1),
                                   nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                   nn.Dropout(0.5),
                                   nn.Sigmoid())         
        self.cluster_classifier = nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.7),   
                    nn.Conv1d(embed_dim, n_class+1, 1),
                    nn.LeakyReLU(0.2),                          
                    )        
        #1*1、3*1、5*1三个卷积
        self.conv_1x1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, padding=0)
        self.conv_3x1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv_5x1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        # 金字塔特征融合
        self.concat_conv = nn.Sequential(nn.Conv1d(embed_dim*3, embed_dim, kernel_size=3, padding=1),
                                        nn.LeakyReLU(0.2),)        
        #时间增强分支
        self.MHSA_Intra = MHSA_Intra(dim_in=embed_dim, num_pos=max_seqlen, heads=4)        
        


        #导入整个数据集的聚类特征[21,2048]
        if self.dataset_name == 'Thumos14reduced':
            self.THUMOS_clustering = h5py.File('./cluster_features/gmm_21center_seed22THUMOS.h5','r')['feature'][:,:]
            #转换为tensor
            self.clustering = torch.from_numpy(self.THUMOS_clustering).float().cuda()               #[21,2048]
            self.register_parameter('clustering_thumos_train', nn.Parameter(torch.from_numpy(self.THUMOS_clustering).unsqueeze(0).float().cuda()))   #[1,21,2048]       
        else:
            print("using ActivityNet.h5".format("ActivityNet.h5"))
            self.ActivityNet_clustering = h5py.File('./cluster_features/ActivityNet.h5','r')['feature'][:,:]
            self.register_buffer('clustering_activitynet_train', torch.from_numpy(self.ActivityNet_clustering).float())
        #cluster encoder     
        self.cluster_transformer = GridEncoderLayer(d_model=embed_dim, d_k=64, d_v=64, h=8, d_ff=embed_dim, dropout=dropout_ratio)


        
        self.apply(weights_init)

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        b,c,n=feat.size()
        #多尺度金字塔
        embeded_feature_1x1 = self.conv_1x1(feat)
        feat = feat+embeded_feature_1x1
        embeded_feature_3x1 = self.conv_3x1(feat)
        feat = feat+embeded_feature_3x1
        embeded_feature_5x1 = self.conv_5x1(feat)
        # 特征拼接
        multiscal_feature = self.concat_conv(torch.cat([embeded_feature_1x1,embeded_feature_3x1 ,embeded_feature_5x1 ], dim=1)) # [10, 2048, 500]  

        #时间增强分支的输出
        time_feature = self.MHSA_Intra(multiscal_feature)  #(B, C, T)

        #导入聚类特征[1,21,2048]-->[b,21,2048]
        clustering = self.clustering_thumos_train.repeat(b, 1, 1)  
        #聚类分支的输出
        cluster_feature = self.cluster_transformer(feat,clustering,clustering,None)         # [1, feature_dim, temp_len]


        #time_enhance分支的attn和cas
        time_cls = self.time_classifier(time_feature)
        time_atn = self.time_Attn(time_feature)
        #cluster分支的attn和cas
        cluster_cls =  self.cluster_classifier(cluster_feature)
        cluster_atn =  self.cluster_Attn(cluster_feature)

        #权重
        weight = self.fuse_weight
        total_atn = weight*time_atn + (1-weight)*cluster_atn
        total_cls = weight*time_cls + (1-weight)*cluster_cls

        out = {#test的返回值 
              'cas':total_cls.transpose(-1, -2), 
              'attn':total_atn.transpose(-1, -2),
              #时间增强分支的返回值
              'multiscal_feat':multiscal_feature.transpose(-1, -2), #金字塔特征
              'time_feat':time_feature.transpose(-1, -2),   
              'cas_time':time_cls.transpose(-1, -2),
              'attn_time':time_atn.transpose(-1, -2),
              #cluster分支的返回值
              'cluster_feat':cluster_feature.transpose(-1, -2),
              'cas_cluster':cluster_cls.transpose(-1, -2),
              'attn_cluster':cluster_atn.transpose(-1, -2),
              }

        return out                      


    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    #聚类分支loss
    def criterion(self, outputs, labels, **args):
        #cluster的输出
        feat, element_logits,element_atn= outputs['cluster_feat'],outputs['cas_cluster'],outputs['attn_cluster']
        #multiscal的输出
        feat_multiscal = outputs['multiscal_feat']

        b,n,c = element_logits.shape
        # loss_MIL--------------------------------------------------------------1
        #多实例学习，取top-k（前景+背景抑制）
        #前景
        loss_mil_orig, _ = self.topkloss(element_logits,
                                       labels,
                                       is_back=True,
                                       rat=args['opt'].k,
                                       reduce=None)
        #背景抑制
        element_logits_supp = self._multiply(element_logits, element_atn,include_min=True)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)   

        # loss_guide ----------------------------------------------------------2
        loss_guide = (1 - element_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()   
           

        # loss_3_supp_Contrastive ------------------------------------------------3
        # 将动作和背景特征进行分离
        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)   
        
        # loss_recon -------------------------------------------------------------4
        #重建损失
        loss_recon = F.mse_loss(feat,feat_multiscal)  
        
        # total loss
        total_loss = (loss_mil_orig.mean() + loss_mil_supp.mean()       
                      + args['opt'].alpha1*loss_guide                   #    alpha1=0.5    
#                      +args['opt'].alpha2*loss_norm                    #    alpha2=1
                      + args['opt'].alpha3 * loss_3_supp_Contrastive  #    alpha3=1 
                      +1.0*loss_recon  
                      )

        return total_loss
    #时间增强分支loss    
    def time_criterion(self, outputs, labels, **args):
        feat, element_logits,element_atn= outputs['time_feat'],outputs['cas_time'],outputs['attn_time']
        b,n,c = element_logits.shape
        # loss_MIL------------------------------------------------------------------1
        #多实例学习，取top-k（前景+背景抑制）
        #前景
        loss_mil_orig, _ = self.topkloss(element_logits,
                                       labels,
                                       is_back=True,
                                       rat=args['opt'].k,
                                       reduce=None)
        #背景抑制
        element_logits_supp = self._multiply(element_logits, element_atn,include_min=True)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)   
        
        # loss_guide -------------------------------------------------------------2
        loss_guide = (1 - element_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()   
        #norm loss
#        loss_norm = element_atn.abs().mean()          

        # loss_3_supp_Contrastive ------------------------------------------------3
        # 将动作和背景特征进行分离
        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)   

        # total loss
        total_loss = (loss_mil_orig.mean() + loss_mil_supp.mean()
                      + args['opt'].alpha1*loss_guide                   #    alpha1=0.5
#                      +args['opt'].alpha2*loss_norm 
                      + args['opt'].alpha3 * loss_3_supp_Contrastive  # 背景     alpha3=1 
                      )

        return total_loss


    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):
        
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
            torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind
    
     # 相反损失函数
    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat((labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat((labels, torch.zeros_like(labels[:, [0]])), dim=-1) # 选此
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3*2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i+1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)      # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i+1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1)/n1)
            Lf2 = torch.mm(torch.transpose(x[i+1], 1, 0), (1 - atn2)/n2)

            d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
            d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
        sim_loss = sim_loss / n_tmp
        return sim_loss