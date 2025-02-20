import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_logger import log_value
import utils.wsad_utils_v2 as utils
import numpy as np
from torch.autograd import Variable
from eval.classificationMAP import getClassificationMAP as cmAP
from eval.eval_detection import ANETdetection
import wsad_dataset
from eval.detectionMAP import getDetectionMAP as dmAP
import scipy.io as sio
from tensorboard_logger import Logger
import multiprocessing as mp
import options
#import model
import models.v1_ms_asm9_h4_cluster_v5_4_w10.model_baselinev3_ms_asm9_h4_cluster_v5_4_w10 as model
import proposal_methods as PM
import pandas as pd
from collections import defaultdict
import os
#from visualize_pictures import visualize_attention, visualize_cas

#test文件，v3添加了可视化双分支attention和cas的代码
#输入：
#features：特征，labels：标签，vn：视频名，done：是否结束

torch.set_default_tensor_type('torch.cuda.FloatTensor')
@torch.no_grad()
def test(itr, dataset, args, model, logger, device,pool):
    model.eval()
    done = False
    instance_logits_stack = []
    element_logits_stack = []
    labels_stack = []

    back_norms=[]
    front_norms=[]
    ind=0
    proposals = []
    results = defaultdict(dict)
    logits_dict = defaultdict(dict)
    while not done:
        if dataset.currenttestidx % (len(dataset.testidx)//5) ==0:
            print('Testing test data point %d of %d' %(dataset.currenttestidx, len(dataset.testidx)))

        features, labels,vn, done = dataset.load_data(is_training=False)

        seq_len = [features.shape[0]]
        if seq_len == 0:
            continue
        features = torch.from_numpy(features).float().to(device).unsqueeze(0)
        #禁止梯度计算
        with torch.no_grad():
            #调用model的forward函数
            outputs = model(Variable(features), is_training=False,seq_len=seq_len)
            #获取输出的cas和attn
            element_logits = outputs['cas']            
            attn = outputs['attn']
            #muitiscale 分支的cas和attn
            cas_ms = outputs['cas_time']
            attn_ms = outputs['attn_time']
            #cluster 分支的cas和attn
            cas_cluster = outputs['cas_cluster']
            attn_cluster = outputs['attn_cluster']
            #获取视频名
            vnd = vn.decode()
            #可视化注意力权重
            #visualize_attention(attn, attn_ms,attn_cluster, vnd)
            #可视化cas
            #visualize_cas(element_logits, cas_ms,cas_cluster, vnd)
            #将输出的cas和attn保存到results中
            results[vn] = {'cas':outputs['cas'],'attn':outputs['attn']}
            '''
            prediction:
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
                "score": score_lst,
            '''
            proposals.append(getattr(PM, args.proposal_method)(vn,outputs))
            logits=element_logits.squeeze(0)
        #计算分类的topk
        tmp = F.softmax(torch.mean(torch.topk(logits, k=int(np.ceil(len(features)/8)), dim=0)[0], dim=0), dim=0).cpu().data.numpy()
        
        instance_logits_stack.append(tmp)
        labels_stack.append(labels)

    if not os.path.exists('temp'):
        os.mkdir('temp')
    np.save('temp/{}.npy'.format(args.model_name),results)

    instance_logits_stack = np.array(instance_logits_stack)
    labels_stack = np.array(labels_stack)
    #proposals是预测的定位结果
    proposals = pd.concat(proposals).reset_index(drop=True)

    #CVPR2020
    if 'Thumos14' in args.dataset_name:
        iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args)
        dmap_detect.prediction = proposals
        dmap = dmap_detect.evaluate()
    else:
        iou = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95]

        dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args,subset='validation')
        dmap_detect.prediction = proposals
        dmap = dmap_detect.evaluate()

    if args.dataset_name == 'Thumos14':
        test_set = sio.loadmat('test_set_meta.mat')['test_videos'][0]
        for i in range(np.shape(labels_stack)[0]):
            if test_set[i]['background_video'] == 'YES':
                labels_stack[i,:] = np.zeros_like(labels_stack[i,:])

    cmap = cmAP(instance_logits_stack, labels_stack)
    print('Classification map %f' %cmap)
    print('||'.join(['map @ {} = {:.3f} '.format(iou[i],dmap[i]*100) for i in range(len(iou))]))
    print('mAP Avg ALL: {:.3f}'.format(sum(dmap)/len(iou)*100))
    
    logger.log_value('Test Classification mAP', cmap, itr)
    for item in list(zip(dmap,iou)):
        logger.log_value('Test Detection mAP @ IoU = ' + str(item[1]), item[0], itr)
    #计算mAP Avg ALL、mAP Avg 0.1-0.5、mAP Avg 0.1-0.7
    mAP_avg_all = sum(dmap) / len(iou) * 100
    mAP_avg_0_1_0_5 = sum(dmap[:5]) / 5 * 100
    mAP_avg_0_1_0_7 = sum(dmap[:7]) / 7 * 100
    logger.log_value('Test Detection mAP Avg ALL', sum(dmap)/len(iou), itr)
    logger.log_value('Test Detection mAP Avg 0.1-0.5', np.mean(dmap[:5]), itr)
    logger.log_value('Test Detection mAP Avg 0.1-0.7', np.mean(dmap[:7]), itr)
    # 将所有值添加到要写入文件的元组中
    values_to_write = (dmap, cmap, itr, mAP_avg_all, mAP_avg_0_1_0_5, mAP_avg_0_1_0_7)

    # 创建 logs 文件夹和 model_name 子文件夹（如果尚不存在）
    if not os.path.exists('logs'):
        os.mkdir('logs')
    model_folder = os.path.join('logs', args.model_name)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    # 创建 dataset_name 子文件夹（如果尚不存在）
    dataset_folder = os.path.join(model_folder, args.dataset_name)
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)

    # 将元组中的值写入日志文件
    utils.write_to_file(os.path.join(dataset_folder, 'log.txt'), dmap, cmap, itr, mAP_avg_all, mAP_avg_0_1_0_5, mAP_avg_0_1_0_7)
    return iou,dmap

if __name__ == '__main__':
    args = options.parser.parse_args()
    device = torch.device("cuda")
    dataset = getattr(wsad_dataset,args.dataset)(args)

    model = getattr(model,args.use_model)(dataset.feature_size, dataset.num_class,opt=args).to(device)
    model.load_state_dict(torch.load('./ckpt/best_' + args.model_name + '.pkl'))
    logger = Logger('./logs/test_' + args.model_name)
    pool = mp.Pool(5)

    iou,dmap = test(-1, dataset, args, model, logger, device,pool)
    print('mAP Avg 0.1-0.5: {}, mAP Avg 0.1-0.7: {}, mAP Avg ALL: {}'.format(np.mean(dmap[:5])*100,np.mean(dmap[:7])*100,np.mean(dmap)*100))

    
