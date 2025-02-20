from __future__ import print_function
import numpy as np
import utils.wsad_utils as utils
import random
import os
import options
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import h5py
import torch.nn as nn


class SampleDataset:
    def __init__(self, args, mode="both",sampling='random'):
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.sampling=sampling
        self.num_segments = args.max_seqlen
        self.feature_size = args.feature_size
        self.path_to_features = os.path.join(args.path_dataset,self.dataset_name + "-I3D-JOINTFeatures.npy")
        self.path_to_annotations = os.path.join(args.path_dataset,self.dataset_name + "-Annotations/")
        self.features = np.load(
            self.path_to_features, encoding="bytes", allow_pickle=True
        )
        self.segments = np.load(
            self.path_to_annotations + "segments.npy", allow_pickle=True
        )
        self.labels = np.load(
            self.path_to_annotations + "labels_all.npy", allow_pickle=True
        )
        # Specific to Thumos14

        self._labels = np.load(
            self.path_to_annotations + "labels.npy", allow_pickle=True
        )
        self.classlist = np.load(
            self.path_to_annotations + "classlist.npy", allow_pickle=True
        )
        self.subset = np.load(
            self.path_to_annotations + "subset.npy", allow_pickle=True
        )
        self.videonames = np.load(
            self.path_to_annotations + "videoname.npy", allow_pickle=True
        )
        self.batch_size = args.batch_size
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist)
            for labs in self.labels
        ]
        try:
            ambilist = self.path_to_annotations + "/Ambiguous_test.txt"
            ambilist = list(open(ambilist, "r"))
            ambilist = [a.strip("\n").split(" ")[0] for a in ambilist]
        except:
            ambilist = []
        self.train_test_idx()
        self.classwise_feature_mapping()

        self.normalize = False
        self.mode = mode
        if mode == "rgb" or mode == "flow":
            self.feature_size = 1024

    #划分训练集和测试集
    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            
            if s.decode("utf-8") == "validation":  # Specific to Thumos14
                
                self.trainidx.append(i)
            elif s.decode("utf-8") == "test":
                self.testidx.append(i)

    #将特征映射到类别
    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode("utf-8"):
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def load_data(self, n_similar=0, is_training=True, similar_size=2):
        if is_training:
            labels = []
            idx = []

            # Load similar pairs
            if n_similar != 0:
                rand_classid = np.random.choice(
                    len(self.classwiseidx), size=n_similar
                )
                for rid in rand_classid:
                    rand_sampleid = np.random.choice(
                        len(self.classwiseidx[rid]),
                        size=similar_size,
                        replace=False,
                    )

                    for k in rand_sampleid:
                        idx.append(self.classwiseidx[rid][k])

            # Load rest pairs
            if self.batch_size - similar_size * n_similar < 0:
                self.batch_size = similar_size * n_similar

            rand_sampleid = np.random.choice(
                len(self.trainidx),
#batch_size
#                size=self.batch_size - similar_size * n_similar,
#all data
                size=len(self.trainidx),
            )

            for r in rand_sampleid:
                idx.append(self.trainidx[r])
            feat = []
            for i in idx:
                ifeat = self.features[i]
                if self.sampling == 'random':
                    sample_idx = self.random_perturb(ifeat.shape[0])
                elif self.sampling == 'uniform':
                    sample_idx = self.uniform_sampling(ifeat.shape[0])
                elif self.sampling == "all":
                    sample_idx = np.arange(ifeat.shape[0])
                else:
                    raise AssertionError('Not supported sampling !')
                ifeat = ifeat[sample_idx]
                feat.append(ifeat)
            feat = np.array(feat)
            labels = np.array([self.labels_multihot[i] for i in idx])
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size :]

            return feat, labels,rand_sampleid

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]
            # feat = utils.process_feat(feat, normalize=self.normalize)
            # feature = feature[sample_idx]
            #从测试视频列表中获取当前测试视频的名称
            vn = self.videonames[self.testidx[self.currenttestidx]]
            if self.currenttestidx == len(self.testidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1
            feat = np.array(feat)
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size :]
            return feat, np.array(labs),vn, done
    def random_avg(self, x, segm=None):
        if len(x) < self.num_segments:
            ind = self.random_perturb(len(x))
            x_n = x[ind]
            segm = segm[ind] if segm is not None else None
            return x_n, segm
        else:
            inds = np.array_split(np.arange(len(x)), self.num_segments)
            x_n = np.zeros((self.num_segments, x.shape[-1])).astype(x.dtype)
            segm_n = np.zeros(
                (self.num_segments, segm.shape[-1])).astype(x.dtype)
            for i, ind in enumerate(inds):
                x_n[i] = np.mean(x[ind], axis=0)
                if segm is not None:
                    segm_n[i] = segm[(ind[0] + ind[-1]) // 2]
            return x_n, segm_n if segm is not None else None

    def random_pad(self, x, segm=None):
        length = self.num_segments
        if x.shape[0] > length:
            strt = np.random.randint(0, x.shape[0] - length)
            x_ret = x[strt:strt + length]
            if segm is not None:
                segm = segm[strt:strt + length]
                return x_ret, segm
        elif x.shape[0] == length:
            return x, segm
        else:
            pad_len = length - x.shape[0]
            x_ret = np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
            if segm is not None:
                segm = np.pad(segm, ((0, pad_len), (0, 0)), mode='constant')
            return x_ret, segm

    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(
                        range(int(samples[i]),
                              int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(
                        range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)


# if __name__ == '__main__':
#     args = options.parser.parse_args()
#     dt = SampleDataset(args)
#     data = dt.load_data()
#     print(data)
#     import pdb
#     pdb.set_trace()
#     print(dt)

#读取数据
args = options.parser.parse_args()
dt = SampleDataset(args)
data = dt.load_data()
features = data[0]
features_array = np.array(features)



#gmm高斯聚类
#print("feat: ", features)
center_number = 1000
seed = 24
gmm = GaussianMixture(n_components=center_number, random_state=seed, covariance_type='full')

gmm.fit(features_array.reshape(-1, 2048))
print(gmm.means_.shape)
#保存聚类结果
path_center_number = str(center_number)
path_seed = str(seed)
result_path = "new_cluster_features/" +"gmm_"+path_center_number+"center_seed"+path_seed+ args['opt'].dataset_name +".h5"
if not os.path.exists(result_path):
    with h5py.File(result_path, 'w') as f:
        f.create_dataset('feature', data=gmm.means_)
        f.close()
        print("保存成功！")

#         # predictions = gmm.predict(features)
# for itr in tqdm(range(args.max_iter)):    
#     # 加载数据
#     features, labels, pairs_id= SampleDataset.load_data(n_similar=args.num_similar)
#     #seq_len是每个视频的长度
#     seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
#     #
#     features = features[:,:np.max(seq_len),:]
    
#     features = torch.from_numpy(features).float().to(device) 
#     #cat整个数据集
#     all_features = torch.cat((all_features,features),dim=0)
#         # labels = data[1]
#         # rand_sampleids = data[2]

#         # 创建GMM模型并拟合数据
#         # gmm = GaussianMixture(n_components=21)
#         # gmm.fit(features)
#         # predictions = gmm.predict(features)

#         # # 计算混淆矩阵
#         # conf_mat = confusion_matrix(labels, predictions)
#         # print("Confusion Matrix:\n", conf_mat)

#         # # 计算准确率
#         # accuracy = np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat)
#         # print("Accuracy: ", accuracy)
