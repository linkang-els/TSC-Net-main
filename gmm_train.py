import os 
import json
import torch
from sklearn.cluster import KMeans
import numpy as np
import h5py
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from torch.utils.data import Dataset
from sklearn.cluster import SpectralClustering
import options 
import utils.wsad_utils as utils
from scipy import interpolate


class THUMOS14Feature:
    def __init__(self, args):
        self.args = args
        self.data_dir = args.data_dir
        self.phase = "train"
        # self.data_dir = feature_path
        # 数据集
        self.dataset = args.dataset
        # print(args.action_cls)

        with open(os.path.join(self.data_dir, "gt.json")) as gt_f:
            self.gt_dict = json.load(gt_f)["database"]  # 读取gt.json文件
        gt_dict = self.gt_dict
        self.feature_dir = os.path.join(self.data_dir, "train")  # 特征路径
        self.data_list = list(open(os.path.join(self.data_dir, "split_train.txt")))  # 数据列表
        self.data_list = [item.strip() for item in self.data_list]  # 去除空格

        self.class_name_lst = args.class_name_lst  # 类名列表

        self.action_class_idx_dict = {action_cls: idx for idx, action_cls in enumerate(self.class_name_lst)}  # 动作类别索引字典
        self.action_class_num = args.action_cls_num  # 动作类别数

    def __len__(self):
        return len(self.data_list)  # 返回数据列表的长度

    def getitem(self, idx):  # 获取数据
        self.label_dict = {}
        cat_feature = []
        for vid_name in self.data_list:  # 对于数据列表中的每个视频
            item_label = np.zeros(self.action_class_num)  # 初始化标签
            for ann in self.gt_dict[vid_name]["annotations"]:  # 对于每个注释
                ann_label = ann["label"]
                item_label[self.action_class_idx_dict[ann_label]] = 1.0  # 标签为1
        self.label_dict[vid_name] = item_label  # 将标签放入标签字典中
        vid_name = self.data_list[idx]
        # print("vid_name:",vid_name)
        # vid_label = self.label_dict[vid_name]
        if 'full' in self.phase:
            if self.dataset == 'THUMOS':
                if 'validation' in vid_name:
                    con_vid_feature = np.load(os.path.join(self.feature_dir, 'train', vid_name + ".npy"))
                else:
                    con_vid_feature = np.load(os.path.join(self.feature_dir, 'test', vid_name + ".npy"))
            elif self.dataset == 'ActivityNet':
                if os.path.isfile(os.path.join(self.feature_dir, 'train', vid_name + ".npy")):
                    con_vid_feature = np.load(os.path.join(self.feature_dir, 'train', vid_name + ".npy"))
                elif os.path.isfile(os.path.join(self.feature_dir, 'test', vid_name + ".npy")):
                    con_vid_feature = np.load(os.path.join(self.feature_dir, 'test', vid_name + ".npy"))
        else:
            con_vid_feature = np.load(os.path.join(self.feature_dir, vid_name + ".npy"))
        con_vid_feature = torch.from_numpy(con_vid_feature).float()
        vid_len = con_vid_feature.shape[0]  # 视频长度
        # print("vid_len:",vid_len)
        for k in range(vid_len):
            cat_feature.append(con_vid_feature[k].unsqueeze(0).cpu())

        train_feature = torch.cat(cat_feature, dim=0)
        print("train_feature:", train_feature.shape)
        # print("con_vid_feature:",con_vid_feature)

        return train_feature

class AntSampleDataset:
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
        self.t_max = args.max_seqlen
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
        self.filter()
    def filter(self):
        new_testidx = []
        for idx in self.testidx:
            feat = self.features[idx]
            if len(feat)>10:
                new_testidx.append(idx)
        self.testidx = new_testidx

        new_trainidx = []
        for idx in self.trainidx:
            feat = self.features[idx]
            if len(feat)>10:
                new_trainidx.append(idx)
        self.trainidx = new_trainidx
    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode("utf-8") == "training":  # Specific to Thumos14
                self.trainidx.append(i)
            elif s.decode("utf-8") == "validation":
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                if self.features[i].sum() == 0:
                    continue
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
                size=self.batch_size - similar_size * n_similar,
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

        # else:
        #     labs = self.labels_multihot[self.testidx[self.currenttestidx]]
        #     feat = self.features[self.testidx[self.currenttestidx]]
        #     # feat = utils.process_feat(feat, normalize=self.normalize)
        #     # feature = feature[sample_idx]
        #     vn = self.videonames[self.testidx[self.currenttestidx]]
        #     if self.currenttestidx == len(self.testidx) - 1:
        #         done = True
        #         self.currenttestidx = 0
        #     else:
        #         done = False
        #         self.currenttestidx += 1
        #     feat = np.array(feat)
        #     if self.mode == "rgb":
        #         feat = feat[..., : self.feature_size]
        #     elif self.mode == "flow":
        #         feat = feat[..., self.feature_size :]
        #     return feat, np.array(labs),vn, done
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

    def temporal_interpolation(self, temp_data, output_length):
        temp_len = temp_data.shape[0]
        temp_x = np.arange(temp_len)
        # 检查 temp_data 和 temp_x 的长度
        if temp_len < 2:
            temp_data = np.repeat(temp_data, 2, axis=0)
            temp_x = np.arange(2)
        f = interpolate.interp1d(temp_x, temp_data, axis=0, kind="linear", fill_value="extrapolate")
        int_temp_x = np.linspace(0, temp_len - 1, output_length)
        temp_data = f(int_temp_x)
        return temp_data

    def random_sample(self, input_feature, sample_len):
        input_len = input_feature.shape[0]
        assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(sample_len)

        if input_len < sample_len:
            return self.temporal_interpolation(input_feature, sample_len)
        elif input_len > sample_len:
            index_list = np.rint(np.linspace(0, input_len - 1, sample_len + 1)).astype(np.int)
            sample_idxs = np.zeros(sample_len)
            for i in range(sample_len):
                sample_idxs[i] = np.random.choice(range(index_list[i], index_list[i + 1]))
        else:
            sample_idxs = np.arange(input_len)
        return input_feature[sample_idxs.astype(np.int), :]

    def get_all_train_features(self, sample_len):
        all_train_features = []
        for idx in range(len(self.features)):
            train_features = np.array(self.features[idx])
            sampled_features = self.random_sample(train_features, sample_len)
            all_train_features.append(sampled_features)
        all_train_features = np.vstack(all_train_features)
        return all_train_features






args = options.parser.parse_args()
dataset_list = []     #数据集列表
feature_path = args.path_dataset   #特征路径
dataset = args.dataset_name      #数据集
sample_len = 60        #随机采样长度，TH14：320,ANT：60

#args = build_args(dataset="THUMOS")
#如果是THUMOS数据集
if args.dataset_name == "Thumos14reduced":
    sample_len = 320        #随机采样长度，TH14：320,ANT：60
    train_features = THUMOS14Feature(args).getitem(0)  #[265,2048] 
else:
    sample_len = 20        #随机采样长度，TH14：320,ANT：60
    Dataset = AntSampleDataset(args)
    # 调用load_data方法
    train_features= Dataset.get_all_train_features(sample_len)

         
###############使用KMeans聚类#################
# print("Dataset:",Dataset.shape)
# print("Begin KMeans...")
# KMeans = MiniBatchKMeans(n_clusters=20, random_state=0, batch_size=1024).fit(Dataset.view(-1,2048))
# print(KMeans.cluster_centers_.shape)
# result_path = "/home/lk/lk/ASM-Loc-main/cluster_features/" + dataset +".h5"
# if not os.path.exists(result_path):
#     with h5py.File(result_path, 'w') as f:
#         f.create_dataset('feature', data=KMeans.cluster_centers_)
#         f.close()
#         print("保存成功！")



###############使用GMM聚类#################
print("Begin GMM...")
gmm = GaussianMixture(n_components=101, random_state=0, covariance_type='full')  #tied,full,diag,spherical
gmm.fit(train_features)
print(gmm.means_.shape)
result_path = "/home/lk/lk/TFE-DCN-main/cluster_features/" +"gmm_101center_seed20"+ dataset +".h5"
if not os.path.exists(result_path):
    with h5py.File(result_path, 'w') as f:
        f.create_dataset('feature', data=gmm.means_)
        f.close()
        print("保存成功！")

###############使用谱聚类（Spectral Clustering）#################
# # 使用谱聚类（Spectral Clustering）
# print("Begin Spectral Clustering...")
# spectral_clustering = SpectralClustering(n_clusters=21, affinity='nearest_neighbors', random_state=0)
# labels = spectral_clustering.fit_predict(Dataset.view(-1, 2048))
# print(labels.shape)

# # 将聚类结果转换为（21,2048）维度
# result = labels.reshape((21, 2048))

# result_path = "/home/lk/lk/asm-2/ASM-Loc-main/cluster_features/" + "spectral_21center" + dataset + ".h5"
# if not os.path.exists(result_path):
#     with h5py.File(result_path, 'w') as f:
#         f.create_dataset('result', data=result)
#         f.close()
#         print("保存成功！")

