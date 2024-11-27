import pandas as pd
import os
import math
import random
import numpy as np
import torch
from PIL import Image
from copy import deepcopy
from torch.utils.data import Dataset
from configs.config import Config

class ALTOdataset(Dataset):
    def __init__(self, root, train=True, transform=None, n_triplets = None, batch_size = None, fliprot=True, use_strategy = False,threshold = [5.0, [10 for _ in range(7)]],*args, **kw):
        super(ALTOdataset, self).__init__(*args, **kw)
        self.root=root
        self.transform = transform
        self.fliprot=fliprot
        self.train = train
        self.n_triplets = n_triplets
        self.batch_size = batch_size
        self.use_strategy = use_strategy
        self.triplets = list()
        self.threshold = threshold[0]
        top1 = threshold[1]
        for i in range(len(top1)):
            if top1[i] == 0:
                top1[i] += 1
        for i in range(int(math.ceil(self.threshold)), len(top1)):
            top1[i] = 0
        # 求各等级样本分配比例
        self.top1 = top1
        self.samples_level_weight = (np.array(top1) / np.array(top1).sum()).tolist()

        # create triplets
        if self.train:
            self.make_triplets()
        else:
            self.load_test()

    def make_triplets(self,):
        # load data info from csv file
        self.match_csv = pd.read_csv(os.path.join(self.root, "gt_matches.csv"))
        self.query_csv = pd.read_csv(os.path.join(self.root, "query.csv"))
        self.reference_csv = pd.read_csv(os.path.join(self.root, "reference.csv"))

        # sample n triplets
        if not self.use_strategy:
            alpha = math.ceil(self.n_triplets / len(self.match_csv))
            # select the negative based on the reference
            negatives = list()
            for p in self.match_csv["ref_ind"].tolist() * alpha:
                # select the p 10 out of it
                left = p - 10 if p - 10 > 0 else 0
                right = p + 5 if p + 10 < len(self.match_csv) else len(self.match_csv)
                negatives.append(random.sample(list(range(0, left)) + list(range(right, len(self.match_csv))), k=1)[0])

            # make n anchor - positive
            self.triplets = list(zip(self.match_csv["query_ind"].tolist() * alpha, self.match_csv["ref_ind"].tolist() *alpha, negatives))[:self.n_triplets]
        else:
            # cal each level triplet nums
            total_nums = int(self.n_triplets / self.batch_size) * self.batch_size
            nums_of_different_level = np.ceil(np.array(self.samples_level_weight) * total_nums)
            if not nums_of_different_level.sum() == total_nums:
                nums_of_different_level[np.argmax(nums_of_different_level)] -= nums_of_different_level.sum() - total_nums

            # 记录所有采样的数据
            all_sample_data = [[], [], []]
            for hard_level, sample_nums in enumerate(nums_of_different_level):
                # 按照难度载入数据
                filter_columns = self.match_csv["distance"] <= hard_level
                self.match_imgs = list(zip(self.match_csv[filter_columns]["query_ind"], self.match_csv[filter_columns]["ref_ind"]))
                # given match indices return a list of match querys
                if len(self.match_imgs) == 0:
                    continue

                indices = dict()
                for p in list(set(self.match_csv["ref_ind"])):
                    # find out all possiable anchor
                    indices[p] = list(self.match_csv[self.match_csv["ref_ind"] == p]["query_ind"])

                # n_classes是有多少张match imgs在数据里面
                n_classes = len(indices)

                # 筛掉没有query的match img
                skip_list = []
                for id in range(n_classes):
                    if len(indices[id]) == 0:
                        skip_list.append(id)

                counter = 0
                while not counter >= sample_nums:
                    # 采样一个match, means the Satellite image
                    select_match = random.randint(0, n_classes - 1)
                    # 确保match合法
                    if select_match in skip_list:
                        continue
                    # 确定要采样多少个anchor(query)
                    max_sampler_num = len(indices[select_match])
                    num_of_anchor = max_sampler_num if sample_nums - counter >= max_sampler_num else sample_nums - counter
                    num_of_anchor = int(num_of_anchor)                
                    # 采样anchor和negative
                    anchors = random.sample(indices[select_match], num_of_anchor)
                    
                    # lateset version, context-aware method
                    negatives = list()
                    for _ in anchors:
                        # 学的越好的，remote negative应该要越小
                        negative_length = self.cosine_decay(int(self.top1[hard_level]), 10.0, 5.0, 0, 100)
                        negative_length = negative_length if negative_length > 6 else 6
                        negative_lower_bound = select_match - negative_length if select_match - negative_length >= 0 else 0
                        negative_lower_bound = int(negative_lower_bound)
                        negative_higher_bound = select_match + negative_length if select_match + negative_length < len(self.reference_csv) else len(self.reference_csv)
                        negative_higher_bound = int(negative_higher_bound)
                        negatives.append(np.random.choice(list(range(negative_lower_bound, select_match)) + list(range(select_match + 1, negative_higher_bound)), 1).tolist()[0])
                    
                    all_sample_data[0] += anchors
                    all_sample_data[1] += [select_match for _ in range(num_of_anchor)]
                    all_sample_data[2] += negatives
                    
                    counter += num_of_anchor

            # 组合数据，打乱，分batch
            all_sample_data = list(zip(all_sample_data[0], all_sample_data[1], all_sample_data[2]))
            random.shuffle(all_sample_data)
            head = 0
            tail = 0 + self.batch_size
            while not tail > len(all_sample_data):
                batch_data = all_sample_data[head: tail]
                head += self.batch_size
                tail += self.batch_size
                self.triplets += batch_data


    def load_test(self, is_query):
        pass

    def __getitem__(self, index):
        if self.train:
            # get images
            a, p, n = self.triplets[index]
            a = self.transform(Image.open(os.path.join(self.root, "query_images", self.query_csv.iloc[a]["name"])).convert("RGB"))
            p = self.transform(Image.open(os.path.join(self.root, "reference_images", self.reference_csv.iloc[p]["name"])).convert("RGB"))
            n = self.transform(Image.open(os.path.join(self.root, "reference_images", self.reference_csv.iloc[n]["name"])).convert("RGB"))
            
            if self.fliprot:
                do_flip = random.random() > 0.5
                do_rot = random.random() > 0.5
                if do_rot:
                    a = a.permute(0,2,1)
                    p = p.permute(0,2,1)
                    n = n.permute(0,2,1)
                if do_flip:
                    a = torch.from_numpy(deepcopy(a.numpy()[:,:,::-1]))
                    p = torch.from_numpy(deepcopy(p.numpy()[:,:,::-1]))
                    n = torch.from_numpy(deepcopy(n.numpy()[:,:,::-1]))

            return (a, p, n)

    def __len__(self):
        return len(self.triplets)

    def cosine_decay(self, epoch, start_value, end_value, start_epoch, end_epoch):
        if epoch < start_epoch:
            return start_value
        elif epoch > end_epoch:
            return end_value
        else:
            progress = (epoch - start_epoch) / (end_epoch - start_epoch)
            cosine_decay_value = 0.5 * (1 + math.cos(math.pi * progress))
            return end_value + (start_value - end_value) * cosine_decay_value
        
class ALTOEvalDataset(Dataset):
    def __init__(self, root, transform, is_query) -> None:
        super().__init__()
        self.root = root
        self.is_query = is_query
        self.transform = transform
        if is_query:
            self.data_csv = pd.read_csv(os.path.join(root, "query.csv"))
        else:
            self.data_csv = pd.read_csv(os.path.join(root, "reference.csv"))

    def __getitem__(self, index):
        if self.is_query:
            img = self.transform(Image.open(os.path.join(self.root, "query_images", self.data_csv.iloc[index]["name"])).convert("RGB"))
        else:
            img = self.transform(Image.open(os.path.join(self.root, "reference_images", self.data_csv.iloc[index]["name"])).convert("RGB"))
        return img

    def __len__(self):
        return len(self.data_csv)
    
class NYFdataset(Dataset):
    def __init__(self, root, train=True, transform=None, n_triplets = None, batch_size = None, fliprot=True, use_strategy = False,threshold = [5.0, [10 for _ in range(7)]],*args, **kw):
        super(NYFdataset, self).__init__(*args, **kw)
        self.root=root
        self.transform = transform
        self.fliprot=fliprot
        self.train = train
        self.n_triplets = n_triplets
        self.batch_size = batch_size
        self.use_strategy = use_strategy
        self.triplets = list()
        self.threshold = threshold[0]
        top1 = threshold[1]
        for i in range(len(top1)):
            if top1[i] == 0:
                top1[i] += 1
        for i in range(int(math.ceil(self.threshold)), len(top1)):
            top1[i] = 0
        # 求各等级样本分配比例
        self.top1 = top1
        self.samples_level_weight = (np.array(top1) / np.array(top1).sum()).tolist()

        # create triplets
        if self.train:
            self.make_triplets()
        else:
            self.load_test()

    def make_triplets(self,):
        # load data info from csv file
        self.match_csv = pd.read_csv(os.path.join(self.root, "gt_matches.csv"))
        self.query_csv = pd.read_csv(os.path.join(self.root, "query.csv"))
        self.reference_csv = pd.read_csv(os.path.join(self.root, "reference.csv"))

        # sample n triplets
        if not self.use_strategy:
            alpha = math.ceil(self.n_triplets / len(self.match_csv))
            # select the negative based on the reference
            negatives = list()
            for p in self.match_csv["ref_ind"].tolist() * alpha:
                # select the p 10 out of it
                left = p - 10 if p - 10 > 0 else 0
                right = p + 5 if p + 10 < len(self.match_csv) else len(self.match_csv)
                negatives.append(random.sample(list(range(0, left)) + list(range(right, len(self.match_csv))), k=1)[0])

            # make n anchor - positive
            self.triplets = list(zip(self.match_csv["query_ind"].tolist() * alpha, self.match_csv["ref_ind"].tolist() *alpha, negatives))[:self.n_triplets]
        else:
            # cal each level triplet nums
            total_nums = int(self.n_triplets / self.batch_size) * self.batch_size
            nums_of_different_level = np.ceil(np.array(self.samples_level_weight) * total_nums)
            if not nums_of_different_level.sum() == total_nums:
                nums_of_different_level[np.argmax(nums_of_different_level)] -= nums_of_different_level.sum() - total_nums

            # 记录所有采样的数据
            all_sample_data = [[], [], []]
            for hard_level, sample_nums in enumerate(nums_of_different_level):
                # 按照难度载入数据
                filter_columns = self.match_csv["distance"] <= hard_level
                self.match_imgs = list(zip(self.match_csv[filter_columns]["query_ind"], self.match_csv[filter_columns]["ref_ind"]))
                # given match indices return a list of match querys
                if len(self.match_imgs) == 0:
                    continue

                indices = dict()
                for p in list(set(self.match_csv["ref_ind"])):
                    # find out all possiable anchor
                    indices[p] = list(self.match_csv[self.match_csv["ref_ind"] == p]["query_ind"])

                # n_classes是有多少张match imgs在数据里面
                n_classes = len(indices)

                # 筛掉没有query的match img
                skip_list = []
                for id in range(n_classes):
                    if len(indices[id]) == 0:
                        skip_list.append(id)

                counter = 0
                while not counter >= sample_nums:
                    # 采样一个match, means the Satellite image
                    select_match = random.randint(0, n_classes - 1)
                    # 确保match合法
                    if select_match in skip_list:
                        continue
                    # 确定要采样多少个anchor(query)
                    max_sampler_num = len(indices[select_match])
                    num_of_anchor = max_sampler_num if sample_nums - counter >= max_sampler_num else sample_nums - counter
                    num_of_anchor = int(num_of_anchor)                
                    # 采样anchor和negative
                    anchors = random.sample(indices[select_match], num_of_anchor)
                    
                    # lateset version, context-aware method
                    negatives = list()
                    for _ in anchors:
                        # 学的越好的，remote negative应该要越小
                        negative_length = self.cosine_decay(int(self.top1[hard_level]), 10.0, 5.0, 0, 100)
                        negative_length = negative_length if negative_length > 6 else 6
                        negative_lower_bound = select_match - negative_length if select_match - negative_length >= 0 else 0
                        negative_lower_bound = int(negative_lower_bound)
                        negative_higher_bound = select_match + negative_length if select_match + negative_length < len(self.reference_csv) else len(self.reference_csv)
                        negative_higher_bound = int(negative_higher_bound)
                        negatives.append(np.random.choice(list(range(negative_lower_bound, select_match)) + list(range(select_match + 1, negative_higher_bound)), 1).tolist()[0])
                    
                    all_sample_data[0] += anchors
                    all_sample_data[1] += [select_match for _ in range(num_of_anchor)]
                    all_sample_data[2] += negatives
                    
                    counter += num_of_anchor

            # 组合数据，打乱，分batch
            all_sample_data = list(zip(all_sample_data[0], all_sample_data[1], all_sample_data[2]))
            random.shuffle(all_sample_data)
            head = 0
            tail = 0 + self.batch_size
            while not tail > len(all_sample_data):
                batch_data = all_sample_data[head: tail]
                head += self.batch_size
                tail += self.batch_size
                self.triplets += batch_data


    def load_test(self, is_query):
        pass

    def __getitem__(self, index):
        if self.train:
            # get images
            a, p, n = self.triplets[index]
            a = self.transform(Image.open(os.path.join(self.root, "query_images", self.query_csv.iloc[a]["name"])).convert("RGB"))
            p = self.transform(Image.open(os.path.join(self.root, "reference_images", self.reference_csv.iloc[p]["name"])).convert("RGB"))
            n = self.transform(Image.open(os.path.join(self.root, "reference_images", self.reference_csv.iloc[n]["name"])).convert("RGB"))
            
            if self.fliprot:
                do_flip = random.random() > 0.5
                do_rot = random.random() > 0.5
                if do_rot:
                    a = a.permute(0,2,1)
                    p = p.permute(0,2,1)
                    n = n.permute(0,2,1)
                if do_flip:
                    a = torch.from_numpy(deepcopy(a.numpy()[:,:,::-1]))
                    p = torch.from_numpy(deepcopy(p.numpy()[:,:,::-1]))
                    n = torch.from_numpy(deepcopy(n.numpy()[:,:,::-1]))

            return (a, p, n)

    def __len__(self):
        return len(self.triplets)

    def cosine_decay(self, epoch, start_value, end_value, start_epoch, end_epoch):
        if epoch < start_epoch:
            return start_value
        elif epoch > end_epoch:
            return end_value
        else:
            progress = (epoch - start_epoch) / (end_epoch - start_epoch)
            cosine_decay_value = 0.5 * (1 + math.cos(math.pi * progress))
            return end_value + (start_value - end_value) * cosine_decay_value