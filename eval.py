import torch.utils.data.dataloader
from configs.config import Config
from dataset.dataset import ALTOEvalDataset
import torchvision.transforms as transforms
import os
import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np
import torch.utils.data
import math

class Evaler(object):
    @staticmethod
    def calEuliden(p1, p2):
        return math.sqrt( (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    @staticmethod
    def eval(model, config):
        query_csv = pd.read_csv(os.path.join(config.root, "query.csv"))
        reference_csv = pd.read_csv(os.path.join(config.root, "reference.csv"))

        query_coords = list(zip(query_csv["easting"], query_csv["northing"]))
        reference_coords = list(zip(reference_csv["easting"], reference_csv["northing"]))

        # get features
        query_dataset = torch.utils.data.DataLoader(
            dataset=ALTOEvalDataset(
                root=os.path.join(config.eval_root),
                is_query=True,
                transform=transforms.Compose([
                    transforms.Resize(config.input_dim),
                    transforms.ToTensor()
                ])
            ),
            shuffle=False,
            num_workers=config.num_workers,
            batch_size=config.test_batch_size
        )
        query_features = list()
        for query in query_dataset:
            query = query.cuda()
            feature = model(query)
            query_features.extend(feature.cpu().detach().numpy())

        import pdb
        pdb.set_trace()
        reference_dataset = torch.utils.data.DataLoader(
            dataset=ALTOEvalDataset(
                root=os.path.join(config.eval_root),
                is_query=False,
                transform=transforms.Compose([
                    transforms.Resize(config.input_dim),
                    transforms.ToTensor()
                ])
            ),
            shuffle=False,
            num_workers=config.num_workers,
            batch_size=config.test_batch_size
        )

        reference_features = list()
        for reference in reference_dataset:
            reference = reference.cuda()
            feature = model(reference)
            reference_features.extend(feature.cpu().detach().numpy())
        # release
        query_dataset = None
        reference_dataset = None

        ref_feature_KDTree = KDTree(np.array(reference_features))
        ref_coord_KDTree = KDTree(np.arange(reference_coords))

        TopN_precision = [0] * 5
        average_match_dis = 0.0

        match_csv = pd.read_csv(os.path.join(config.root, "gt_matches.csv"))
        # 记录一下每一种距离的匹配准确度
        num_of_hard_levels = len(set(list(match_csv["distance"].apply(lambda x: math.ceil(x))))) + 1
        match_success = [0 for _ in range(num_of_hard_levels)]
        match_nums = [0 for _ in range(num_of_hard_levels)]

        for query in range(len(query_csv)):
            gt_match = match_csv.iloc[query]["ref_ind"]
            predict = ref_feature_KDTree.query(query_features[query], k=5)
            for i in range(len(predict)):
                if gt_match == predict[i]:
                    TopN_precision[i] += 1
                    if i == 0:
                        match_success[int(math.ceil(match_csv.iloc[query]["distance"]))] += 1
            average_match_dis += Evaler.calEuliden(query_coords[query], reference[predict[0]])
            match_nums[int(math.ceil(match_csv.iloc[query]["distance"]))] += 1

        TopN_precision = np.cumsum(TopN_precision) / float(len(query_csv))
        average_match_dis /= len(query_csv)

        # 返回top1的准确率
        top1 = []
        
        for i in range(len(match_nums)):
            if match_nums[i] == 0:
                continue
            top1.append((match_success[i]) * 100 / match_nums[i])
            print(f"[Eval] Dis: {i}, TOP 1 Match Rate: {match_success[i] * 100 / match_nums[i]}")

        return TopN_precision, average_match_dis, top1
    
from Factorys.ModelFactory import ModelMaker
config = Config()
config.n_triplets = 200
model = ModelMaker().create(config)
Evaler(model=model, config=config)
