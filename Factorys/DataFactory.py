import torch.utils.data.dataloader
from dataset.dataset import ALTOdataset, NYFdataset
import torch

def Log(msg):
    print(f"[DataSet] {msg}")

class DataMaker(object):
    @staticmethod
    def create(config):
        Log("Creating Dataset")
        if config.dataset_name == "ALTO":
            dataset = ALTOdataset(
                root=config.root,
                transform=config.transform,
                fliprot=config.fliprot,
                train=config.train,
                batch_size=config.batch_size,
                n_triplets=config.n_triplets,
                use_strategy=config.use_strategy,
                threshold=[config.threshold, config.each_level_top1]
            )
            return torch.utils.data.DataLoader(
                dataset,
                config.batch_size,
                shuffle=False,
                **config.kwargs
            )
        elif config.dataset_name == "NYF":
            dataset = NYFdataset(
                root=config.root,
                transform=config.transform,
                fliprot=config.fliprot,
                train=config.train,
                batch_size=config.batch_size,
                n_triplets=config.n_triplets,
                use_strategy=config.use_strategy,
                threshold=[config.threshold, config.each_level_top1]
            )
            return torch.utils.data.DataLoader(
                dataset,
                config.batch_size,
                shuffle=False,
                **config.kwargs
            )
        else:
            assert f"datset name {config.dataset_name} is not excepted."