import torch.utils.data.dataloader
from dataset.dataset import ALTOdataset
import torch

def Log(msg):
    print(f"[DataSet] {msg}")

class DataMaker(object):
    def __init__(self, config):
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
                threshold=[config.threshold, config.topN_precision]
            )
            return torch.utils.data.dataloader(
                dataset,
                config.batch_size,
                shuffle=False,
                **config.kwargs
            )
        else:
            assert f"datset name {config.dataset_name} is not excepted."