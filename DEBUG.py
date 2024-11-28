from dataset.dataset import ALTOdataset
from configs.config import Config

config = Config()
dataset = ALTOdataset(
    root = config.root,
    train=config.train,
    transform=config.transform,
    n_triplets=config.n_triplets,
    fliprot=config.fliprot,
    use_strategy=False,
    batch_size=config.batch_size,
    threshold=[config.threshold, config.each_level_top1]
)
data = dataset.__getitem__(0)