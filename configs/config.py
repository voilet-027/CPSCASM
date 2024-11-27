import torchvision.transforms as transforms
import PIL
import os
import shutil

class Config(object):
    def __init__(self):
        # 实验名称
        self.experiment_name = "DEBUG"
        # 模型权重存放路径
        self.save_path = f"/home/dzh/experiment/{self.experiment_name}"
        # 权重保存模式
        self.save_best = True
        # 配置文件全局路径
        self.config_file_path = "/home/dzh/CASM/configs/config.py"
        # 时间种子
        self.seed = 0

        # 定义模型信息
        self.model_name = "vit"
        self.input_dim = 128
        self.output_dim = 256
        self.is_SPT = False
        self.is_LSA = False
        self.device = 'cuda:0'
        self.checkpoint = None

        # 定义数据集的信息
        self.dataset_name = "ALTO"
        self.root = "/home/datasets/ALTO/Train"
        self.transform = transforms.Compose([
            transforms.RandomRotation(30,PIL.Image.BILINEAR, expand=True),
            transforms.Resize(self.input_dim),
            transforms.ToTensor()])
        self.train = True
        self.n_triplets = 20000
        self.batch_size = 48
        self.fliprot = True
        self.use_strategy = True
        self.threshold = 5
        self.each_level_top1 = [10 for _ in range(7)]
        self.num_workers = 10
        self.kwargs = {
            'num_workers': self.num_workers,
            'pin_memory': True
        }
        self.test_batch_size = 1

        # 定义训练的信息
        self.epochs = 2
        self.weight_decay = 0.05
        self.lr = 0.0001
        # 测试信息
        self.eval_root = "/home/datasets/ALTO/Val"

        # 备份配置文件
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        shutil.copy2(self.config_file_path, os.path.join(self.save_path, 'config.py'))

class NYFConfig(object):
    def __init__(self):
        # 实验名称
        self.experiment_name = "DEBUG"
        # 模型权重存放路径
        self.save_path = f"/home/dzh/experiment/{self.experiment_name}"
        # 权重保存模式
        self.save_best = True
        # 配置文件全局路径
        self.config_file_path = "/home/dzh/CASM/configs/config.py"
        # 时间种子
        self.seed = 0

        # 定义模型信息
        self.model_name = "vit"
        self.input_dim = 128
        self.output_dim = 256
        self.is_SPT = False
        self.is_LSA = False
        self.device = 'cuda:0'
        self.checkpoint = None

        # 定义数据集的信息
        self.dataset_name = "NYF"
        self.root = "/home/datasets/NewYorkFly/Train"
        self.transform = transforms.Compose([
            transforms.RandomRotation(30,PIL.Image.BILINEAR, expand=True),
            transforms.Resize(self.input_dim),
            transforms.ToTensor()])
        self.train = True
        self.n_triplets = 200000
        self.batch_size = 48
        self.fliprot = True
        self.use_strategy = True
        self.threshold = 5
        self.each_level_top1 = [10 for _ in range(7)]
        self.num_workers = 10
        self.kwargs = {
            'num_workers': self.num_workers,
            'pin_memory': True
        }
        self.test_batch_size = 1

        # 定义训练的信息
        self.epochs = 20
        self.weight_decay = 0.05
        self.lr = 0.0001
        # 测试信息
        self.eval_root = "/home/datasets/NewYorkFly/Test"

        # 备份配置文件
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        shutil.copy2(self.config_file_path, os.path.join(self.save_path, 'config.py'))