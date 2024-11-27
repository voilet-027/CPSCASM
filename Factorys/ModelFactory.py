from models import vit
import torch
import torch.nn as nn

def Log(msg):
    print(f"[Model] {msg}")

class ModelMaker(object):
    @staticmethod
    def create(config):
        Log("Creating Model")
        config 
        # make model
        if config.model_name == "vit":
            sd = 0.1
            patch_size = 4 if config.input_dim == 32 else 8
            model = vit.ViT(
                img_size=config.input_dim,
                patch_size=patch_size,
                num_classes=config.output_dim,
                dim=192,
                mlp_dim_ratio=2,
                depth=9,
                heads=12,
                dim_head=192//12,
                stochastic_depth=sd,
                is_SPT=config.is_SPT,
                is_LSA=config.is_LSA).to(config.device)
            if config.checkpoint is not None:
                Log("Loading Checkpoint")
                checkpoint = torch.load(config.checkpoint)
                model.load_state_dict(checkpoint["state_dict"])
            return model
        else:
            assert f"Config.model_name {config.model_name} is not expected"