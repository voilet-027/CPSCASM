from models import vit
import torch

def Log(msg):
    print(f"[Model] {msg}")

class ModelMaker(object):
    def __init__(self, config):
        self.cfg = config
        
        Log("Creating Model")

        # make model
        if self.cfg.model_name == "vit":
            sd = 0.1
            patch_size = 4 if self.cfg.input_dim == 32 else 8
            model = vit.ViT(
                img_size=self.cfg.input_dim,
                patch_size=patch_size,
                num_classes=self.cfg.output_dim,
                dim=192,
                mlp_dim_ratio=2,
                depth=9,
                heads=12,
                dim_head=192//12,
                stochastic_depth=sd,
                is_SPT=self.cfg.is_SPT,
                is_LSA=self.cfg.is_LSA).to(self.cfg.device)
            if self.cfg.checkpoint is not None:
                Log("Loading Checkpoint")
                checkpoint = torch.load(self.cfg.checkpoint)
                model.load_state_dict(checkpoint["state_dict"])
        else:
            assert f"Config.model_name {self.cfg.model_name} is not expected"