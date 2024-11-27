from configs.config import Config
from Factorys.ModelFactory import ModelMaker
from Factorys.DataFactory import DataMaker
from loss.loss import SyntheticLoss
import torch
import random
import numpy as np
import math
import sys
from tqdm import tqdm
from torch.autograd import Variable
from eval import Evaler
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def setSeedGlobal(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cosine_decay(epoch, start_value, end_value, start_epoch, end_epoch):
    if epoch < start_epoch:
        return start_value
    elif epoch > end_epoch:
        return end_value
    else:
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        cosine_decay_value = 0.5 * (1 + math.cos(math.pi * progress))
        return end_value + (start_value - end_value) * cosine_decay_value

def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

if __name__ == "__main__":
    # ===== Config =====
    config = Config()

    # ===== Seed Frozen =====
    setSeedGlobal(config.seed)

    # ===== Model =====
    model = ModelMaker().create(config)
    parameters = get_params_groups(model, weight_decay=config.weight_decay)
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)

    # ===== Record Train Data =====
    df = pd.DataFrame(columns=[
        "epoch", "loss", "triplet", "hardnet", "cop"
    ])
    df_eval = pd.DataFrame(columns=[
        "top1", "top2", "top3", "top4", "top5", "dis" 
    ])
    # ===== Trainning =====
    threshold = 10.0
    top1 = [7 for _ in range(10)]

    precision_best = 0

    for epoch in range(0, config.epochs):

        threshold = cosine_decay(epoch, 10.0, 0.0, 0, config.epochs)
        config.threshold = threshold 
        config.each_level_top1 = top1

        if config.use_strategy == False:
            if epoch == 0:
                dataloader = DataMaker().create(config)
                lr_scheduler = create_lr_scheduler(optimizer, len(dataloader), config.epochs,warmup=False, warmup_epochs=1)
        else:
            dataloader = DataMaker().create(config)
            if epoch == 0:
                lr_scheduler = create_lr_scheduler(optimizer, len(dataloader), config.epochs,warmup=False, warmup_epochs=1)

        # switch to train mode
        model.train()
        optimizer.zero_grad()
        mean_loss = torch.zeros(1).to(config.device)

        # Log train info
        data_loader = tqdm(dataloader, file=sys.stdout)
        syntheticloss = SyntheticLoss()
        syntheticloss.setEpoch(epoch=epoch)

        for step, data in enumerate(data_loader):

            a, p, n = data
            a, p, n = a.to(config.device), p.to(config.device), n.to(config.device)
            a, p, n = Variable(a), Variable(p), Variable(n),
            a, p, n = model(a), model(p), model(n)

            triplet, hardnet, cop, loss = syntheticloss(a, p, n)
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # update lr
            lr_scheduler.step()
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

            data_loader.desc = "[train epoch {}]  loss: {:.5f}, lr: {:.7f}".format(
                epoch,
                mean_loss.item(),
                loss.item(),
                optimizer.param_groups[0]["lr"])
            
            # record data
            df = pd.concat([df, pd.DataFrame([{
                "epoch": epoch, 
                "loss": loss.item(), 
                "triplet": triplet, 
                "hardnet": hardnet, 
                "cop": cop
            }])], ignore_index=True)
        
        # Eval
        topN, dis, top1 = Evaler.eval(model=model, config=config)

        # record data
        df_eval = pd.concat([df_eval, pd.DataFrame([{
            "top1": topN[0], 
            "top2": topN[1], 
            "top3": topN[2], 
            "top4": topN[3], 
            "top5": topN[4], 
            "dis": dis
        }])], ignore_index=True)
        
        # Save model
        if topN[0] > precision_best:
            precision_best = topN[0]
            if config.save_best:
                torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                        os.path.join(config.save_path, f"{math.ceil(precision_best)}.pth"))
                print(f"[Save] Save the New best checkpoint, top N: {topN}")
        if not config.save_best:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                        os.path.join(config.save_path, f"epoch{epoch}_{math.ceil(precision_best)}.pth"))
            print(f"[Save] Save the {epoch} checkpoint, top N: {topN}")

    df.to_csv(os.path.join(config.save_path, "train_loss.csv"))
    df_eval.to_csv(os.path.join(config.save_path, "eval.csv"))