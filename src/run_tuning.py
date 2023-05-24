import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import argparse
from typing import Any
import modeling.trainer as trainer
from data_loading import dataloaders
import data_loading
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import multiprocessing
from modeling.model import OpenPoseMC, ResNetMC
from settings import ABS_PATH

torch.manual_seed(0)
global_args = None;

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def train_mc(tune_config, filename, model_name, out_path):
    # Save the device
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(device)

    video_transformer = data_loading.transforms.VideoFilePathToTensor(
        max_len=None, fps=10, padding_mode="last"
    )
    H, W = 256, 256
    transforms = torchvision.transforms.Compose(
        [
            data_loading.transforms.VideoResize([H, W]),
            # transforms.VideoRandomHorizontalFlip(),
        ]
    )

    # get the dataloaders. can make test and val sizes 0 if you don't want them
    if tune_config["dataloader"] == "mc":
        train_dl, val_dl, test_dl = dataloaders.get_mc_data_loaders(
            video_transformer=video_transformer,
            batch_size=1,
            val_batch_size=1,
            test_batch_size=1,
            transforms=transforms,
            preload_videos=False,
            labels=[
                "pelvis_tilt",
                "ankle_angle_l",
                "ankle_angle_r",
                "hip_adduction_r",
                "hip_adduction_l",
            ],
            num_workers=2,
        )
    elif tune_config["dataloader"] == "vid":
        raise NotImplementedError

    # TensorBoard training log
    writer = SummaryWriter(log_dir="expt/")

    train_config = trainer.TrainerConfig(
        max_epochs=tune_config["max_epochs"],
        learning_rate=tune_config["lr"],
        num_workers=4,
        writer=writer,
        ckpt_path="expt/params_mc_testing.pt",
    )

    if model_name == "resnetMC":
        model = ResNetMC(num_outputs=5, H=H, W=W, freeze=tune_config["freeze"])
    elif model_name == "openposeMC":
        model = OpenPoseMC(num_outputs=5, H=H, W=W, device=device, freeze=tune_config["freeze"])

    trainer_cls = trainer.Trainer(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        config=train_config,
        val_dataloader=val_dl,
        median_freq_weights=False,
    )
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    for epoch in range(train_config.max_epochs):
        print(epoch)
        train_losses.append(trainer_cls.train(split="train", step=epoch))
        val_loss = trainer_cls.train(split="val", step=epoch)
        val_losses.append(val_loss)
        print("Val loss:", val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ABS_PATH, f"{out_path}/bestmodel_{epoch}.params"))
    # write csv of losses
    with open(f"{out_path}/loss.csv", "w") as f:
        for train_loss, val_loss in zip(train_losses, val_losses):
            f.write(f"{train_loss},{val_loss}\n")


def main(model_name, outpath, num_samples=5, max_num_epochs=20, gpus_per_trial=1, filename=None, version=''):
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1" 

    tune_config = {
        "lr": tune.loguniform(1e-5, 1e-4),
        "lr_decay": tune.choice([False, True]),
        "max_epochs": tune.choice([15, 20]),
        "model_name" : model_name,
        "freeze": tune.choice([False]),
        "dataloader": global_args.dataloader,
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=max_num_epochs,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"]
    )
    
    result = tune.run(
        partial(train_mc, filename=filename, model_name = model_name, out_path = outpath),
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=tune_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('--model', type=str, help='Choose resnetMC/openposeMC', required=True)
    clargs = argp.parse_args()

    import sys
    sys.stdout.fileno = lambda: False

    # Make args for tuning and set global equal to those args.
    resnetMC_args = Namespace(
        dataloader = "mc",
    )
    
    if clargs.model == 'resnetMC':
         global_args = resnetMC_args
         params_output_name = "resnetMC-model.params"
         trials, epochs_per_trial  = 5, 20
    elif clargs.model == 'openposeMC':
         global_args = resnetMC_args
         params_output_name = "openposeMC-model.params"
         trials, epochs_per_trial  = 5, 20
    else:
         print("Choose a valid model.")
         sys.exit(0)
    model_name = clargs.model
        
    main(model_name=model_name, outpath=clargs.model,
         num_samples=trials, max_num_epochs=epochs_per_trial, gpus_per_trial=1, filename=params_output_name)