import argparse

import numpy as np
import torch
import torch.nn.functional as F
import os
import math

from files.trajectories import Trajectories
from model.models_ur import ur_vit_base_patch16
from tqdm.auto import tqdm

from utils.utils import calc_files, myprepare
from utils.visualizations import visual

from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import torch.nn as nn

import pandas as pd


# os.environ['TORCH_DISTRIBUTED_DEBUG'] = "INFO"


def build_model(config, accelerator):
    if config.from_terminal is False:
        device = accelerator.device
    else:
        device = config.gpu

    if config.model == "ur_base":
        model = ur_vit_base_patch16().to(device)
    else:
        raise NotImplementedError()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(10000000 * 100),
    )

    model, optimizer, lr_scheduler = myprepare(accelerator,
                                               model,
                                               optimizer,
                                               lr_scheduler
                                               )

    return model, optimizer, lr_scheduler


def init_accelerator(config):
    accelerator = Accelerator(
        project_dir="./saved_models",
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard"
    )

    if accelerator.is_main_process:
        if os.path.join("./logs", config.model_name) is not None:
            os.makedirs(f"./logs/{config.model_name}", exist_ok=True)
        if os.path.join("./visualizations", config.model_name) is not None:
            os.makedirs(f"./visualizations/{config.model_name}", exist_ok=True)
        if os.path.join("./metrices", config.model_name) is not None:
            os.makedirs(f"./metrices/{config.model_name}", exist_ok=True)
        accelerator.init_trackers("train_example")

    seed = args.seed  # + torch.distributed.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    return accelerator


def main(config):
    train_files = calc_files(config.train_path)

    accelerator = init_accelerator(config)

    if config.from_terminal is False:
        device = accelerator.device
    else:
        device = config.gpu

    model, optimizer, lr_scheduler = build_model(config, accelerator)

    st_epoch = 1
    if config.load_path != "":
        st_epoch = int(config.load_path) + 1

    for epoch in range(st_epoch, config.num_epochs + 1):

        if config.test:
            gen_traj = model.gen_trajs(config, model)
            visual(gen_traj, epoch, "fake", config.eval_batch_size, config.model_name)
            exit(0)

        model.train()

        if len(accelerator._dataloaders) == len(train_files):
            train_dataloader = accelerator._dataloaders[epoch % len(train_files)]
        else:
            train_dataset = Trajectories(train_files, config.iur_size, epoch, config.scale)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                                                           shuffle=True)
            train_dataloader = accelerator.prepare_data_loader(train_dataloader)

        if epoch == st_epoch and config.load_path != "":
            print("loading models")
            accelerator.load_state(f"./logs/{config.model_name}/{config.load_path}.ckpt")

        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        global_step = 0

        now_trajs = []
        now_homes = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            clean_traj = batch[2].float().to(device)

            now_homes.extend(batch[1].cpu().numpy())
            now_trajs.extend(batch[2].cpu().numpy())

            with accelerator.accumulate(model):
                loss = model(clean_traj)

                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"Loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            global_step += 1

        if accelerator.is_main_process and epoch % config.save_model_epochs == 0:
            # visual(now_trajs, epoch, "real", config.eval_batch_size, config.model_name)

            gen_traj = model.module.gen_trajs(config, model)

            visual(gen_traj, epoch, "fake", config.eval_batch_size, config.model_name)
            
            # exit(0)

            accelerator.save_state(output_dir=f"./logs/{config.model_name}/{epoch}.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--iur_size", type=int, default=96)
    parser.add_argument("--train_batch_size", type=int, default=350)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=50000)
    parser.add_argument("--gpu", type=str, default="cuda:1")

    parser.add_argument("--model", type=str, default="ur_base")

    parser.add_argument("--from_terminal", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="FinalurD")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--save_model_epochs", type=int, default=100)
    parser.add_argument("--mixed_precision", type=str, default='fp16')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_path", type=str, default="/home/zhangrx/TrajDiff/files/withGID")
    parser.add_argument("--load_path", type=str, default="")

    parser.add_argument("--postProcess", type=int, default=0)

    parser.add_argument("--scale", type=int, default=0)
    parser.add_argument("--test", type=int, default=0)

    args = parser.parse_args()
    # CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" accelerate launch train_modelur.py
    main(args)