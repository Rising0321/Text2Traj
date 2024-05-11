import argparse

import numpy as np
import torch
import torch.nn.functional as F
import os
import math

from files.trajectories import Trajectories
from model.RAG import RAG
from model.RandomRet import RandomRet
from model.models_Textur import ur_vit_base_patch16
from tqdm.auto import tqdm

from utils.utils import calc_files, myprepare, build_testfile, calculate_metrices
from utils.visualizations import visual

from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import torch.nn as nn
import random
import pandas as pd


# os.environ['TORCH_DISTRIBUTED_DEBUG'] = "INFO"


def build_model(config, accelerator):
    if config.from_terminal is False:
        device = accelerator.device
    else:
        device = config.gpu

    if config.model == "ur_base":
        model = ur_vit_base_patch16(text_condition=config.text_condition).to(device)
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

    seed = args.seed  + torch.distributed.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return accelerator


def train(train_files, config, accelerator, model, optimizer, lr_scheduler, device, epoch):
    train_dataset = Trajectories(train_files, config.image_size, epoch, config.text_condition)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                                                   shuffle=True)
    progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")

    global_step = 0

    model.train()
    for step, batch in enumerate(train_dataloader):

        if config.text_condition == 0:
            clean_text = None
        else:
            clean_text = list(batch[1])
        clean_traj = batch[2].int().to(device)

        with accelerator.accumulate(model):
            loss = model(clean_traj, clean_text)

            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        logs = {"Loss": loss.detach().item()}
        progress_bar.set_postfix(**logs)
        global_step += 1


def test(test_file, test_reduced, config, accelerator, model, epoch):
    test_dataloader = torch.utils.data.DataLoader(test_file, batch_size=config.eval_batch_size,
                                                  shuffle=False)
    progress_bar = tqdm(total=len(test_dataloader), disable=not accelerator.is_local_main_process)

    res_trajs = []

    with torch.no_grad():
        for step, clean_text in enumerate(test_dataloader):
            with torch.autocast("cuda"):
                res_traj = model.module.gen_trajs(config, clean_text).cpu().numpy()

            res_trajs.extend(res_traj)

            progress_bar.update(1)

        visual(res_trajs, epoch, "test", config.eval_batch_size,
               model_name=config.model_name, res_texts=test_file)

        calculate_metrices(test_reduced, res_trajs)


def main(config):
    train_files = calc_files(config.train_path)

    accelerator = init_accelerator(config)

    if accelerator.is_local_main_process:
        test_files, test_reduced = build_testfile(config.train_path)
    else:
        test_files = None
        test_reduced = None

    if config.from_terminal is False:
        device = accelerator.device
    else:
        device = config.gpu

    if config.model == "ur_base":
        model, optimizer, lr_scheduler = build_model(config, accelerator)
    else:

        if config.model == "rag":
            with torch.autocast("cuda"):
                model = RAG(train_files, config, accelerator, device)
        elif config.model == "random":
            with torch.autocast("cuda"):
                model = RandomRet(train_files, config, accelerator, device)
        else:
            raise NotImplementedError()

        test_dataloader = torch.utils.data.DataLoader(test_files, batch_size=config.eval_batch_size,
                                                      shuffle=False)
        res_trajs = []

        for step, clean_text in enumerate(test_dataloader):
            with torch.autocast("cuda"):
                res = model.search(clean_text)
            res_trajs.extend(res)

        # visual(res_trajs, 0, "testRAG", config.eval_batch_size,
        #       model_name=config.model_name, res_texts=test_files)

        calculate_metrices(test_reduced, res_trajs)
        return

    st_epoch = 1
    if config.load_path != "":
        st_epoch = int(config.load_path) + 1

    for epoch in range(st_epoch, config.num_epochs + 1):

        if epoch == st_epoch and config.load_path != "":
            print("loading models")
            accelerator.load_state(f"./logs/{config.model_name}/{config.load_path}.ckpt")

        '''
        if accelerator.is_local_main_process and epoch % config.test_model_epochs == 0:
            test(test_files, test_reduced, config, accelerator, model, epoch)
        '''

        train(train_files, config, accelerator, model, optimizer, lr_scheduler, device, epoch)

        if accelerator.is_local_main_process and epoch % config.save_model_epochs == 0:
            accelerator.save_state(output_dir=f"./logs/{config.model_name}/{epoch}.ckpt")

        if accelerator.is_local_main_process and epoch % config.test_model_epochs == 0:
            test(test_files, test_reduced, config, accelerator, model, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=50000)
    parser.add_argument("--gpu", type=str, default="cuda:2")

    parser.add_argument("--model", type=str, default="ur_base")

    parser.add_argument("--from_terminal", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="Beijing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--save_model_epochs", type=int, default=100)
    parser.add_argument("--test_model_epochs", type=int, default=500)
    parser.add_argument("--mixed_precision", type=str, default='fp16')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_path", type=str, default="./files/Beijing/")
    parser.add_argument("--load_path", type=str, default="")

    parser.add_argument("--test", type=int, default=0)

    parser.add_argument("--text_condition", type=int, default=1)

    args = parser.parse_args()
    #  CUDA_VISIBLE_DEVICES="0,1" accelerate launch train_modelTextur.py
    main(args)

'''
with text:
/home/zhangrx/OpenUR/files/testFiles
'''
