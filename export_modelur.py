import argparse

import numpy as np
import torch
import torch.nn.functional as F
import os
import math

from files.trajectories import Trajectories
from model.models_ur import ur_vit_base_patch16
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

from utils.utils import calc_files, myprepare

from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator


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

    print("loading models")
    accelerator.load_state(f"./logs/{config.model_name}/{config.load_path}.ckpt")

    now_rank = torch.distributed.get_rank()

    uids = []
    embeddings = []

    with torch.no_grad():

        for epoch in range(st_epoch, config.num_epochs + 1):

            model.eval()

            if len(accelerator._dataloaders) == len(train_files):
                break
            else:
                train_dataset = Trajectories(train_files, config.iur_size, epoch, config.scale)
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                                                               shuffle=True)
                train_dataloader = accelerator.prepare_data_loader(train_dataloader)

            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            model.train()

            for step, batch in enumerate(train_dataloader):
                uid = batch[0].long().to(device)
                uid = uid[:,0]
                clean_traj = batch[2].float().to(device)
                # print(uid.shape)
                mask = uid < 1000000
                # print(mask)
                uid, clean_traj = uid[mask], clean_traj[mask]

                embed = model.module.get_embed(clean_traj)

                uids.append(uid.cpu())
                embeddings.append(embed.cpu())

                progress_bar.update(1)

    uids = torch.cat(uids, dim=0).cpu().numpy()
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()

    np.save(f"./embeddings/{config.model_name}_uids_{now_rank}.npy", uids)
    np.save(f"./embeddings/{config.model_name}_embeddings_{now_rank}.npy", embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--iur_size", type=int, default=96)
    parser.add_argument("--train_batch_size", type=int, default=350)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=50000)
    parser.add_argument("--gpu", type=str, default="cuda:0")

    parser.add_argument("--model", type=str, default="ur_base")

    parser.add_argument("--from_terminal", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="Finalur")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--save_traj_epochs", type=int, default=300)
    parser.add_argument("--save_model_epochs", type=int, default=300)
    parser.add_argument("--mixed_precision", type=str, default='fp16')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_path", type=str, default="/workdir/file/grids/")
    parser.add_argument("--load_path", type=str, default="6000")

    parser.add_argument("--postProcess", type=int, default=0)

    parser.add_argument("--scale", type=int, default=0)
    parser.add_argument("--test", type=int, default=0)

    args = parser.parse_args()
    # CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" accelerate launch export_modelur.py
    main(args)

