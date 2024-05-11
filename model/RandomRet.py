from files.trajectories import Trajectories
import torch
from tqdm.auto import tqdm
import numpy as np

class RandomRet:

    def __init__(self, train_files, config, accelerator, device):

        self.trajectory = []

        for epoch in range(len(train_files)):
            train_dataset = Trajectories(train_files, config.image_size, epoch, config.text_condition)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                                                           shuffle=True)
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_traj = batch[2].int().to(device)
                for i in range(len(clean_traj)):
                    self.trajectory.append(clean_traj[i])
                progress_bar.update(1)

    def search(self, text):
        ans = []
        from tqdm import tqdm
        for j in tqdm(range(len(text))):
            # random choose a pos from self.trajectory
            pos = np.random.randint(0, len(self.trajectory))
            ans.append(self.trajectory[pos].cpu().numpy())

        return ans
