from functools import partial

from files.trajectories import Trajectories
import torch
from tqdm.auto import tqdm

from model.t5 import t5_encode_text


class RAG:

    def __init__(self, train_files, config, accelerator, device):

        self.encode_text = partial(t5_encode_text, name="files/LLM")

        self.vector_database = []
        self.trajectory = []
        self.texts = []

        for epoch in range(len(train_files)):
            train_dataset = Trajectories(train_files, config.image_size, epoch, config.text_condition)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                                                           shuffle=True)
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_text = list(batch[1])
                clean_traj = batch[2].int().to(device)
                text_embeds = self.encode_text(clean_text).float()  # B * L * D
                for i in range(len(clean_text)):
                    text = clean_text[i]
                    flag = 0
                    for saved in self.texts:
                        if text == saved:
                            flag = 1
                            break
                    if flag == 0:
                        self.texts.append(text)
                        self.vector_database.append(torch.max(text_embeds[i], dim=0)[0])
                        self.trajectory.append(clean_traj[i])
                progress_bar.update(1)
                # break
        self.vector_database = torch.stack(self.vector_database, dim=0).float().cuda()

    def search(self, text):
        ans = []
        text_embeds = self.encode_text(text).float()

        from tqdm import tqdm
        for j in tqdm(range(len(text_embeds))):
            max_pos = 0
            max_sim = 0
            embed = torch.max(text_embeds[j], dim=0)[0].cuda()  # D

            # print(embed.unsqueeze(0).shape, self.vector_database.shape)
            sims = torch.cosine_similarity(embed.unsqueeze(0), self.vector_database, dim=1)

            # choose the max pos using torch.max
            max_pos = torch.argmax(sims)

            ans.append(self.trajectory[max_pos].cpu().numpy())

        return ans
