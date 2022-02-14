import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, phase="train", data_dir="../data", num_points=2048):
        super().__init__()
        self.num_points = num_points

        if phase == "train":
            self.data_dir = os.path.join(data_dir, "2SET_STL")
        elif phase == "val" or phase == "test":
            self.data_dir = os.path.join(data_dir, "1SET_STL")

        with open(os.path.join(self.data_dir, "models.txt"), "r") as f:
            self.models = [line.rstrip() for line in f]

        self.images = []
        self.pointclouds = []
        for x in self.models:
            # view point 1
            imgname1 = os.path.join(self.data_dir, "images", f"{x}_elev_70_azim_160.png")
            img1 = Image.open(imgname1).convert("RGB")
            # view point 2
            imgname2 = os.path.join(self.data_dir, f"images", f"{x}_elev_40_azim_40.png")
            img2 = Image.open(imgname2).convert("RGB")
            # view point 3
            imgname3 = os.path.join(self.data_dir, "images", f"{x}_elev_40_azim_70.png")
            img3 = Image.open(imgname3).convert("RGB")
            self.images.append((img1, img2, img3))

            pcname = os.path.join(self.data_dir, "points", f"{x}_10000.npy")
            pc = np.load(pcname)[:num_points].astype(np.float32)
            self.pointclouds.append(pc)

    def __getitem__(self, idx):
        multiview_imgs = self.images[idx]
        
        new_imgs = []
        for img in multiview_imgs:
            new_imgs.append(preprocess(img))
        
        new_imgs = torch.stack(new_imgs)
        pc = torch.from_numpy(self.pointclouds[idx])
        
        return new_imgs, pc

    def __len__(self):
        return len(self.models)
