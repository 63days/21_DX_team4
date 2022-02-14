import torch
import torch.nn as nn
from torchvision import models

class MyModel(nn.Module):
    def __init__(self, num_points=2048):
        super().__init__()
        self.num_points = num_points

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 512)

        self.decoder = nn.Sequential(
            nn.Linear(1024, num_points // 4),
            nn.BatchNorm1d(num_points//4),
            nn.ReLU(),
            nn.Linear(num_points // 4, num_points //2),
            nn.BatchNorm1d(num_points // 2),
            nn.ReLU(),
            nn.Linear(num_points//2, num_points),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(num_points),
            nn.ReLU(),
            nn.Linear(num_points, num_points * 3)
        )

    def forward(self, imgs):
        """
        Input:
            img: [B, num_views, 3, H: 224, W: 224]
        Output:
            x: predicted point clouds [B, num_points: 2048, 3]
        """
        B, num_views = imgs.shape[0], imgs.shape[1]

        latent_codes = []
        for i in range(num_views):
            x = self.resnet(imgs[:,i])
            latent_codes.append(x)

        latent_codes = torch.stack(latent_codes, 1) # [B, num_views, 512]
        l_mean = torch.mean(latent_codes, 1) # obtaining the mean latent code of multi-view images. # [B,512]
        l_max = torch.max(latent_codes, 1)[0] # maxpooling the latent codes. #[B, 512]

        x = torch.cat([l_mean, l_max], 1) #[B, 1024]
        x = self.decoder(x) # [B,num_points*3]
        x = x.reshape(B, self.num_points, 3)

        return x
