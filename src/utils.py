import torch
from pytorch3d.loss.chamfer import chamfer_distance
import matplotlib.pyplot as plt

def get_chamfer_distance_loss(x1: torch.Tensor, x2:torch.Tensor):
    return chamfer_distance(x1, x2, batch_reduction="mean", point_reduction="mean")[0]

def visualize_sample(pc: torch.Tensor, pred_pc: torch.Tensor, savename: str):
    """
    Input:
        pc, pred_pc: [num_sample, num_ponts, 3]
    """
    pc = pc.clone().detach().cpu().numpy()
    pred_pc = pred_pc.clone().detach().cpu().numpy()

    fig = plt.figure()
    num_sample = pc.shape[0]
    for i in range(num_sample):
        ax = fig.add_subplot(num_sample,2,i*2+1,projection="3d")
        ax.scatter(pc[i,:,0], pc[i,:,2], pc[i,:,1])
        ax.axis("off")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        ax = fig.add_subplot(num_sample, 2, i*2+2, projection="3d")
        ax.scatter(pred_pc[i,:,0], pred_pc[i,:,2], pred_pc[i,:,1])
        ax.axis("off")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

    plt.tight_layout()
    plt.savefig(savename)
