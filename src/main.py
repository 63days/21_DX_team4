from dataset import MyDataset
from model import MyModel
from utils import get_chamfer_distance_loss, visualize_sample
import torch
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


def step(model, imgs, pc, device):
    imgs = imgs.to(device)
    pc = pc.to(device)
    pred_pc = model(imgs)
    
    assert pred_pc.shape == pc.shape
    loss = get_chamfer_distance_loss(pc, pred_pc)

    return loss, pred_pc

def main(args):
    now = datetime.now()
    save_dir = now.strftime("../results/%m-%d/%H-%M-%S")
    ckpt_dir = os.path.join(save_dir, "checkpoints")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    device = f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu"

    model = MyModel(num_points=args.num_points).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 80], gamma=0.1
    )

    train_ds = MyDataset(phase="train", num_points=args.num_points)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_ds = MyDataset(phase="val", num_points=args.num_points)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    print("loaded dataloaders")

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):

        """ Training """
        pbar = tqdm(train_dl)
        train_loss_one_epoch = []
        for imgs, pc in pbar:
            loss, pred_pc = step(model, imgs, pc, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"E: {epoch:02} | train_loss: {loss:.4f}")
            train_loss_one_epoch.append(loss.item())

        scheduler.step()

        train_loss_one_epoch = sum(train_loss_one_epoch) / len(train_loss_one_epoch)
        train_losses.append(train_loss_one_epoch)
        
        """ Validation """
        with torch.no_grad():
            val_loss_one_epoch = []
            for imgs, pc in val_dl:
                loss, pred_pc = step(model, imgs, pc, device)
                val_loss_one_epoch.append(loss.item())

            val_loss_one_epoch = sum(val_loss_one_epoch) / len(val_loss_one_epoch)
            val_losses.append(val_loss_one_epoch)

            print(f"val_loss: {val_loss_one_epoch:.4f}")

            # For every 10 epochs, save a checkpoint if current loss is lower than best loss.
            if (epoch + 1) % 10 == 0:
                if val_loss_one_epoch < best_val_loss:
                    best_val_loss = val_loss_one_epoch
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            ckpt_dir,
                            f"epoch={epoch:02}-val_loss={val_loss_one_epoch:.4f}.ckpt",
                        ),
                    )
            savename = os.path.join(save_dir, f"epoch={epoch}_sample.png")
            visualize_sample(pc[4:8], pred_pc[4:8], savename)
    
    # save loss plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(train_losses)), train_losses, label="train losses")
    ax.plot(range(len(train_losses)), val_losses, label="valid losses")
    ax.legend()
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--num_points", type=int, default=10000)

    args = parser.parse_args()
    main(args)
