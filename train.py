import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader

import wandb
from data.utils import (
    HFDatasetWrapper,
    pad_collate,
    split_stats,
)
from metrics import log_metrics, plot_all_metrics
from model.load_model import load_EAT_model


def train_one_epoch(model, dataloader, optimizer, loss_fn, device="cuda"):
    """Performs training phase"""
    model.train()
    epoch_loss, correct, seen = 0.0, 0, 0
    pbar = tqdm.tqdm(dataloader, desc="train", leave=False)
    cnt = 0
    for xb, yb, *_ in pbar:
        xb, yb = xb.to(device), yb.float().to(device)
        optimizer.zero_grad()
        logits = model(xb).squeeze(1)  # [B]

        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)
        preds = torch.sigmoid(logits) > 0.5
        correct += (preds == yb.bool()).sum().item()
        seen += xb.size(0)
        cnt += 1
        pbar.set_postfix(
            {"loss": f"{epoch_loss / seen:.4f}", "acc": f"{correct / seen:.3f}"}
        )

    return epoch_loss / seen, correct / seen  # loss, acc


@torch.no_grad()
def valid_one_epoch(model, dataloader, loss_fn, device="cuda"):
    """Performs validation phase"""
    model.eval()
    epoch_loss, correct, seen = 0.0, 0, 0
    all_preds, all_labels = [], []

    for xb, yb, *_ in dataloader:
        xb, yb = xb.to(device), yb.float().to(device)

        logits = model(xb).squeeze(1)
        loss = loss_fn(logits, yb)

        probs = torch.sigmoid(logits)  # ← keep probabilities
        all_preds.append(probs.cpu())
        all_labels.append(yb.cpu())

        epoch_loss += loss.item() * xb.size(0)
        preds = probs > 0.5
        correct += (preds == yb.bool()).sum().item()
        seen += xb.size(0)

    preds_cat = torch.cat(all_preds)  # 1-D tensor, length = dataset
    labels_cat = torch.cat(all_labels)

    avg_loss = epoch_loss / seen
    acc = correct / seen
    return avg_loss, acc, preds_cat, labels_cat


def trainer(args):
    """Fine-tunes EAT model (base/large)"""
    # dist.barrier()  # sync all devices here
    # device = rank
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    if args.wandb:
        wandb.init(name=args.exp_name)
    # load model
    assert args.model_arch == "EAT_large", "Non implemented architecture"

    model = load_EAT_model(args.ckpt_path, args.model_cfg_path).to(args.device)
    # load dataset
    data = load_from_disk(args.preproc_dataset_path)

    split_stats(data["train"], data["val"], data["test"])

    # create dataloaders
    train_wrap = HFDatasetWrapper(data["train"], args.spec_aug)
    val_wrap = HFDatasetWrapper(data["val"])
    test_wrap = HFDatasetWrapper(data["test"])

    train_dl = DataLoader(
        train_wrap,
        batch_size=args.batch_size,
        batch_sampler=None,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_wrap,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=True,
    )

    test_dl = DataLoader(
        test_wrap,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=True,
    )
    # torch.set_float32_matmul_precision("high")

    # Optimizer config
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # lambda1 = lambda epoch: 0.65**epoch # No scheduler for baseline
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # ddp_mp_model = torch.compile(model_slow)
    # ddp_mp_model = DDP(ddp_mp_model, device_ids=[rank])

    # Defining loss function
    loss_function = nn.BCEWithLogitsLoss()

    # def loss_function_weighted(output, target):
    #     # Problem of class imbalance may hinder our progress in training NeuralNetwork
    #     """
    #     target: (B,1)? or (B,2)
    #     output: (B,1)
    #     """
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_acc = 0.0
    for epoch in range(args.num_epochs):
        print(f"Epoch [{epoch + 1}]: Starting training...")
        train_loss, train_acc = train_one_epoch(
            model,
            train_dl,
            optimizer,
            loss_function,
            args.device,
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        valid_loss, valid_acc, val_probs, val_labels = valid_one_epoch(
            model, val_dl, loss_function, args.device
        )

        val_losses.append(valid_loss)
        val_accs.append(valid_acc)

        if args.wandb:
            # existing scalar logs
            wandb.log(
                {
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "validation_Accuracy": valid_acc,
                    "valid_loss": valid_loss,
                }
            )
            # ROC & PR curves on validation set
            log_metrics(val_probs.numpy(), val_labels.numpy())
        print("-------------------------------------------------------------------")
        print(
            f"epoch: {epoch + 1} | train_acc: {train_acc:.3f} | train_loss: {train_loss:.3f} | valid_acc: {valid_acc:.3f} | valid_loss: {valid_loss:.3f}"
        )

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            torch.save(
                model.state_dict(),
                f"exp/{args.exp_name}/{args.save_path}_valid_best_acc.pt",
            )

    _, _, test_probs, test_labels = valid_one_epoch(
        model, test_dl, loss_function, args.device
    )
    plot_all_metrics(
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        val_probs,
        val_labels,
        test_probs,
        test_labels,
        args.exp_name,
        num_classes=2,
    )

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True, help="pre-trained EAT checkpoint")
    p.add_argument("--save_path", default="eat_finetuned", help="prefix for .pt files")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb", type=bool, default=False)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model_arch", type=str, default="EAT_large")
    p.add_argument(
        "--preproc_dataset_path", required=True, type=str, default="EAT_large"
    )
    p.add_argument("--train_frac", default=0.8)
    p.add_argument("--val_frac", default=0.1)
    p.add_argument("--test_frac", default=0.1)
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--spec_aug", type=str, required=True)
    p.add_argument("--model_cfg_path", type=str, required=True)
    args = p.parse_args()

    exp_dir = Path(f"exp/{args.exp_name}")

    if exp_dir.exists():
        backup = exp_dir.with_name(exp_dir.name + "_1")
        if backup.exists():
            shutil.rmtree(backup)  # drop any previous “…_1”
        exp_dir.rename(backup)

    img_path = exp_dir / "images"
    img_path.mkdir(parents=True, exist_ok=True)

    cfg_path = exp_dir / "config.json"
    with cfg_path.open("w") as f:
        json.dump(vars(args), f, indent=4)

    trainer(args)
