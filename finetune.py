import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import get_model
from train import (
    EMA, FocalLoss, NumericImageFolder,
    set_seed, validate,
)

# --- Fine-tune 參數（獨立於 train.py，避免互相干擾）---
DATA_DIR = "./cv_hw1_data/data"
INPUT_SIZE = 448
BATCH_SIZE = 16
FT_EPOCHS = 8
FT_LR = 5e-6        # backbone 極小 LR，避免破壞已收斂的特徵
FT_HEAD_LR = 5e-5   # head 稍大但仍保守
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 100
EMA_DECAY = 0.9998
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_finetune_loaders():
    """Fine-tune 用的資料載入器：降低 augmentation 強度以減少訓練雜訊。

    第二階段的目標是讓模型在較乾淨的樣本上校準決策邊界，
    過強的 augmentation 反而會干擾已收斂的表示。
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # 不使用 TrivialAugmentWide / RandomErasing / RandomVerticalFlip
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(INPUT_SIZE * 256 / 224)),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = NumericImageFolder(os.path.join(DATA_DIR, "train"), train_transform)
    val_set = NumericImageFolder(os.path.join(DATA_DIR, "val"), val_transform)

    counts = Counter(train_set.targets)
    sample_weights = [1.0 / counts[t] for t in train_set.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE * 4, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    return train_loader, val_loader


def finetune_one_epoch(model, loader, optimizer, criterion, scheduler, scaler, ema):
    """第二階段訓練：硬標籤，不使用 Jigsaw / Mix augmentation。

    fine-tune 的目標是讓分類頭在乾淨樣本上收斂，
    不再需要 jigsaw 強迫多粒度學習（backbone 已有此能力）。
    """
    model.train()
    t_corr, t_total = 0, 0

    pbar = tqdm(loader, bar_format="{l_bar}{bar:20}{r_bar}")
    for imgs, lbls in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        lbls = lbls.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            xc1, xc2, xc3, x_concat = model(imgs)
            loss = (
                criterion(xc1, lbls) + criterion(xc2, lbls)
                + criterion(xc3, lbls) + criterion(x_concat, lbls)
            )

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        ema.update(model)
        scheduler.step()
        t_total += imgs.size(0)
        t_corr += (x_concat.argmax(1) == lbls).sum().item()

    return t_corr / t_total


def main():
    """Fine-tune 流程：載入 best_model.pth → 低 LR 校準 → 儲存 best_finetune.pth。"""
    set_seed()
    train_loader, val_loader = get_finetune_loaders()

    model = get_model().to(DEVICE)
    ckpt = torch.load("best_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt)
    print("Loaded best_model.pth")
    print(f"Total Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = FocalLoss(gamma=1.0, smoothing=0.05)  # fine-tune 用更軟的 loss
    ema = EMA(model, decay=EMA_DECAY)

    head_keys = ("classifier", "conv_block", "pool", "se_extra")
    backbone_params = [p for n, p in model.named_parameters()
                       if not any(k in n for k in head_keys)]
    head_params = [p for n, p in model.named_parameters()
                   if any(k in n for k in head_keys)]

    optimizer = optim.AdamW(
        [{"params": backbone_params, "lr": FT_LR},
         {"params": head_params, "lr": FT_HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    # Cosine decay 從當前 LR 降到 0，不再有 warmup 衝高 LR 破壞已收斂的權重
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FT_EPOCHS * len(train_loader), eta_min=0,
    )
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    # 先測一次載入權重的基準 val acc
    ema.apply(model)
    baseline = validate(model, val_loader)
    ema.restore(model)
    print(f"Baseline Val Acc (from best_model.pth): {baseline * 100:.2f}%")

    best_acc = baseline
    for epoch in range(FT_EPOCHS):
        train_acc = finetune_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, scaler, ema
        )

        ema.apply(model)
        val_acc = validate(model, val_loader)
        ema.restore(model)

        print(f"FT Epoch {epoch + 1}: Train Acc: {train_acc * 100:.2f}%  Val Acc: {val_acc * 100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ema.state_dict(), "best_finetune.pth")
            print(f"--> Saved Best: {val_acc * 100:.2f}%")

    print(f"\nFine-tune 完成。最終最佳 Val Acc: {best_acc * 100:.2f}%")
    print("推論時使用best_finetune.pth。")


if __name__ == "__main__":
    main()