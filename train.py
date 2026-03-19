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

# --- 參數配置 ---
DATA_DIR = "./cv_hw1_data/data"
INPUT_SIZE = 448
BATCH_SIZE = 16         # 合併後每次實際跑 2×BATCH_SIZE 張
EPOCHS = 30
LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 100
MIX_PROB = 0.3
EMA_DECAY = 0.9998      # EMA 權重衰減率，接近 1 代表更平滑
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JIGSAW_N = [8, 4, 2]   # 對應 PMG 論文 n=2^(L-l+1)，S=3


def set_seed(seed=42):
    """設置隨機種子以保證可重現性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jigsaw_generator(images, n):
    """將圖像切成 n×n 塊並隨機打亂，強迫模型學習局部特徵。"""
    if n <= 1:
        return images

    block = images.size(2) // n
    new_images = images.clone()
    grid = [(i, j) for i in range(n) for j in range(n)]

    for b in range(images.size(0)):
        shuffled = grid.copy()
        random.shuffle(shuffled)
        for dst_idx, (src_i, src_j) in enumerate(shuffled):
            dst_i, dst_j = dst_idx // n, dst_idx % n
            new_images[
                b, :,
                dst_i * block:(dst_i + 1) * block,
                dst_j * block:(dst_j + 1) * block,
            ] = images[
                b, :,
                src_i * block:(src_i + 1) * block,
                src_j * block:(src_j + 1) * block,
            ]

    return new_images


class NumericImageFolder(ImageFolder):
    """自定義數據讀取，處理數字命名的資料夾順序。"""

    def find_classes(self, directory):
        classes = [
            d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]
        classes = sorted(classes, key=lambda x: int(x) if x.isdigit() else x)
        class_to_idx = {
            cls: int(cls) if cls.isdigit() else i
            for i, cls in enumerate(classes)
        }
        return classes, class_to_idx


class FocalLoss(nn.Module):
    """處理樣本不均與標籤平滑的 Focal Loss。"""

    def __init__(self, gamma=1.5, smoothing=0.15):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(inputs, self.smoothing / n_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_probs = F.log_softmax(inputs, dim=-1)
        ce = -(smooth_targets * log_probs).sum(dim=-1)
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def soft_cross_entropy(logits, soft_labels):
    """Mix augmentation 用的 soft label cross entropy。"""
    return -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def cutmix_data(x, y, alpha=0.5):
    """執行 CutMix 正則化變換。"""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape

    cut_h = int(H * np.sqrt(1.0 - lam))
    cut_w = int(W * np.sqrt(1.0 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)

    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)

    onehot_a = torch.zeros(x.size(0), NUM_CLASSES, device=x.device).scatter_(
        1, y.unsqueeze(1), 1
    )
    onehot_b = torch.zeros(x.size(0), NUM_CLASSES, device=x.device).scatter_(
        1, y[idx].unsqueeze(1), 1
    )
    return mixed, lam * onehot_a + (1.0 - lam) * onehot_b


def mixup_data(x, y, alpha=0.4):
    """執行 Mixup 正則化變換。"""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[idx]

    onehot_a = torch.zeros(x.size(0), NUM_CLASSES, device=x.device).scatter_(
        1, y.unsqueeze(1), 1
    )
    onehot_b = torch.zeros(x.size(0), NUM_CLASSES, device=x.device).scatter_(
        1, y[idx].unsqueeze(1), 1
    )
    return mixed, lam * onehot_a + (1.0 - lam) * onehot_b


def get_data_loaders():
    """構建訓練與驗證數據載入器。"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
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


def train_one_epoch(model, loader, optimizer, criterion, scheduler, scaler, ema):
    """整合 Jigsaw (隨機粒度) 與 Mix (Step 4) 的 PMG 訓練邏輯。

    優化：每個 batch 隨機抽取一個 jigsaw 粒度（跨 batch 覆蓋全部三個粒度），
    改為 2-way merged forward（jigsaw + mix），計算量為原本 4-way 的一半。
    每次 optimizer.step() 後以 EMA_DECAY 更新 ema 的影子權重。
    """
    model.train()
    t_corr, t_total = 0, 0

    jigsaw_output_map = {8: 0, 4: 1, 2: 2}

    pbar = tqdm(loader, bar_format="{l_bar}{bar:20}{r_bar}")
    for imgs, lbls in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        lbls = lbls.to(DEVICE, non_blocking=True)
        b = imgs.size(0)
        optimizer.zero_grad()

        n = random.choice(JIGSAW_N)
        img_jigsaw = jigsaw_generator(imgs, n=n)

        use_mix = random.random() < MIX_PROB
        if use_mix:
            img_mix, soft_lbls = (
                cutmix_data(imgs, lbls) if random.random() < 0.5
                else mixup_data(imgs, lbls)
            )
        else:
            img_mix = imgs

        all_imgs = torch.cat([img_jigsaw, img_mix], dim=0)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            a1, a2, a3, ac = model(all_imgs)

            outputs = [a1, a2, a3]
            xc_jig = outputs[jigsaw_output_map[n]][:b]
            x_concat = ac[b:]

            loss_jigsaw = criterion(xc_jig, lbls)
            loss_concat = (
                soft_cross_entropy(x_concat, soft_lbls) if use_mix
                else criterion(x_concat, lbls)
            )
            total_loss = loss_jigsaw + loss_concat

        if scaler:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # 每步更新 EMA 影子權重
        ema.update(model)

        scheduler.step()
        t_total += b
        gt = soft_lbls.argmax(1) if use_mix else lbls
        t_corr += (x_concat.argmax(1) == gt).sum().item()

    return t_corr / t_total


def validate(model, loader):
    """驗證模型性能，合併四輸出 softmax（PMG C2 rule）。"""
    model.eval()
    v_corr, v_total = 0, 0

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            lbls = lbls.to(DEVICE, non_blocking=True)
            xc1, xc2, xc3, x_concat = model(imgs)
            probs = (
                F.softmax(xc1, dim=-1) + F.softmax(xc2, dim=-1)
                + F.softmax(xc3, dim=-1) + F.softmax(x_concat, dim=-1)
            )
            v_corr += (probs.argmax(1) == lbls).sum().item()
            v_total += lbls.size(0)

    return v_corr / v_total


class EMA:
    """Exponential Moving Average 維護模型權重的平滑版本。

    訓練時保存一份影子權重（shadow），驗證與推論時套用影子權重，
    結束後還原訓練權重，對模型本身無侵入性。
    前期使用 warmup decay 避免影子權重卡在隨機初始化。
    """

    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            k: v.clone().float()
            for k, v in model.state_dict().items()
        }

    def update(self, model):
        """以目前模型權重更新影子權重，decay 隨 step 數暖身至目標值。"""
        # warmup：前期 effective_decay 接近 0，後期趨近 self.decay
        effective_decay = min(
            self.decay, (1 + self.num_updates) / (10 + self.num_updates)
        )
        self.num_updates += 1
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(effective_decay).add_(
                    v.float(), alpha=1.0 - effective_decay
                )

    def apply(self, model):
        """將影子權重載入模型（驗證前呼叫）。"""
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(next(model.parameters()).dtype)
                               for k, v in self.shadow.items()})

    def restore(self, model):
        """還原訓練權重（驗證後呼叫）。"""
        model.load_state_dict(self._backup)

    def state_dict(self):
        """回傳影子權重，供儲存 checkpoint 使用。"""
        return self.shadow


def main():
    """主訓練流程：初始化 → 訓練 → EMA 驗證 → 儲存最佳模型。"""
    set_seed()
    train_loader, val_loader = get_data_loaders()
    model = get_model().to(DEVICE)
    print(f"Total Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if hasattr(torch, "compile") and os.name != "nt":
        model = torch.compile(model)

    criterion = FocalLoss()  # 使用 class 預設值 gamma=1.5, smoothing=0.15
    ema = EMA(model)

    head_keys = ("classifier", "conv_block", "pool", "se_extra")
    backbone_params = [p for n, p in model.named_parameters()
                       if not any(k in n for k in head_keys)]
    head_params = [p for n, p in model.named_parameters()
                   if any(k in n for k in head_keys)]

    optimizer = optim.AdamW(
        [{"params": backbone_params, "lr": LR},
         {"params": head_params, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[LR, HEAD_LR],
        epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    best_acc = 0.0
    for epoch in range(EPOCHS):
        train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, scaler, ema
        )

        # 驗證時套用 EMA 影子權重，結束後還原
        ema.apply(model)
        val_acc = validate(model, val_loader)
        ema.restore(model)

        print(f"Epoch {epoch + 1}: Train Acc: {train_acc * 100:.2f}%  Val Acc: {val_acc * 100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ema.state_dict(), "best_model.pth")  # 儲存 EMA 影子權重
            print(f"--> Saved Best: {val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()