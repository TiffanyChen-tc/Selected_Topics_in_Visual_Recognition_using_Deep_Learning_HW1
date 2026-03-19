import os

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import get_model

# --- 參數配置 ---
TEST_DIR = "./cv_hw1_data/data/test"
INPUT_SIZE = 448    # 須與訓練時的 INPUT_SIZE 一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tta_transforms(size):
    """生成 TTA 變換列表，涵蓋多尺度與水平翻轉組合。

    每張圖會以 8 種不同的裁切/翻轉方式各推論一次，
    再將所有 softmax 機率加總後取 argmax，降低單一裁切帶來的偏差。
    """
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    hflip = transforms.RandomHorizontalFlip(p=1.0)

    # 三種 resize 倍率 × 是否水平翻轉 = 6 種，再加兩種局部裁切
    scales = [1.0, 1.14, 1.25]
    tta = []
    for s in scales:
        base = transforms.Compose([
            transforms.Resize(int(size * s)),
            transforms.CenterCrop(size),
            transforms.ToTensor(), norm,
        ])
        flipped = transforms.Compose([
            transforms.Resize(int(size * s)),
            transforms.CenterCrop(size),
            hflip, transforms.ToTensor(), norm,
        ])
        tta += [base, flipped]

    # 先縮小裁切再放大回 size：模擬推論時看到更多背景的情況
    tta += [
        transforms.Compose([
            transforms.Resize(int(size * 1.4)),
            transforms.CenterCrop(int(size * 1.15)),
            transforms.Resize(size),
            transforms.ToTensor(), norm,
        ]),
        transforms.Compose([
            transforms.Resize(int(size * 1.4)),
            transforms.CenterCrop(int(size * 1.15)),
            transforms.Resize(size),
            hflip, transforms.ToTensor(), norm,
        ]),
    ]
    return tta


def main():
    """載入 EMA 模型權重，對測試集逐張進行 TTA 推論並輸出 prediction.csv。"""
    model = get_model().to(DEVICE)
    # model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.load_state_dict(torch.load("best_finetune.pth", map_location=DEVICE))
    model.eval()

    all_files = sorted([
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    tta_list = get_tta_transforms(INPUT_SIZE)

    results = []
    for fname in tqdm(all_files, desc="Inference", bar_format="{l_bar}{bar:20}{r_bar}"):
        img = Image.open(os.path.join(TEST_DIR, fname)).convert("RGB")
        agg_probs = None

        with torch.no_grad():
            for t in tta_list:
                tensor = t(img).unsqueeze(0).to(DEVICE)
                xc1, xc2, xc3, x_concat = model(tensor)
                probs = (
                    F.softmax(xc1, dim=-1) + F.softmax(xc2, dim=-1)
                    + F.softmax(xc3, dim=-1) + F.softmax(x_concat, dim=-1)
                )
                agg_probs = probs if agg_probs is None else agg_probs + probs

        results.append({
            "image_name": fname.split(".")[0],
            "pred_label": agg_probs.argmax(1).item(),
        })

    pd.DataFrame(results).to_csv("prediction.csv", index=False)
    print("Saved prediction.csv")


if __name__ == "__main__":
    main()