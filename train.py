# train_optimized.py
import os
import torch
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from dataset import TBRDataset
from model import TBRPredictionModel, indicators, TIME_BINS
# =====================================================
# 训练配置
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 150
BATCH_SIZE = 8
LR = 1e-5
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SAVE_DIR = os.path.join("modelSavePath", now)
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")

# 不同预测时间区间的样本权重（按 time_idx）
TIME_WEIGHTS = {
    0: 1.0,  # 6 个月
    1: 1.0,  # 12 个月
    2: 0.8,  # 18 个月
    3: 0.5   # 24 个月
}
time_weights_tensor = torch.tensor([TIME_WEIGHTS[k] for k in range(len(TIME_BINS))], device=DEVICE)

# 各種損失的權重係數
LAMBDA_SPARSE    = 0.01      # 稀疏正則通常比較小
LAMBDA_TREAT     = 0.1       # 治療效應重建

# =====================================================
# 单折训练函数
# =====================================================
def train_one_fold(train_dataset, val_dataset, fold_id):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.dataset.collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=val_dataset.dataset.collate
    )

    model = TBRPredictionModel(embed_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mae = float("inf")
    fold_model_path = os.path.join(SAVE_DIR, f"best_model_fold{fold_id}.pth")

    for epoch in range(EPOCHS):
        # =================== Train ===================
        model.train()
        total_mae, total_samples = 0.0, 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}

            out = model(
                phys_values=batch["phys_values"],
                last_treat_types=batch["last_treat_types"],
                delta_t=batch["delta_t"],
                treat_events=batch["treat_events"],
                current_time=batch["current_time"],
                time_idx=batch["time_idx"]
            )

            pred = out["tbr_pred"]
            weights = out.get("weights", None)
            acute = out.get("acute_contrib", None)
            cum = out.get("cum_contrib", None)

            y = batch["target"].to(DEVICE)
            y_baseline = batch.get("baseline_tbr", y)

            mae_weights = time_weights_tensor[batch["time_idx"]]
            loss_main = (mae_weights * torch.abs(pred - y)).mean()

            loss_sparse = 0.0
            if weights is not None:
                loss_sparse = LAMBDA_SPARSE * torch.norm(weights, p=1, dim=1).mean()

            loss_treat = 0.0
            if acute is not None and cum is not None:
                predicted_delta = acute + cum
                true_delta = y - y_baseline
                loss_treat = LAMBDA_TREAT * torch.mean((predicted_delta - true_delta) ** 2)

            loss = loss_main + 0.5 * loss_treat + 0.5 * loss_sparse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = pred.size(0)
            total_mae += torch.abs(pred - y).sum().item()
            total_samples += bs

        train_mae = total_mae / total_samples

        # =================== Validation ===================
        model.eval()
        val_mae, val_samples = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
                out = model(
                    phys_values=batch["phys_values"],
                    last_treat_types=batch["last_treat_types"],
                    delta_t=batch["delta_t"],
                    treat_events=batch["treat_events"],
                    current_time=batch["current_time"],
                    time_idx=batch["time_idx"]
                )
                pred = out["tbr_pred"]
                y = batch["target"].to(DEVICE)
                val_mae += torch.abs(pred - y).sum().item()
                val_samples += pred.size(0)

        val_mae /= val_samples

        print(
            f"[Fold {fold_id}] Epoch [{epoch+1}/{EPOCHS}] "
            f"Train MAE = {train_mae:.4f} | Val MAE = {val_mae:.4f}"
        )

        # =================== Save Best ===================
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), fold_model_path)

    print(f"✅ Fold {fold_id} best Val MAE = {best_val_mae:.4f}")
    return best_val_mae

# =====================================================
# 五折循环
# =====================================================
def train_kfold(excel_file, n_splits=5):
    full_dataset = TBRDataset(excel_file)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
        print(f"\n========== Fold {fold + 1}/{n_splits} ==========")

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        best_mae = train_one_fold(train_subset, val_subset, fold + 1)
        fold_maes.append(best_mae)

    print("\n========== Cross Validation Result ==========")
    print(f"Fold MAEs: {fold_maes}")
    print(f"Mean MAE : {np.mean(fold_maes):.4f}")
    print(f"Std  MAE : {np.std(fold_maes):.4f}")

# =====================================================
# 程序入口
# =====================================================
if __name__ == "__main__":
    train_kfold("Data/trainData1.xlsx", n_splits=5)
