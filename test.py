# test.py
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from dataset import TBRDataset
from model import TBRPredictionModel

# ==============================
# 配置
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8

MODEL_DIR = "modelSavePath/2026-01-14_16-01-27"

MODEL_PATHS = [
    os.path.join(MODEL_DIR, f"best_model_fold{i}.pth")
    for i in range(1, 6)
]

TEST_EXCEL = "Data/testData1.xlsx"

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SAVE_DIR = os.path.join("output", now)
os.makedirs(SAVE_DIR, exist_ok=True)

FIG_PATH = os.path.join(SAVE_DIR, "ensemble_tbr_pred_vs_true.png")

# ==============================
# Test (5-fold Ensemble)
# ==============================
def test_ensemble():
    # -------- 1. Dataset & Loader --------
    test_dataset = TBRDataset(TEST_EXCEL)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=test_dataset.collate
    )

    all_fold_preds = []   # list of [N]
    targets = []          # list of [batch_size]，只在 fold 1 收集

    # -------- 2. Loop over folds --------
    for fold_id, model_path in enumerate(MODEL_PATHS, 1):
        assert os.path.exists(model_path), f"Model not found: {model_path}"
        print(f"\n[Fold {fold_id}] Loading model: {model_path}")

        model = TBRPredictionModel(embed_dim=64).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        preds = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {
                    k: v.to(DEVICE) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                out = model(
                    phys_values=batch["phys_values"],
                    last_treat_types=batch["last_treat_types"],
                    delta_t=batch["delta_t"],
                    treat_events=batch["treat_events"],
                    current_time=batch["current_time"],
                    time_idx=batch["time_idx"],
                    baseline_phys = batch["baseline_phys"],
                    baseline_tbr = batch["baseline_tbr"],
                    static_feat = batch["static_feat"]
                )

                preds.append(out["tbr_pred"].cpu().numpy())

                # ✅ targets 只在第一个 fold 收集一次
                if fold_id == 1:
                    targets.append(batch["target"].cpu().numpy())

        preds = np.concatenate(preds)      # [N]
        all_fold_preds.append(preds)

        # 临时 MAE（此时 targets 还没 fully concat，但 fold 1 已完整）
        if fold_id == 1:
            tmp_targets = np.concatenate(targets)
        else:
            tmp_targets = targets_concat

        fold_mae = np.mean(np.abs(preds - tmp_targets))
        print(f"[Fold {fold_id}] MAE = {fold_mae:.4f}")

        if fold_id == 1:
            targets_concat = tmp_targets  # 固定 targets

    # -------- 3. Ensemble --------
    targets = targets_concat                      # [N]
    all_fold_preds = np.stack(all_fold_preds)     # [5, N]

    fold_maes = np.mean(np.abs(all_fold_preds - targets[None, :]), axis=1)
    mean_preds = all_fold_preds.mean(axis=0)
    ensemble_mae = np.mean(np.abs(mean_preds - targets))

    print("\n========== Test Result ==========")
    print(f"Fold MAEs    : {fold_maes}")
    print(f"Mean ± Std   : {fold_maes.mean():.4f} ± {fold_maes.std():.4f}")
    print(f"Ensemble MAE : {ensemble_mae:.4f}")

    # -------- 4. Scatter Plot (Ensemble) --------
    # plt.figure(figsize=(6, 6))
    # plt.scatter(targets, mean_preds, alpha=0.6)
    #
    # min_v = min(targets.min(), mean_preds.min())
    # max_v = max(targets.max(), mean_preds.max())
    # plt.plot([min_v, max_v], [min_v, max_v], "r--")
    #
    # plt.xlabel("Ground Truth TBR")
    # plt.ylabel("Ensemble Predicted TBR")
    # plt.title("Ensemble Prediction vs Ground Truth (Test Set)")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(FIG_PATH, dpi=300)
    # plt.show()
    #
    # print(f"\n[OK] Ensemble test finished.")
    # print(f"Scatter plot saved to: {FIG_PATH}")

    # -------- 3. Ensemble statistics --------
    mean_preds = all_fold_preds.mean(axis=0)
    std_preds  = all_fold_preds.std(axis=0)
    x = np.arange(len(targets))

    plt.figure(figsize=(10, 4))

    # 阴影：± std
    plt.fill_between(
        x,
        mean_preds - std_preds,
        mean_preds + std_preds,
        color="red",
        alpha=0.25,
        label="Prediction ± Std"
    )

    # 真值（黑点）
    plt.scatter(
        x,
        targets,
        color="black",
        s=35,
        label="True values",
        zorder=3
    )

    # 预测均值（红星）
    plt.scatter(
        x,
        mean_preds,
        color="red",
        marker="*",
        s=80,
        label="Predicted values",
        zorder=4
    )

    plt.xlabel("Unobserved Testing Samples")
    plt.ylabel("TBR Value")
    plt.title("Ensemble Prediction with Uncertainty (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    plt.show()

    print(f"[OK] Figure saved to: {FIG_PATH}")

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    test_ensemble()
