"""
Train the multi-task malnutrition risk model on cleaned DHS data.
Saves: trained model weights, feature scaler, region encoder, and metrics.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
import joblib
import json

from model import MalnutritionMultiTaskNet

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

NUMERIC_FEATURES = ["age_months", "sex", "wealth_index", "residence_type",
                     "mother_education", "mother_age"]
REGION_FEATURE = "region"

TARGET_Z = ["haz", "waz", "whz"]
TARGET_CLASS = ["stunting_class", "underweight_class", "wasting_class"]


class MalnutritionDataset(Dataset):
    def __init__(self, X_numeric, X_region, y):
        self.X_numeric = torch.tensor(X_numeric, dtype=torch.float32)
        self.X_region = torch.tensor(X_region, dtype=torch.long)
        self.y = {k: torch.tensor(v, dtype=torch.float32 if 'z' == k[-1] else torch.long)
                   for k, v in y.items()}

    def __len__(self):
        return len(self.X_numeric)

    def __getitem__(self, idx):
        item_y = {k: v[idx] for k, v in self.y.items()}
        return self.X_numeric[idx], self.X_region[idx], item_y


def compute_class_weights(series, n_classes=3):
    """Inverse-frequency class weights so rare 'severe' cases aren't ignored.
    Missing false negatives on severe malnutrition is the costly error here,
    so we deliberately bias the loss toward catching minority (severe) cases."""
    counts = series.value_counts().reindex(range(n_classes), fill_value=0) + 1  # +1 smoothing
    weights = (1.0 / counts)
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights.values, dtype=torch.float32)


def multitask_loss(preds, targets, class_weights=None, class_weight=1.0, reg_weight=1.0):
    mse = nn.MSELoss()
    cw = class_weights or {}
    ce_haz = nn.CrossEntropyLoss(weight=cw.get("haz"))
    ce_waz = nn.CrossEntropyLoss(weight=cw.get("waz"))
    ce_whz = nn.CrossEntropyLoss(weight=cw.get("whz"))

    loss = 0.0
    loss += reg_weight * mse(preds["haz_z"], targets["haz"])
    loss += reg_weight * mse(preds["waz_z"], targets["waz"])
    loss += reg_weight * mse(preds["whz_z"], targets["whz"])

    loss += class_weight * ce_haz(preds["haz_logits"], targets["stunting_class"])
    loss += class_weight * ce_waz(preds["waz_logits"], targets["underweight_class"])
    loss += class_weight * ce_whz(preds["whz_logits"], targets["wasting_class"])
    return loss


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "clean_kr.csv"))
    print("Training on", len(df), "records")

    # Encode region
    region_encoder = LabelEncoder()
    df["region_enc"] = region_encoder.fit_transform(df[REGION_FEATURE])
    n_regions = len(region_encoder.classes_)

    # Train/val/test split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # Scale numeric features (fit on train only)
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[NUMERIC_FEATURES])
    X_val_num = scaler.transform(val_df[NUMERIC_FEATURES])
    X_test_num = scaler.transform(test_df[NUMERIC_FEATURES])

    def build_targets(d):
        return {
            "haz": d["haz"].values, "waz": d["waz"].values, "whz": d["whz"].values,
            "stunting_class": d["stunting_class"].values,
            "underweight_class": d["underweight_class"].values,
            "wasting_class": d["wasting_class"].values,
        }

    train_ds = MalnutritionDataset(X_train_num, train_df["region_enc"].values, build_targets(train_df))
    val_ds = MalnutritionDataset(X_val_num, val_df["region_enc"].values, build_targets(val_df))
    test_ds = MalnutritionDataset(X_test_num, test_df["region_enc"].values, build_targets(test_df))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    class_weights = {
        "haz": compute_class_weights(train_df["stunting_class"]),
        "waz": compute_class_weights(train_df["underweight_class"]),
        "whz": compute_class_weights(train_df["wasting_class"]),
    }
    print("Class weights (normal/moderate/severe):")
    for k, v in class_weights.items():
        print(f"  {k}: {v.tolist()}")

    model = MalnutritionMultiTaskNet(n_numeric=len(NUMERIC_FEATURES), n_region=n_regions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    best_val_loss = float("inf")
    patience, patience_counter = 25, 0
    n_epochs = 200

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for x_num, x_region, y in train_loader:
            optimizer.zero_grad()
            preds = model(x_num, x_region)
            loss = multitask_loss(preds, y, class_weights=class_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_num)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_num, x_region, y in val_loader:
                preds = model(x_num, x_region)
                loss = multitask_loss(preds, y, class_weights=class_weights)
                val_loss += loss.item() * len(x_num)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "best_model.pt"))
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter >= patience:
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "best_model.pt")))
    model.eval()

    all_preds = {"haz": [], "waz": [], "whz": []}
    all_targets = {"haz": [], "waz": [], "whz": []}
    all_class_preds = {"stunting": [], "underweight": [], "wasting": []}
    all_class_targets = {"stunting": [], "underweight": [], "wasting": []}

    with torch.no_grad():
        for x_num, x_region, y in test_loader:
            preds = model(x_num, x_region)
            all_preds["haz"].extend(preds["haz_z"].tolist())
            all_preds["waz"].extend(preds["waz_z"].tolist())
            all_preds["whz"].extend(preds["whz_z"].tolist())
            all_targets["haz"].extend(y["haz"].tolist())
            all_targets["waz"].extend(y["waz"].tolist())
            all_targets["whz"].extend(y["whz"].tolist())

            all_class_preds["stunting"].extend(preds["haz_logits"].argmax(1).tolist())
            all_class_preds["underweight"].extend(preds["waz_logits"].argmax(1).tolist())
            all_class_preds["wasting"].extend(preds["whz_logits"].argmax(1).tolist())
            all_class_targets["stunting"].extend(y["stunting_class"].tolist())
            all_class_targets["underweight"].extend(y["underweight_class"].tolist())
            all_class_targets["wasting"].extend(y["wasting_class"].tolist())

    metrics = {}
    print("\n--- Test Set Performance ---")
    for k in ["haz", "waz", "whz"]:
        mae = mean_absolute_error(all_targets[k], all_preds[k])
        metrics[f"{k}_mae"] = mae
        print(f"{k.upper()} regression MAE: {mae:.3f} SD")

    for k in ["stunting", "underweight", "wasting"]:
        acc = accuracy_score(all_class_targets[k], all_class_preds[k])
        f1 = f1_score(all_class_targets[k], all_class_preds[k], average="macro")
        metrics[f"{k}_accuracy"] = acc
        metrics[f"{k}_macro_f1"] = f1
        print(f"{k} classification: accuracy={acc:.3f}  macro-F1={f1:.3f}")

    # Save preprocessing artifacts + metrics
    joblib.dump(scaler, os.path.join(BASE_DIR, "models", "scaler.pkl"))
    joblib.dump(region_encoder, os.path.join(BASE_DIR, "models", "region_encoder.pkl"))
    with open(os.path.join(BASE_DIR, "models", "feature_config.json"), "w") as f:
        json.dump({"numeric_features": NUMERIC_FEATURES, "n_regions": int(n_regions)}, f)
    with open(os.path.join(BASE_DIR, "models", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model, scaler, encoder, and metrics to models/ dir")


if __name__ == "__main__":
    main()
