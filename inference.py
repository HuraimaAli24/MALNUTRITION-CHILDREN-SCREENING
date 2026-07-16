"""
Inference utilities: MC Dropout uncertainty estimation + SHAP explainability
for the malnutrition multi-task model.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
import shap

from model import MalnutritionMultiTaskNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEVERITY_LABELS = {0: "normal", 1: "moderate", 2: "severe"}


class MalnutritionPredictor:
    def __init__(self):
        self.scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        self.region_encoder = joblib.load(os.path.join(BASE_DIR, "region_encoder.pkl"))
        self.cfg = json.load(open(os.path.join(BASE_DIR, "feature_config.json")))
        self.numeric_features = self.cfg["numeric_features"]

        self.model = MalnutritionMultiTaskNet(
            n_numeric=len(self.numeric_features), n_region=self.cfg["n_regions"]
        )
        self.model.load_state_dict(
            torch.load(os.path.join(BASE_DIR, "best_model.pt"), map_location="cpu")
        )
        self.model.eval()

        clean_path = os.path.join(BASE_DIR, "clean_kr.csv")
        df = pd.read_csv(clean_path)
        df["region_enc"] = self.region_encoder.transform(df["region"])
        bg_sample = df.sample(min(100, len(df)), random_state=42)
        self.bg_numeric = self.scaler.transform(bg_sample[self.numeric_features])
        self.bg_region = bg_sample["region_enc"].values

        self._build_shap_explainer()

    def _model_forward_numpy(self, x_combined: np.ndarray) -> np.ndarray:
        n_numeric = len(self.numeric_features)
        x_numeric = torch.tensor(x_combined[:, :n_numeric], dtype=torch.float32)
        region_idx = torch.tensor(x_combined[:, n_numeric].round().astype(int), dtype=torch.long)
        region_idx = torch.clamp(region_idx, 0, self.cfg["n_regions"] - 1)
        with torch.no_grad():
            preds = self.model(x_numeric, region_idx)
        out = torch.stack([preds["haz_z"], preds["waz_z"], preds["whz_z"]], dim=1)
        return out.numpy()

    def _build_shap_explainer(self):
        bg_combined = np.concatenate(
            [self.bg_numeric, self.bg_region.reshape(-1, 1)], axis=1
        )
        self.explainer = shap.KernelExplainer(self._model_forward_numpy, bg_combined[:30])

    def _prepare_input(self, record: dict):
        numeric_vals = np.array([[record[f] for f in self.numeric_features]], dtype=float)
        numeric_scaled = self.scaler.transform(numeric_vals)
        region_enc = self.region_encoder.transform([record["region"]])[0]
        return numeric_scaled, region_enc

    def predict(self, record: dict, mc_samples: int = 30, explain: bool = True) -> dict:
        numeric_scaled, region_enc = self._prepare_input(record)
        x_numeric = torch.tensor(numeric_scaled, dtype=torch.float32)
        region_idx = torch.tensor([region_enc], dtype=torch.long)

        self.model.enable_mc_dropout()
        haz_samples, waz_samples, whz_samples = [], [], []
        haz_cls_samples, waz_cls_samples, whz_cls_samples = [], [], []

        with torch.no_grad():
            for _ in range(mc_samples):
                preds = self.model(x_numeric, region_idx)
                haz_samples.append(preds["haz_z"].item())
                waz_samples.append(preds["waz_z"].item())
                whz_samples.append(preds["whz_z"].item())
                haz_cls_samples.append(preds["haz_logits"].argmax(1).item())
                waz_cls_samples.append(preds["waz_logits"].argmax(1).item())
                whz_cls_samples.append(preds["whz_logits"].argmax(1).item())

        self.model.eval()

        def who_severity_from_z(z):
            if z < -3:
                return 2
            elif z < -2:
                return 1
            return 0

        def summarize(samples, cls_samples):
            mean_z = float(np.mean(samples))
            std_z = float(np.std(samples))
            severity_cls = who_severity_from_z(mean_z)

            vals, counts = np.unique(cls_samples, return_counts=True)
            classifier_majority = int(vals[np.argmax(counts)])
            classifier_agreement = float(np.max(counts) / len(cls_samples))
            heads_agree = (classifier_majority == severity_cls)

            return {
                "z_score_mean": round(mean_z, 2),
                "z_score_std": round(std_z, 2),
                "severity": SEVERITY_LABELS[severity_cls],
                "mc_dropout_confidence": round(1.0 - min(std_z / 2.0, 1.0), 2),
                "classifier_cross_check": {
                    "agrees_with_regression": heads_agree,
                    "classifier_agreement": round(classifier_agreement, 2),
                },
            }

        result = {
            "stunting": summarize(haz_samples, haz_cls_samples),
            "underweight": summarize(waz_samples, waz_cls_samples),
            "wasting": summarize(whz_samples, whz_cls_samples),
        }

        if explain:
            x_combined = np.concatenate([numeric_scaled, [[region_enc]]], axis=1)
            shap_values = self.explainer.shap_values(x_combined, nsamples=100, silent=True)
            feature_names = self.numeric_features + ["region"]
            explanations = {}
            outputs = ["stunting", "underweight", "wasting"]
            sv = np.array(shap_values)
            if sv.ndim == 3:
                for i, out_name in enumerate(outputs):
                    contribs = sv[0, :, i]
                    explanations[out_name] = {
                        feature_names[j]: round(float(contribs[j]), 3) for j in range(len(feature_names))
                    }
            result["explanations"] = explanations

        return result
