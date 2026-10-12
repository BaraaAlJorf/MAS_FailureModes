#!/usr/bin/env python3
"""AgentiCARE.py (Updated & Corrected)

Unified Patient-Complexity / Agentic-Escalation Routing for Medical AI Benchmarks.
Supports Mortality and Length of Stay (LOS) tasks across 5 routing settings.
Includes Uncalibrated vs. Calibrated Candidate Baselines, Per-Feature-Family Test Evaluation,
Robust Prediction Loading, DeLong's Test (AUROC), Paired Bootstrap Test (AUPRC),
and JSON Export of Routed Predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import math
import os
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import ttest_rel
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from tqdm import tqdm

try:
  import torch
except Exception as exc:
  torch = None
  _TORCH_IMPORT_ERROR = exc
else:
  _TORCH_IMPORT_ERROR = None

warnings.filterwarnings("ignore")

# =============================================================================
# 1. Constants & Configurations
# =============================================================================

DEFAULT_RESULTS_BASE = Path("results/")

DATASET_PATHS = {
    "Mortality": {
        "Train": Path(
            "datasets/multimodal_dataset_splits/train.jsonl"
        ),
        "Test": Path(
            "datasets/multimodal_dataset_splits/test.jsonl"
        ),
        "label_key": "in_hospital_mortality_48hr",
        "prob_key": "mortality_probability",
    },
    "LOS": {
        "Train": Path(
            "datasets/corrected_los_labels/train.jsonl"
        ),
        "Test": Path(
            "datasets/corrected_los_labels/test.jsonl"
        ),
        "label_key": "los_7",
        "prob_key": "los_probability",
    },
}

MODEL_IDS = {
    "Huatuo": "FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL",
    "Qwen": "Qwen_Qwen2.5-VL-7B-Instruct",
}

INPUT7_COLUMNS = [
    "has_cxr",
    "has_rr",
    "ps_length_score",
    "ps_lexical_score",
    "rr_length_score",
    "rr_lexical_score",
    "ehr_acuity_score",
]

RR_ABNORMAL_TERMS = [
    "opacity",
    "opacities",
    "consolidation",
    "edema",
    "effusion",
    "pneumothorax",
    "atelectasis",
    "cardiomegaly",
    "infiltrate",
    "infiltrates",
    "vascular congestion",
    "pulmonary congestion",
    "pulmonary edema",
    "pleural",
    "enlarged",
    "worsening",
    "severe",
    "bilateral",
    "low lung volume",
    "hypoinflation",
    "airspace",
    "interstitial",
    "hazy",
    "basilar",
    "bibasilar",
    "collapse",
    "fracture",
    "tube",
    "line",
    "catheter",
    "endotracheal",
    "ng tube",
    "picc",
    "central venous",
]

RR_REASSURING_PHRASES = [
    "no acute cardiopulmonary abnormality",
    "no acute cardiopulmonary process",
    "clear lungs",
    "lungs are clear",
    "no focal consolidation",
    "no consolidation",
    "no pleural effusion",
    "no pneumothorax",
    "without pneumothorax",
    "cardiomediastinal silhouette is normal",
    "normal cardiomediastinal",
    "no acute disease",
    "no acute findings",
    "unremarkable",
]

PS_PROBLEM_TERMS = [
    "sepsis",
    "shock",
    "respiratory failure",
    "renal failure",
    "kidney failure",
    "heart failure",
    "liver failure",
    "cirrhosis",
    "pneumonia",
    "ards",
    "myocardial infarction",
    "nstemi",
    "stemi",
    "stroke",
    "cva",
    "hemorrhage",
    "bleeding",
    "metastatic",
    "cancer",
    "malignancy",
    "leukemia",
    "lymphoma",
    "copd",
    "dialysis",
    "esrd",
    "aki",
    "dka",
    "coma",
    "intubated",
    "vasopressor",
    "pressors",
    "hypotension",
    "hypoxia",
    "infection",
    "bacteremia",
    "transplant",
    "immunosuppressed",
    "diabetes",
    "hypertension",
    "cad",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9_./+-]+")

# =============================================================================
# 2. Setup Combinations & Complexity Hierarchy
# =============================================================================


def get_target_combinations(
    task: str, mode: str, prompt_arch: str
) -> List[Dict[str, str]]:
  """Orders setups strictly by escalating complexity. LOS flips Simple and Complex."""
  if mode == "prompt-only":
    setup_dir = prompt_arch
    arch_code = "SAS" if prompt_arch == "SingleAgent" else "Independant MAS"
    if task == "LOS":
      return [
          {
              "arch": arch_code,
              "setup_dir": setup_dir,
              "prompt": "none",
              "prompt_name": "None",
          },
          {
              "arch": arch_code,
              "setup_dir": setup_dir,
              "prompt": "complex",
              "prompt_name": "Complex",
          },
          {
              "arch": arch_code,
              "setup_dir": setup_dir,
              "prompt": "simple",
              "prompt_name": "Simple",
          },
      ]
    else:
      return [
          {
              "arch": arch_code,
              "setup_dir": setup_dir,
              "prompt": "none",
              "prompt_name": "None",
          },
          {
              "arch": arch_code,
              "setup_dir": setup_dir,
              "prompt": "simple",
              "prompt_name": "Simple",
          },
          {
              "arch": arch_code,
              "setup_dir": setup_dir,
              "prompt": "complex",
              "prompt_name": "Complex",
          },
      ]

  if task == "LOS":
    return [
        {
            "arch": "SAS",
            "setup_dir": "SingleAgent",
            "prompt": "none",
            "prompt_name": "None",
        },
        {
            "arch": "SAS",
            "setup_dir": "SingleAgent",
            "prompt": "simple",
            "prompt_name": "Simple",
        },
        {
            "arch": "SAS",
            "setup_dir": "SingleAgent",
            "prompt": "complex",
            "prompt_name": "Complex",
        },
        {
            "arch": "Independant MAS",
            "setup_dir": "MajorityVote",
            "prompt": "none",
            "prompt_name": "None",
        },
        {
            "arch": "Independant MAS",
            "setup_dir": "MajorityVote",
            "prompt": "complex",
            "prompt_name": "Complex",
        },
        {
            "arch": "Independant MAS",
            "setup_dir": "MajorityVote",
            "prompt": "simple",
            "prompt_name": "Simple",
        },
    ]
  else:
    return [
        {
            "arch": "SAS",
            "setup_dir": "SingleAgent",
            "prompt": "none",
            "prompt_name": "None",
        },
        {
            "arch": "SAS",
            "setup_dir": "SingleAgent",
            "prompt": "simple",
            "prompt_name": "Simple",
        },
        {
            "arch": "Independant MAS",
            "setup_dir": "MajorityVote",
            "prompt": "simple",
            "prompt_name": "Simple",
        },
        {
            "arch": "Independant MAS",
            "setup_dir": "MajorityVote",
            "prompt": "complex",
            "prompt_name": "Complex",
        },
    ]


def setup_name(model_name: str, combo: Dict[str, str]) -> str:
  return f"{model_name} | {combo['arch']} | {combo['prompt_name']}"


def expected_setup_names(
    model_name: str, combinations: List[Dict[str, str]]
) -> List[str]:
  return [setup_name(model_name, c) for c in combinations]


# =============================================================================
# DeLong's Test (AUROC) & Bootstrap Test (AUPRC)
# =============================================================================


def compute_midrank(x):
  J = np.argsort(x)
  Z = x[J]
  N = len(x)
  T = np.zeros(N, dtype=float)
  i = 0
  while i < N:
    j = i
    while j < N and Z[j] == Z[i]:
      j += 1
    T[i:j] = 0.5 * (i + j - 1)
    i = j
  T2 = np.empty(N, dtype=float)
  T2[J] = T + 1
  return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
  m = label_1_count
  n = predictions_sorted_transposed.shape[1] - m
  positive_examples = predictions_sorted_transposed[:, :m]
  negative_examples = predictions_sorted_transposed[:, m:]
  tx = np.empty([positive_examples.shape[0], m], dtype=float)
  ty = np.empty([negative_examples.shape[0], n], dtype=float)
  tz = np.empty([predictions_sorted_transposed.shape[0], m + n], dtype=float)
  for r in range(predictions_sorted_transposed.shape[0]):
    tx[r, :] = compute_midrank(positive_examples[r, :])
    ty[r, :] = compute_midrank(negative_examples[r, :])
    tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
  aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
  v01 = (tz[:, :m] - tx[:, :]) / n
  v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
  sx = np.cov(v01)
  sy = np.cov(v10)
  delongcov = sx / m + sy / n
  return aucs, delongcov


def calc_pvalue(aucs, sigma):
  l = np.array([[1, -1]])
  z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
  z = z[0, 0]
  return 10 ** scipy.stats.norm.logsf(z, loc=0, scale=1) * 2


def delong_roc_test(
    ground_truth: np.ndarray,
    predictions_one: np.ndarray,
    predictions_two: np.ndarray,
) -> float:
  """Computes DeLong's p-value comparing the AUROC of two predictions on the same test set."""
  ground_truth = np.asarray(ground_truth)
  predictions_one = np.asarray(predictions_one)
  predictions_two = np.asarray(predictions_two)

  order = np.argsort(ground_truth)[::-1]
  ground_truth = ground_truth[order]
  predictions_one = predictions_one[order]
  predictions_two = predictions_two[order]

  label_1_count = int(ground_truth.sum())
  preds = np.vstack((predictions_one, predictions_two))
  aucs, cov = fastDeLong(preds, label_1_count)
  p_val = calc_pvalue(aucs, cov)
  return p_val


def paired_bootstrap_auprc_test(
    ground_truth: np.ndarray,
    predictions_one: np.ndarray,
    predictions_two: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
  """Computes observed AUPRC difference (predictions_two - predictions_one) and its

  two-tailed paired bootstrap p-value under H0: delta_AUPRC = 0.
  """
  ground_truth = np.asarray(ground_truth)
  p1 = np.asarray(predictions_one)
  p2 = np.asarray(predictions_two)

  obs_diff = safe_auprc(ground_truth, p2) - safe_auprc(ground_truth, p1)

  rng = np.random.default_rng(seed)
  n = len(ground_truth)
  diffs = []

  for _ in range(n_bootstraps):
    idx = rng.choice(n, n, replace=True)
    y_b = ground_truth[idx]
    if np.unique(y_b).size < 2:
      continue
    auprc1 = safe_auprc(y_b, p1[idx])
    auprc2 = safe_auprc(y_b, p2[idx])
    diffs.append(auprc2 - auprc1)

  diffs = np.array(diffs)
  if len(diffs) == 0:
    return obs_diff, np.nan

  p_lower = np.mean(diffs <= 0)
  p_upper = np.mean(diffs >= 0)
  p_val = min(1.0, 2.0 * min(p_lower, p_upper))
  return float(obs_diff), float(p_val)


# =============================================================================
# 3. Base Metrics & Bootstrapping
# =============================================================================


def safe_auc(
    y_true: np.ndarray, y_prob: np.ndarray, default: float = np.nan
) -> float:
  y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
  if y_true.size == 0 or np.unique(y_true).size < 2:
    return default
  try:
    return float(roc_auc_score(y_true, y_prob))
  except Exception:
    return default


def safe_auprc(
    y_true: np.ndarray, y_prob: np.ndarray, default: float = np.nan
) -> float:
  y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
  if y_true.size == 0 or np.unique(y_true).size < 2:
    return default
  try:
    return float(average_precision_score(y_true, y_prob))
  except Exception:
    return default


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
  """Confidence-based Expected Calibration Error."""
  y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
  if len(y_true) == 0:
    return np.nan

  preds = (y_prob >= 0.5).astype(int)
  confs = np.maximum(y_prob, 1.0 - y_prob)
  accs = (preds == y_true).astype(float)

  bins = np.linspace(0.0, 1.0, n_bins + 1)
  binned = np.digitize(confs, bins, right=True) - 1
  binned = np.clip(binned, 0, n_bins - 1)

  ece = 0.0
  n = len(y_true)

  for b in range(n_bins):
    mask = binned == b
    if np.any(mask):
      bin_acc = np.mean(accs[mask])
      bin_conf = np.mean(confs[mask])
      ece += np.abs(bin_acc - bin_conf) * (np.sum(mask) / float(n))

  return float(ece)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
  y_true = np.asarray(y_true, dtype=int)
  y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
  return {
      "n": int(len(y_true)),
      "prevalence": float(np.mean(y_true)) if len(y_true) else np.nan,
      "AUROC": safe_auc(y_true, y_prob),
      "AUPRC": safe_auprc(y_true, y_prob),
      "ECE": expected_calibration_error(y_true, y_prob),
  }


def evaluate_with_bootstrap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[Dict[str, Any], np.ndarray]:
  base_metrics = compute_metrics(y_true, y_prob)
  if n_bootstraps <= 0:
    return {
        k: (v, v, v) if k in ["AUROC", "AUPRC", "ECE"] else v
        for k, v in base_metrics.items()
    }, np.array([])

  rng = np.random.default_rng(seed)
  n = len(y_true)
  boot_results = {k: [] for k in ["AUROC", "AUPRC", "ECE"]}

  for _ in tqdm(range(n_bootstraps), desc="Bootstrapping CIs", leave=False):
    idx = rng.choice(n, n, replace=True)
    m = compute_metrics(y_true[idx], y_prob[idx])
    for k in boot_results.keys():
      boot_results[k].append(m[k])

  alpha = (1.0 - ci_level) / 2.0
  lower_q = alpha * 100
  upper_q = (1.0 - alpha) * 100

  final_res = {}
  for k, val in base_metrics.items():
    if k not in boot_results:
      final_res[k] = val
    else:
      arr = np.array(boot_results[k])
      arr_clean = arr[~np.isnan(arr)]
      if len(arr_clean) == 0 or np.isnan(val):
        final_res[k] = (val, np.nan, np.nan)
      else:
        final_res[k] = (
            val,
            np.percentile(arr_clean, lower_q),
            np.percentile(arr_clean, upper_q),
        )

  return final_res, np.array(boot_results["AUROC"])


def print_formatted_metrics(prefix: str, metrics_dict: Dict[str, Any]):
  def fmt(k):
    v = metrics_dict.get(k)
    if isinstance(v, tuple) and len(v) == 3:
      if np.isnan(v[0]):
        return "NaN"
      return f"{v[0]:.4f} (95% CI: {v[1]:.4f}-{v[2]:.4f})"
    if isinstance(v, float):
      return "NaN" if np.isnan(v) else f"{v:.4f}"
    return str(v)

  print(
      f"{prefix} AUROC: {fmt('AUROC')} | AUPRC: {fmt('AUPRC')} | ECE:"
      f" {fmt('ECE')}"
  )


# =============================================================================
# 4. JSONL Metadata & Text Extraction
# =============================================================================


def tokenize(text: Any) -> List[str]:
  return TOKEN_RE.findall(text.lower()) if isinstance(text, str) else []


def count_terms(text: str, terms: Sequence[str]) -> int:
  if not isinstance(text, str) or not text.strip():
    return 0
  text_l = text.lower()
  cnt = 0
  for term in terms:
    term_l = term.lower()
    if " " in term_l:
      cnt += text_l.count(term_l)
    else:
      cnt += len(re.findall(r"\b" + re.escape(term_l) + r"\b", text_l))
  return int(cnt)


def text_metrics(text: Any, unavailable_marker: str) -> Tuple[int, int]:
  if (
      not isinstance(text, str)
      or not text.strip()
      or text.strip() == unavailable_marker
  ):
    return 0, 0
  toks = tokenize(text)
  return len(toks), len(set(toks))


def read_jsonl_metadata(json_path: Path, label_key: str) -> pd.DataFrame:
  records = []
  with open(json_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc=f"Reading {json_path.name}"):
      if not line.strip():
        continue
      data = json.loads(line)
      stay_id = str(data.get("stay_id"))
      labels = data.get("labels", {})
      label = labels.get(label_key, data.get(label_key, None))

      ps_text = data.get("patient_summary_text", "")
      rr_text = data.get("radiology_report_text", "")
      cxr_path = data.get("cxr_image_path", "")
      ehr_path = data.get("ehr_timeseries_path", "")

      # --- Demographic Extraction ---
      age_match = re.search(r"-\s*Age:\s*([0-9.]+)", ps_text)
      gender_match = re.search(r"-\s*Gender:\s*([a-zA-Z]+)", ps_text)

      age_raw = float(age_match.group(1)) if age_match else np.nan
      gender_str = (
          gender_match.group(1).strip().lower() if gender_match else ""
      )

      is_female = 1.0 if gender_str in ["f", "female"] else 0.0
      is_male = 1.0 if gender_str in ["m", "male"] else 0.0

      ps_len, ps_lex = text_metrics(ps_text, "Patient summary not available")
      rr_len, rr_lex = text_metrics(rr_text, "Radiology report not available")

      records.append({
          "stay_id": stay_id,
          "label": int(label) if label is not None else np.nan,
          "has_cxr": float(
              bool(cxr_path) and str(cxr_path) != "CXR not available"
          ),
          "has_rr": float(
              bool(rr_text) and str(rr_text) != "Radiology report not available"
          ),
          "has_ehr": float(bool(ehr_path) and os.path.exists(str(ehr_path))),
          "ehr_path": str(ehr_path) if ehr_path else "",
          "age_raw": age_raw,
          "is_female": is_female,
          "is_male": is_male,
          "ps_len_raw": float(ps_len),
          "ps_lex_raw": float(ps_lex),
          "ps_ttr_raw": float(ps_lex) / max(float(ps_len), 1.0),
          "rr_len_raw": float(rr_len),
          "rr_lex_raw": float(rr_lex),
          "rr_ttr_raw": float(rr_lex) / max(float(rr_len), 1.0),
          "ps_problem_count_raw": float(
              count_terms(ps_text, PS_PROBLEM_TERMS)
          ),
          "rr_abnormal_count_raw": float(
              count_terms(rr_text, RR_ABNORMAL_TERMS)
          ),
          "rr_reassuring_count_raw": float(
              count_terms(rr_text, RR_REASSURING_PHRASES)
          ),
      })
  df = pd.DataFrame(records)
  if df["label"].isna().any():
    raise ValueError(f"Missing labels in {json_path} using key '{label_key}'.")
  df["label"] = df["label"].astype(int)
  return df


def ehr_patient_stats_fallback(meta_df: pd.DataFrame) -> pd.DataFrame:
  df = meta_df.copy()
  df["ehr_row_count_raw"] = df["has_ehr"] * 10.0
  df["ehr_density_raw"] = df["has_ehr"] * 5.0
  df["ehr_avg_measurement_frequency_raw"] = df["has_ehr"] * 2.0
  df["ehr_abnormality_raw"] = df["has_ehr"] * 1.0
  df["ehr_recent_abnormality_raw"] = df["has_ehr"] * 1.0
  df["ehr_volatility_raw"] = df["has_ehr"] * 0.5
  df["ehr_trend_raw"] = df["has_ehr"] * 0.1
  return df


# =============================================================================
# 5. Feature Normalization
# =============================================================================


@dataclass
class TransformSpec:
  col: str
  lo: float
  hi: float
  log1p: bool


class RobustFeatureNormalizer:

  def __init__(self) -> None:
    self.specs: Dict[str, TransformSpec] = {}

  def fit_column(
      self,
      df_val: pd.DataFrame,
      raw_col: str,
      out_col: str,
      log1p: bool = True,
  ) -> None:
    x = (
        pd.to_numeric(df_val[raw_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    if log1p:
      x = np.log1p(np.maximum(x, 0.0))
    lo, hi = np.nanpercentile(x, [5, 95]) if x.size else (0.0, 1.0)
    if not np.isfinite(lo):
      lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
      hi = lo + 1.0
    self.specs[out_col] = TransformSpec(
        col=raw_col, lo=float(lo), hi=float(hi), log1p=bool(log1p)
    )

  def transform_column(self, df: pd.DataFrame, out_col: str) -> np.ndarray:
    spec = self.specs[out_col]
    x = (
        pd.to_numeric(df[spec.col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    if spec.log1p:
      x = np.log1p(np.maximum(x, 0.0))
    score = (np.clip(x, spec.lo, spec.hi) - spec.lo) / max(
        spec.hi - spec.lo, 1e-12
    )
    return np.clip(score, 0.0, 1.0)


def apply_normalization(
    meta_tr: pd.DataFrame, meta_te: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  norm = RobustFeatureNormalizer()

  tr_age_median = meta_tr["age_raw"].median()
  meta_tr["age_raw"] = meta_tr["age_raw"].fillna(tr_age_median)
  meta_te["age_raw"] = meta_te["age_raw"].fillna(tr_age_median)

  specs = [
      ("age_raw", "age_score", False),
      ("ps_len_raw", "ps_length_score", True),
      ("ps_lex_raw", "ps_lexical_score", True),
      ("ps_ttr_raw", "ps_ttr_score", False),
      ("rr_len_raw", "rr_length_score", True),
      ("rr_lex_raw", "rr_lexical_score", True),
      ("rr_ttr_raw", "rr_ttr_score", False),
      ("ehr_row_count_raw", "ehr_row_count_score", True),
      ("ehr_density_raw", "ehr_density_score", True),
      ("ehr_avg_measurement_frequency_raw", "ehr_avg_measurement_frequency_score", True),
      ("ehr_abnormality_raw", "ehr_abnormality_score", True),
      ("ehr_recent_abnormality_raw", "ehr_recent_abnormality_score", True),
      ("ehr_volatility_raw", "ehr_volatility_score", True),
      ("ehr_trend_raw", "ehr_trend_score", True),
      ("ps_problem_count_raw", "ps_problem_count_score", True),
      ("ps_problem_density_raw", "ps_problem_burden_score", False),
      ("rr_abnormal_count_raw", "rr_abnormal_count_score", True),
      ("rr_abnormal_density_raw", "rr_abnormality_score", False),
      ("rr_reassuring_count_raw", "rr_reassuring_count_score", True),
      ("rr_reassuring_density_raw", "rr_reassuring_score", False),
  ]
  for raw_col, out_col, log1p in specs:
    if raw_col not in meta_tr.columns:
      meta_tr[raw_col] = 0.0
    if raw_col not in meta_te.columns:
      meta_te[raw_col] = 0.0
    norm.fit_column(meta_tr, raw_col, out_col, log1p=log1p)
    meta_tr[out_col] = norm.transform_column(meta_tr, out_col)
    meta_te[out_col] = norm.transform_column(meta_te, out_col)

  meta_tr["ehr_acuity_score"] = meta_tr["ehr_density_score"]
  meta_te["ehr_acuity_score"] = meta_te["ehr_density_score"]
  return meta_tr, meta_te


# =============================================================================
# 6. Model Predictions Extractor & Calibrator
# =============================================================================


def load_candidate_predictions(
    results_base: Path,
    task: str,
    split: str,
    model_name: str,
    combinations: List[Dict[str, str]],
    prob_key: str,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
  model_id = MODEL_IDS[model_name]
  label_map = meta_df.set_index("stay_id")["label"].to_dict()
  rows = []

  fallbacks = [
      prob_key,
      "mortality_probability",
      "los_probability",
      "probability",
      "prob",
  ]

  for combo in combinations:
    prompt = combo["prompt"]
    setup_dir = combo["setup_dir"]
    base_path = results_base / task / split / prompt / setup_dir / model_id

    candidate_files = [
        f"compliance_{setup_dir}_results.json",
        f"communication_{setup_dir}_results.json",
        f"{setup_dir}_results.json",
    ]

    records = []
    for file_name in candidate_files:
      path = base_path / file_name
      if path.exists():
        with open(path, "r", encoding="utf-8") as f:
          data = json.load(f)
          records = data if isinstance(data, list) else data.get("results", [])
          break

    if not records:
      print(f"WARNING: Missing predictions JSON at {base_path}")
      continue

    full_setup = setup_name(model_name, combo)
    for r in records:
      stay_id = str(r.get("stay_id"))
      if stay_id not in label_map:
        continue

      p_dict = (
          r.get("predictions", {})
          if isinstance(r.get("predictions"), dict)
          else {}
      )

      prob = None
      for fb in fallbacks:
        val = p_dict.get(fb, r.get(fb))
        if val is not None:
          try:
            parsed_prob = float(val)
            if not np.isnan(parsed_prob):
              prob = parsed_prob
              break
          except (TypeError, ValueError):
            pass

      if prob is not None:
        rows.append({
            "stay_id": stay_id,
            "Setup": full_setup,
            "raw_prob": prob,
            "label": label_map[stay_id],
        })

  return pd.DataFrame(rows)


def fit_and_apply_isotonic(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    combinations: List[Dict[str, str]],
    model_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  df_tr, df_te = df_train.copy(), df_test.copy()
  df_tr["prob"] = np.nan
  df_te["prob"] = np.nan

  for combo in combinations:
    s_name = setup_name(model_name, combo)
    tr_mask = df_tr["Setup"] == s_name
    te_mask = df_te["Setup"] == s_name

    x_tr = df_tr.loc[tr_mask, "raw_prob"].to_numpy(dtype=float)
    y_tr = df_tr.loc[tr_mask, "label"].to_numpy(dtype=int)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    if len(x_tr) > 5 and np.unique(y_tr).size > 1:
      iso.fit(x_tr, y_tr)
      df_tr.loc[tr_mask, "prob"] = iso.predict(x_tr)
      if te_mask.any():
        x_te = df_te.loc[te_mask, "raw_prob"].to_numpy(dtype=float)
        df_te.loc[te_mask, "prob"] = iso.predict(x_te)
    else:
      df_tr.loc[tr_mask, "prob"] = x_tr
      if te_mask.any():
        df_te.loc[te_mask, "prob"] = df_te.loc[te_mask, "raw_prob"]

  df_tr["prob"] = df_tr["prob"].fillna(0.5).clip(0, 1)
  df_te["prob"] = df_te["prob"].fillna(0.5).clip(0, 1)
  return df_tr, df_te


def add_prediction_derived_features(
    df_prob: pd.DataFrame,
    meta: pd.DataFrame,
    model_name: str,
    combinations: List[Dict[str, str]],
) -> pd.DataFrame:
  expected = expected_setup_names(model_name, combinations)
  piv = df_prob.pivot(index="stay_id", columns="Setup", values="prob")
  for col in expected:
    if col not in piv.columns:
      piv[col] = np.nan

  out = pd.DataFrame(index=piv.index)
  all_probs = piv[expected].astype(float)
  out["setup_prob_std_score"] = all_probs.std(axis=1).fillna(0.0)
  out = out.reset_index()
  return meta.merge(out, on="stay_id", how="left").fillna(0.0)


# =============================================================================
# 7. Weight Generation & Feature Families
# =============================================================================


def generate_grid_weights(steps: int, num_vars: int) -> np.ndarray:
  combos = []

  def rec(prefix: List[int], remaining: int, k: int) -> None:
    if k == 1:
      combos.append(prefix + [remaining])
      return
    for i in range(remaining + 1):
      rec(prefix + [i], remaining - i, k - 1)

  rec([], steps, num_vars)
  return np.asarray(combos, dtype=np.float32) / float(steps)


def generate_random_weights(n: int, d: int, seed: int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  chunks = []
  for alpha, frac in [(0.2, 0.35), (0.5, 0.30), (1.0, 0.25), (2.0, 0.10)]:
    m = max(1, int(round(n * frac)))
    chunks.append(rng.dirichlet(np.full(d, alpha), size=m).astype(np.float32))
  return np.vstack(chunks)[:n]


def make_weight_candidates(
    feature_columns: Sequence[str],
    resolution_steps: int,
    max_grid_dim: int,
    random_weight_samples: int,
    seed: int = 42,
) -> Tuple[np.ndarray, str]:
  d = len(feature_columns)
  deterministic = [
      np.ones(d, dtype=np.float32) / float(d),
      np.eye(d, dtype=np.float32),
  ]
  if d <= max_grid_dim:
    base = generate_grid_weights(resolution_steps, d)
    mode = f"grid_{resolution_steps}_steps"
  else:
    base = generate_random_weights(random_weight_samples, d, seed=seed)
    mode = f"random_dirichlet_{random_weight_samples}"
  arr = np.vstack([base] + deterministic).astype(np.float32)
  _, idx = np.unique(np.round(arr, 6), axis=0, return_index=True)
  arr = arr[np.sort(idx)]
  return arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-12), mode


def available_feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
  input7 = INPUT7_COLUMNS
  input7_plus_demographics = input7 + ["age_score", "is_female", "is_male"]
  input7_plus_uncertainty = input7 + ["setup_prob_std_score"]
  cognitive_acuity = input7_plus_uncertainty + [
      "ps_ttr_score",
      "rr_ttr_score",
      "ehr_abnormality_score",
  ]
  full_autopsy = cognitive_acuity + ["ehr_avg_measurement_frequency_score"]

  families = {
      "input7": input7,
      "input7_plus_demographics": input7_plus_demographics,
      "input7_plus_uncertainty": input7_plus_uncertainty,
      "cognitive_acuity": cognitive_acuity,
      "full_autopsy": full_autopsy,
  }

  cleaned = {}
  for name, cols in families.items():
    present = [c for c in cols if c in df.columns]
    if present:
      cleaned[name] = list(dict.fromkeys(present))
  return cleaned


# =============================================================================
# 8. Dense Matrices & GPU Escalation Engine
# =============================================================================


@dataclass
class SplitMatrices:
  stay_ids: np.ndarray
  labels: np.ndarray
  probs: np.ndarray
  setup_names: List[str]
  feature_frame: pd.DataFrame


def build_matrices(
    df: pd.DataFrame, combinations: List[Dict[str, str]], model_name: str
) -> SplitMatrices:
  setups = expected_setup_names(model_name, combinations)
  piv = df.pivot(index="stay_id", columns="Setup", values="prob")
  for s in setups:
    if s not in piv.columns:
      piv[s] = 0.5
  piv = piv[setups]
  unique = df.drop_duplicates("stay_id").set_index("stay_id").loc[piv.index]
  return SplitMatrices(
      stay_ids=piv.index.astype(str).to_numpy(),
      labels=unique["label"].astype(int).to_numpy(),
      probs=np.clip(
          np.nan_to_num(piv.to_numpy(dtype=np.float32), nan=0.5), 0.0, 1.0
      ),
      setup_names=setups,
      feature_frame=unique.reset_index(),
  )


def generate_policy_indices(q_bins: int, n_setups: int) -> np.ndarray:
  out = []

  def rec(prefix: List[int], start: int):
    if len(prefix) == q_bins:
      out.append(prefix.copy())
      return
    for s in range(start, n_setups):
      prefix.append(s)
      rec(prefix, s)
      prefix.pop()

  rec([], 0)
  return np.asarray(out, dtype=np.int16)


def make_rank_bins(scores: np.ndarray, q_bins: int) -> np.ndarray:
  n = len(scores)
  order = np.argsort(scores, kind="mergesort")
  ranks = np.empty(n, dtype=np.int64)
  ranks[order] = np.arange(n, dtype=np.int64)
  bins = np.floor(ranks * q_bins / float(n)).astype(np.int64)
  return np.clip(bins, 0, q_bins - 1)


def apply_threshold_bins(
    scores: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
  interior = thresholds[1:-1]
  return np.clip(
      np.searchsorted(interior, scores, side="right"), 0, len(thresholds) - 2
  ).astype(np.int64)


def torch_batched_auc_no_ties(
    y: "torch.Tensor", p: "torch.Tensor"
) -> "torch.Tensor":
  B, L = y.shape
  n_pos = y.sum(dim=1)
  n_neg = float(L) - n_pos
  valid = (n_pos > 0) & (n_neg > 0)
  order = torch.argsort(p, dim=1)
  y_sorted = torch.gather(y, 1, order)
  ranks = torch.arange(1, L + 1, device=y.device, dtype=torch.float32).view(
      1, L
  )
  rank_sum = (y_sorted * ranks).sum(dim=1)
  auc = (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / torch.clamp(
      n_pos * n_neg, min=1.0
  )
  return torch.where(valid, auc, torch.full_like(auc, -1.0))


def gpu_screen_topk(
    matrices: SplitMatrices,
    features: np.ndarray,
    weights: np.ndarray,
    min_bins: int,
    max_bins: int,
    batch_size: int,
    topk: int,
    device_name: str,
    policy_batch_size: int,
) -> List[Tuple[float, int, int, Tuple[int, ...]]]:

  if torch is None:
    raise RuntimeError(
        f"Torch required for fast_gpu_topk. Error: {_TORCH_IMPORT_ERROR}"
    )
  device = torch.device(device_name)
  y_t = torch.tensor(matrices.labels.astype(np.float32), device=device)
  p_t = torch.tensor(matrices.probs.astype(np.float32), device=device)
  x_t = torch.tensor(features.astype(np.float32), device=device)
  w_all = torch.tensor(weights.astype(np.float32), device=device)

  n, s_count = matrices.probs.shape
  heap = []

  for start in tqdm(
      range(0, weights.shape[0], batch_size),
      desc="GPU Vector Scan",
      leave=False,
  ):
    end = min(start + batch_size, weights.shape[0])
    wb = w_all[start:end].T
    scores = x_t @ wb
    order = torch.argsort(scores, dim=0).T.contiguous()
    B = int(order.shape[0])

    y_ordered = y_t[order]
    p_ordered = torch.stack(
        [p_t[:, s][order] for s in range(s_count)], dim=0
    )

    for q in range(min_bins, max_bins + 1):
      policies_np = generate_policy_indices(q, s_count)
      for p_start in range(0, len(policies_np), policy_batch_size):
        p_end = min(p_start + policy_batch_size, len(policies_np))
        pol_np = policies_np[p_start:p_end]
        Pc = int(pol_np.shape[0])

        comp = torch.empty((B, Pc, n), device=device, dtype=torch.float32)
        for b in range(q):
          a = int(math.floor(b * n / float(q)))
          z = int(math.floor((b + 1) * n / float(q)))
          selected = pol_np[:, b]
          for s_idx in range(s_count):
            pol_mask = selected == s_idx
            if not np.any(pol_mask):
              continue
            pol_cols = torch.tensor(
                np.where(pol_mask)[0], device=device, dtype=torch.long
            )
            comp[:, pol_cols, a:z] = p_ordered[s_idx, :, None, a:z]

        comp_flat = comp.reshape(B * Pc, n)
        y_flat = y_ordered[:, None, :].expand(B, Pc, n).reshape(B * Pc, n)
        auc_flat = torch_batched_auc_no_ties(y_flat, comp_flat)

        vals, flat_idxs = torch.topk(
            auc_flat, k=min(topk, int(auc_flat.numel()))
        )
        for val, flat_i in zip(vals.cpu().numpy(), flat_idxs.cpu().numpy()):
          if val > -0.5:
            wi_local, pi_local = int(flat_i) // Pc, int(flat_i) % Pc
            item = (
                float(val),
                start + wi_local,
                q,
                tuple(int(x) for x in pol_np[pi_local]),
            )
            if len(heap) < topk:
              heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
              heapq.heapreplace(heap, item)

        del comp, comp_flat, y_flat, auc_flat
        if device.type == "cuda":
          torch.cuda.empty_cache()

  return sorted(heap, key=lambda x: x[0], reverse=True)


# =============================================================================
# 9. Exact CPU Evaluation & Application
# =============================================================================


@dataclass
class RouterConfig:
  feature_family: str
  feature_columns: List[str]
  weight_mode: str
  weights: List[float]
  q_bins: int
  winners: List[str]
  thresholds: List[float]
  opt_auroc: float


def build_exact_router(
    matrices: SplitMatrices,
    feature_family: str,
    feature_columns: Sequence[str],
    weight_mode: str,
    weights: np.ndarray,
    q_bins: int,
    policy_tuple: Tuple[int, ...],
) -> RouterConfig:
  feats = matrices.feature_frame[list(feature_columns)].to_numpy(dtype=float)
  scores = feats @ weights.astype(float)
  bins = make_rank_bins(scores, q_bins)

  comp = np.empty(len(scores), dtype=float)
  winners = []
  for b, s_idx in enumerate(policy_tuple):
    comp[bins == b] = matrices.probs[bins == b, s_idx]
    winners.append(matrices.setup_names[s_idx])

  auc = safe_auc(matrices.labels, comp)
  thresholds = np.quantile(
      scores, [i / float(q_bins) for i in range(q_bins + 1)]
  ).tolist()

  return RouterConfig(
      feature_family=feature_family,
      feature_columns=list(feature_columns),
      weight_mode=weight_mode,
      weights=[float(x) for x in weights],
      q_bins=q_bins,
      winners=winners,
      thresholds=thresholds,
      opt_auroc=float(auc),
  )


def apply_router(
    matrices: SplitMatrices, router: RouterConfig, use_rank_bins: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns: (routed_probabilities, scores, chosen_setup_names)"""
  feats = matrices.feature_frame[router.feature_columns].to_numpy(dtype=float)
  scores = feats @ np.asarray(router.weights, dtype=float)
  bins = (
      make_rank_bins(scores, router.q_bins)
      if use_rank_bins
      else apply_threshold_bins(scores, np.asarray(router.thresholds))
  )

  out = np.zeros(len(scores), dtype=float)
  chosen_setups = np.empty(len(scores), dtype=object)
  setup_to_idx = {name: i for i, name in enumerate(matrices.setup_names)}

  for b in range(router.q_bins):
    idx = setup_to_idx[router.winners[b]]
    out[bins == b] = matrices.probs[bins == b, idx]
    chosen_setups[bins == b] = router.winners[b]

  return out, scores, chosen_setups


# =============================================================================
# 10. Baseline Logic
# =============================================================================


def run_random_routing(
    matrices: SplitMatrices, seed: int, n_bootstraps: int, ci_level: float
) -> Dict[str, Any]:
  rng = np.random.default_rng(seed)
  n, s_count = matrices.probs.shape
  assigned_indices = rng.integers(0, s_count, size=n)
  routed_probs = matrices.probs[np.arange(n), assigned_indices]
  metrics, _ = evaluate_with_bootstrap(
      matrices.labels, routed_probs, n_bootstraps, ci_level, seed
  )
  return metrics


def run_heuristic_routing(
    matrices: SplitMatrices, n_bootstraps: int, ci_level: float, seed: int
) -> Dict[str, Any]:
  n, s_count = matrices.probs.shape
  df_feat = matrices.feature_frame
  tier = np.zeros(n, dtype=int)
  if "has_ehr" in df_feat.columns:
    tier += (df_feat["has_ehr"] > 0).astype(int)
  if "ps_len_raw" in df_feat.columns:
    tier += (df_feat["ps_len_raw"] > df_feat["ps_len_raw"].median()).astype(int)

  tier = np.clip(tier, 0, s_count - 1)
  routed_probs = matrices.probs[np.arange(n), tier]
  metrics, _ = evaluate_with_bootstrap(
      matrices.labels, routed_probs, n_bootstraps, ci_level, seed
  )
  return metrics


# =============================================================================
# 11. Argparse & Main Loop
# =============================================================================


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Unified PCAR Escalation Router"
  )
  # Structural Config
  parser.add_argument("--task", choices=["Mortality", "LOS"], required=True)
  parser.add_argument(
      "--mode",
      choices=["oracle", "regular", "prompt-only", "random", "heuristics"],
      required=True,
  )
  parser.add_argument(
      "--prompt-only-arch",
      choices=["SingleAgent", "MajorityVote"],
      default="SingleAgent",
  )
  parser.add_argument("--models", nargs="+", default=["Huatuo", "Qwen"])
  parser.add_argument(
      "--results-base", type=Path, default=DEFAULT_RESULTS_BASE
  )
  parser.add_argument(
      "--outdir", type=Path, default=Path("newer_unified_router_outputs")
  )

  # Optimizer Parameters
  parser.add_argument(
      "--feature-families", nargs="+", default=[
            "cognitive_acuity"
        ]
  )
#   parser.add_argument(
#       "--feature-families", nargs="+", default=[
#             "input7", 
#             "input7_plus_demographics", 
#             "input7_plus_uncertainty", 
#             "cognitive_acuity", 
#             "full_autopsy"
#         ]
#  )
  parser.add_argument(
      "--optimizer",
      choices=["fast_gpu_topk", "exact_cpu"],
      default="fast_gpu_topk",
  )
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--resolution-steps", type=int, default=10)
  parser.add_argument("--max-grid-dim", type=int, default=8)
  parser.add_argument("--random-weight-samples", type=int, default=25000)
  parser.add_argument("--gpu-batch-size", type=int, default=512)
  parser.add_argument("--policy-batch-size", type=int, default=32)
  parser.add_argument("--topk-exact", type=int, default=1024)
  parser.add_argument("--min-bins", type=int, default=3)
  parser.add_argument("--max-bins", type=int, default=11)
  parser.add_argument("--seed", type=int, default=42)

  # Bootstrapping CI Parameters
  parser.add_argument("--n-bootstraps", type=int, default=1000)
  parser.add_argument("--label-ci-level", type=float, default=0.95)

  # Export Flags
  parser.add_argument(
      "--save-feature-tables", action="store_true", default=True
  )
  parser.add_argument(
      "--save-optimization-candidates", action="store_true", default=True
  )

  return parser.parse_args()


def main():
  args = parse_args()
  args.outdir.mkdir(parents=True, exist_ok=True)
  task_cfg = DATASET_PATHS[args.task]
  combinations = get_target_combinations(
      args.task, args.mode, args.prompt_only_arch
  )

  print("=" * 80)
  print(f"RUNNING UNIFIED ROUTER | Task: {args.task} | Mode: {args.mode}")
  print("=" * 80)

  # 1. Load Ground Truth Metadata
  meta_tr = read_jsonl_metadata(task_cfg["Train"], task_cfg["label_key"])
  meta_te = read_jsonl_metadata(task_cfg["Test"], task_cfg["label_key"])
  meta_tr = ehr_patient_stats_fallback(meta_tr)
  meta_te = ehr_patient_stats_fallback(meta_te)
  meta_tr, meta_te = apply_normalization(meta_tr, meta_te)

  for model_name in args.models:
    print(f"\n--- MODEL: {model_name} ---")

    # 2. Load Predictions & Calibrate
    df_tr_raw = load_candidate_predictions(
        args.results_base,
        args.task,
        "Train",
        model_name,
        combinations,
        task_cfg["prob_key"],
        meta_tr,
    )
    df_te_raw = load_candidate_predictions(
        args.results_base,
        args.task,
        "Test",
        model_name,
        combinations,
        task_cfg["prob_key"],
        meta_te,
    )

    if df_te_raw.empty:
      print(f"Skipping {model_name}, no Test predictions found.")
      continue

    if args.mode == "oracle":
      iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
      df_te_cal = df_te_raw.copy()
      for combo in combinations:
        s_name = setup_name(model_name, combo)
        mask = df_te_cal["Setup"] == s_name
        if mask.any():
          df_te_cal.loc[mask, "prob"] = iso.fit_transform(
              df_te_cal.loc[mask, "raw_prob"], df_te_cal.loc[mask, "label"]
          )
      meta_te_feat = add_prediction_derived_features(
          df_te_cal, meta_te, model_name, combinations
      )
      mat_opt = build_matrices(
          df_te_cal.merge(meta_te_feat), combinations, model_name
      )
      mat_eval = mat_opt
    else:
      df_tr_cal, df_te_cal = fit_and_apply_isotonic(
          df_tr_raw, df_te_raw, combinations, model_name
      )
      meta_tr_feat = add_prediction_derived_features(
          df_tr_cal, meta_tr, model_name, combinations
      )
      meta_te_feat = add_prediction_derived_features(
          df_te_cal, meta_te, model_name, combinations
      )
      mat_opt = build_matrices(
          df_tr_cal.merge(meta_tr_feat), combinations, model_name
      )
      mat_eval = build_matrices(
          df_te_cal.merge(meta_te_feat), combinations, model_name
      )

    if args.save_feature_tables:
      eval_csv = (
          args.outdir / f"{model_name}_{args.task}_{args.mode}_eval_features.csv"
      )
      mat_eval.feature_frame.to_csv(eval_csv, index=False)
      if args.mode != "oracle":
        opt_csv = (
            args.outdir
            / f"{model_name}_{args.task}_{args.mode}_opt_features.csv"
        )
        mat_opt.feature_frame.to_csv(opt_csv, index=False)

    # ---------------------------------------------------------------------
    # Print Initial Candidate Baselines (RAW vs CALIBRATED on Test Set)
    # ---------------------------------------------------------------------
    df_te_raw_copy = df_te_raw.copy()
    df_te_raw_copy["prob"] = df_te_raw_copy["raw_prob"]

    merged_raw = df_te_raw_copy.drop(columns=["label"]).merge(
        meta_te_feat, on="stay_id", how="left"
    )
    mat_eval_raw = build_matrices(merged_raw, combinations, model_name)

    best_base_auc = -1.0
    best_base_dist = None
    best_base_probs = None

    print("\n Candidate Baselines (Evaluated on Test):")
    print(" [RAW / UNCALIBRATED PREDICTIONS]")
    for i, s_name in enumerate(mat_eval_raw.setup_names):
      base_m_raw, _ = evaluate_with_bootstrap(
          mat_eval_raw.labels,
          mat_eval_raw.probs[:, i],
          0,
          args.label_ci_level,
          args.seed,
      )
      print_formatted_metrics(f"  [{s_name:<32s} (RAW)]", base_m_raw)

    print(" [CALIBRATED PREDICTIONS]")
    for i, s_name in enumerate(mat_eval.setup_names):
      base_m_cal, base_dist = evaluate_with_bootstrap(
          mat_eval.labels,
          mat_eval.probs[:, i],
          args.n_bootstraps,
          args.label_ci_level,
          args.seed,
      )
      print_formatted_metrics(f"  [{s_name:<32s} (CAL)]", base_m_cal)

      current_auc = base_m_cal.get("AUROC", (0, 0, 0))[0]
      if current_auc > best_base_auc:
        best_base_auc = current_auc
        best_base_dist = base_dist
        best_base_probs = mat_eval.probs[:, i]
    print("-" * 60)

    # 3. Execution by Mode
    if args.mode in ["oracle", "regular", "prompt-only"]:
      all_families = available_feature_families(mat_opt.feature_frame)
      best_global_router = None
      all_evaluated_candidates = []

      for fam_name in args.feature_families:
        if fam_name not in all_families:
          continue
        cols = all_families[fam_name]
        weights, weight_mode = make_weight_candidates(
            cols,
            args.resolution_steps,
            args.max_grid_dim,
            args.random_weight_samples,
            args.seed,
        )

        print(
            f"\n Optimizing {fam_name} | cols={len(cols)} |"
            f" weights={len(weights)} | q={args.min_bins}-{args.max_bins}"
        )
        top_candidates = gpu_screen_topk(
            mat_opt,
            mat_opt.feature_frame[cols].to_numpy(),
            weights,
            args.min_bins,
            args.max_bins,
            args.gpu_batch_size,
            args.topk_exact,
            args.device,
            args.policy_batch_size,
        )

        best_fam_router = None
        for approx_auc, wi, q, pol_tuple in top_candidates:
          router = build_exact_router(
              mat_opt,
              fam_name,
              cols,
              weight_mode,
              weights[wi],
              q,
              pol_tuple,
          )

          if args.save_optimization_candidates:
            all_evaluated_candidates.append({
                "feature_family": fam_name,
                "q_bins": q,
                "opt_auroc": router.opt_auroc,
                "policy": str(pol_tuple),
                "winners": str(router.winners),
                "weights": str(router.weights),
                "thresholds": str(router.thresholds),
            })

          if (
              best_fam_router is None
              or router.opt_auroc > best_fam_router.opt_auroc
          ):
            best_fam_router = router

        if best_fam_router:
          use_rank = args.mode == "oracle"
          fam_eval_probs, _, _ = apply_router(
              mat_eval, best_fam_router, use_rank_bins=use_rank
          )
          fam_test_metrics, _ = evaluate_with_bootstrap(
              mat_eval.labels,
              fam_eval_probs,
              args.n_bootstraps,
              args.label_ci_level,
              args.seed,
          )

          print(f"  [Train Opt AUROC: {best_fam_router.opt_auroc:.4f}]")
          print_formatted_metrics(
              f"  -> Test Performance [{fam_name:<24s} |"
              f" Q={best_fam_router.q_bins}]",
              fam_test_metrics,
          )

          if (
              best_global_router is None
              or best_fam_router.opt_auroc > best_global_router.opt_auroc
          ):
            best_global_router = best_fam_router

      if args.save_optimization_candidates and all_evaluated_candidates:
        df_candidates = pd.DataFrame(all_evaluated_candidates).sort_values(
            by="opt_auroc", ascending=False
        )
        cand_csv = (
            args.outdir
            / f"{model_name}_{args.task}_{args.mode}_optimization_candidates.csv"
        )
        df_candidates.to_csv(cand_csv, index=False)

      if best_global_router:
        print("\n" + "=" * 60)
        use_rank = args.mode == "oracle"
        eval_probs, eval_scores, eval_setups = apply_router(
            mat_eval, best_global_router, use_rank_bins=use_rank
        )
        metrics, router_dist = evaluate_with_bootstrap(
            mat_eval.labels,
            eval_probs,
            args.n_bootstraps,
            args.label_ci_level,
            args.seed,
        )

        print_formatted_metrics(
            f" -> Best Overall Router [{best_global_router.feature_family} |"
            f" Q={best_global_router.q_bins}]",
            metrics,
        )

        # --- Statistical Significance Testing ---
        if best_base_probs is not None:
          # 1. Paired T-Test (Bootstrap AUROC)
          if len(router_dist) > 0 and len(best_base_dist) > 0:
            stat, p_val_ttest = ttest_rel(best_base_dist, router_dist)
            print(
                " -> Paired t-test p-value (AUROC Bootstrap):"
                f" {p_val_ttest:.4e}"
            )

          # 2. DeLong's Test (AUROC)
          p_val_delong = delong_roc_test(
              mat_eval.labels, best_base_probs, eval_probs
          )
          print(
              f" -> DeLong's test p-value (AUROC Difference): {p_val_delong:.4e}"
          )

          # 3. Paired Bootstrap Test (AUPRC)
          auprc_diff, p_val_auprc = paired_bootstrap_auprc_test(
              mat_eval.labels,
              best_base_probs,
              eval_probs,
              n_bootstraps=args.n_bootstraps,
              seed=args.seed,
          )
          print(
              f" -> Paired Bootstrap p-value (AUPRC Difference: {auprc_diff:+.4f}):"
              f" {p_val_auprc:.4e}"
          )

        # --- Save Routed Outputs to JSON ---
        out_json_path = (
            args.outdir / f"{model_name}_{args.task}_{args.mode}_routed_outputs.json"
        )
        records_to_save = []
        for sid, p, s, l in zip(
            mat_eval.stay_ids, eval_probs, eval_setups, mat_eval.labels
        ):
          records_to_save.append({
              "stay_id": str(sid),
              "label": int(l),
              "routed_prob": float(p),
              "chosen_setup": str(s),
          })
        with open(out_json_path, "w", encoding="utf-8") as out_f:
          json.dump(records_to_save, out_f, indent=4)
        print(f" -> Exported routed predictions to: {out_json_path.name}")

    elif args.mode == "random":
      metrics = run_random_routing(
          mat_eval, args.seed, args.n_bootstraps, args.label_ci_level
      )
      print_formatted_metrics(" -> Random Routing Baseline", metrics)

    elif args.mode == "heuristics":
      metrics = run_heuristic_routing(
          mat_eval, args.n_bootstraps, args.label_ci_level, args.seed
      )
      print_formatted_metrics(" -> Heuristics Routing Baseline", metrics)

  print("\nExecution Completed Successfully.")


if __name__ == "__main__":
  main()
