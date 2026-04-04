"""
AML Supervised Model Training

trains multiple classifiers to detect money laundering.
select the most stable one using monte carlo cross validation.
and optionally combine it with anomaly scores for better results.(fusion)
"""

import gc
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression as LR

import shap
import mlflow
import dagshub
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#paths
INPUT_DIR =Path("data/processed_with_anomaly")
MODEL_DIR = Path("models/supervised")
SHAP_DIR =Path("models/supervised/shap")

#cofings
RANDOM_STATE = 42
TARGET_COL = "Is Laundering"
TIME_COL = "Timestamp"

#business requirement: catch as much fraud as possible while keeping 
#false alarms under 1% of all legit transactions
FPR_THRESHOLD = 0.01

#MCCV instead of k-fold to handle extreme imbalance.
MCCV_ITERATIONS = 5
MIN_FRAUD_HARD = 1
MIN_FRAUD_WARN = 10 
MCCV_VAL_TOLERANCE = 0.02
TRAIN_SAMPLE_SIZE = 1_500_000
VAL_SAMPLE_SIZE = 500_000
TEST_SAMPLE_SIZE = 750_000

ZERO_IMPUTE_FEATURES = [
    'burst_score_1h', 'burst_count_24h',
    'txn_in_hour',
    'toxic_corridor_count_28d', 'toxic_corridor_volume_28d',
    'txn_count_28d', 'txn_count_total',
    'total_amount_paid_28d', 'total_amount_received_28d',
    'amount_paid_last_100',
    'flag_high_burst', 'flag_large_gap',
    'flag_extreme_consistency', 'flag_high_concentration',
    'flag_heavy_structuring', 'anomaly_cascade_score',
    'cascade_frequency_28d',
]

#setup
def setup():
    """create output directories."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dirs ready: {MODEL_DIR}, {SHAP_DIR}")


#feature selection
def get_features(schema: dict) -> List[str]:
    """
    look at the file schema and pick out which columns we can use as features.
    we skip the target column and anything that's not a number.
    """
    numeric_types = (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64, pl.Boolean,
    )

    features = [col for col, dtype in schema.items() if isinstance(dtype, numeric_types) and col != TARGET_COL]
    logger.info(f"Found {len(features)} numeric features")

    #warn when no time col is present random spliit dont reflect
    #how fraud patterns shift over time
    if TIME_COL and TIME_COL not in schema:
        logger.warning(f"No '{TIME_COL}' column found. train/val/test splits will be random.")

    if 'anomaly_score' not in features:
        logger.warning('No anomaly_score found in features')
    return features


def remove_low_varience_features(X: np.ndarray, features: List[str], thresh: float=0.001) -> Tuple[np.ndarray, List[str]]:
    """
    remove features with near zero variance across the training set
    a feature that barely changes across 1.5M rows contributes nothing to model
    """
    var = np.var(X, axis=0)
    mask = var > thresh
    
    k = [f for f, keep in zip(features, mask) if keep]
    r = [f for f, keep in zip(features, mask) if not keep]

    if r:
        logger.warning(f"Removed {len(r)} low-variance features: {r}")
    logger.info(f"kept {len(k)}/{len(features)} features after variance filter")

    return X[:, mask], k


# data loading
def load_data(split: str, features: List[str], max_rows: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    memory efficient data loadingg with Polars.
    stratified sampling ensure all the fraud cases are kept
    """

    path = INPUT_DIR / f"{split}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Can't find {path} - run anomaly detection file first")
    
    cols_to_load = features + [TARGET_COL]
    if TIME_COL:
        cols_to_load = list(set(cols_to_load + [TIME_COL]))
    
    #first, just count rows without loading data
    n_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    
    if max_rows and n_rows > max_rows:
        #sample evenly across the file instead of random
        lazy = pl.scan_parquet(path).select(cols_to_load)

        fraud_df =lazy.filter(pl.col(TARGET_COL)== 1).collect()
        n_legit = min(max_rows - len(fraud_df), n_rows - len(fraud_df))
        rng = np.random.RandomState(RANDOM_STATE)
        #sample legit rows indices without materialising all legit rows

        n_legit_total = n_rows - len(fraud_df)
        legit_idx = sorted(rng.choice(n_legit_total, n_legit, replace=False).tolist())
        legit_df = (
            lazy.filter(pl.col(TARGET_COL) == 0)
            .with_row_index('_row_idx')
            .filter(pl.col('_row_idx').is_in(legit_idx))
            .drop('_row_idx')
            .collect()
        )
      
        df = pl.concat([fraud_df, legit_df]).sample(fraction=1.0, shuffle=True, seed=RANDOM_STATE)
        logger.info(
            f"Loaded {split}: {len(df):,} rows "
            f"(all {len(fraud_df):,} fraud + {n_legit:,} legit sampled)"
        )
        del fraud_df, legit_df
        gc.collect()

    else:
        logger.info(f"Loading {split}: all {n_rows:,} rows")
        df= pl.scan_parquet(path).select(cols_to_load).collect()

    logger.info(f" Size in RAM: {df.estimated_size() / 1024**2:.1f} MB")

    timestamps = df[TIME_COL].cast(pl.Int64).to_numpy() if TIME_COL in df.columns else None
    y= df[TARGET_COL].to_numpy().astype(np.int32)
    X = df.select(features).to_numpy().astype(np.float32)

    del df
    gc.collect()

    # processing one column at a time 
    for col_idx, col_name in enumerate(features):
        row_indices = np.where(np.isnan(X[:, col_idx]))[0]
        if len(row_indices) == 0:
            continue
        
        if col_name in ZERO_IMPUTE_FEATURES:
            X[row_indices, col_idx] = 0.0
        else:
            col_mean= float(np.nanmean(X[:, col_idx]))
            X[row_indices, col_idx] = col_mean if not np.isnan(col_mean) else 0.0


    n_fraud= int(y.sum())

    if n_fraud <= MIN_FRAUD_HARD:
        raise ValueError(
            f"{split}: only {n_fraud} fraud cases after loading. increase sample size or check data."
        )

    if n_fraud < MIN_FRAUD_WARN:
        logger.warning(f"{split}: only {n_fraud} fraud cases - metrics may be unstable")

    logger.info(f"{split}: {len(y):,} rows | fraud={n_fraud:,} | legit={len(y)-n_fraud:,}")
    return X, y, timestamps


#metrics
def recall_at_target_fpr(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    linear interpolation to find recall at target FPR
    """
    return float(np.interp(FPR_THRESHOLD, fpr, tpr))


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    calculate metrics that matter for fraud detection:
    - pr-auc: best single number for imbalanced data
    - eoc-auc: standard comparison metric
    - recall at 1@ fpr: business requirement
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    r_at_fpr = recall_at_target_fpr(fpr, tpr)
    #find recall when FPR is closest to 1%
    idx = np.argmin(np.abs(fpr - FPR_THRESHOLD))
    
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(auc(recall, precision)),
        "recall_at_1pct_fpr": r_at_fpr,
        "fpr_at_threshold": float(fpr[idx]),
        "threshold_1pct_fpr": float(thresholds[idx]) if idx < len(thresholds) else 1.0,
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    find the decision threshold that maximises recall subject to fpr <= business constraint.
    this is the threshold save with the model without this the mode has no operational 
    decision boundary"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_mask = fpr <= FPR_THRESHOLD

    if not valid_mask.any():
        logger.warning('No threshold achieves FPR <= 1%. Defaulting to 0.5')
        return 0.5

    best_idx = np.argmax(tpr[valid_mask])
    optimal_thresh = float(thresholds[valid_mask][best_idx])

    logger.info(
        f"Optimal Threshold: {optimal_thresh:.4f} |"
        f"Recall: {tpr[valid_mask][best_idx]:.4f} |"
        f"FPR: {fpr[valid_mask][best_idx]:.4f}"
    )
    return optimal_thresh


def temporal_mccv_split(timestamps: np.ndarray, fold_i: int, n_folds: int=5,) -> Tuple[np.ndarray, np.ndarray]:
    """
    expanding walk forward split each fold trains on earlier time, validates on the immeditely following window.
    """
    sorted_idx = np.argsort(timestamps)
    n= len(sorted_idx)

    #larger validation set to catch temporal drift
    val_size = int(0.1*n)
    
    train_end = n -(n_folds * val_size)
    if train_end <= 0:
        raise ValueError(f"Not enough data for {n_folds} folds with val fraction")
    
    train_end = train_end + fold_i * val_size
    val_start = train_end
    val_end = val_start + val_size
    train_idx = sorted_idx[:train_end]
    val_idx = sorted_idx[val_start:val_end]
    return train_idx, val_idx


def mccv_evaluate(config: dict, X: np.ndarray, y: np.ndarray, timestamps: np.ndarray) -> Dict:
    """
    monte carlo cross validation for extreme imbalance
    Problem with k-fold: Fixed splits can have validation folds with 
    0-2 positive cases, making metrics unstable.
    """
    scores, thresholds = [], []
    logger.info(f" Running {MCCV_ITERATIONS} expanding walk forward folds...")

    for i in range(MCCV_ITERATIONS):
        train_idx, val_idx = temporal_mccv_split(timestamps, fold_i=i, n_folds=MCCV_ITERATIONS)

        n_train_fraud = int(y[train_idx].sum())
        n_val_fraud = int(y[val_idx].sum())

        logger.info(
        f"Fold {i}: train={len(train_idx):,} rows ({n_train_fraud} fraud) | "
        f"val={len(val_idx):,} rows ({n_val_fraud} fraud)"
        )
        if n_train_fraud < MIN_FRAUD_WARN or n_val_fraud < MIN_FRAUD_WARN:
            logger.warning(f" Fold {i}: too few fraud cases - skipping")
            continue
        try:
            model = build_model(config)
            model.fit(X[train_idx], y[train_idx])
            probs = model.predict_proba(X[val_idx])[:,1]
            metrics = calculate_metrics(y[val_idx], probs)

            scores.append(metrics["recall_at_1pct_fpr"])
            thresholds.append(metrics["threshold_1pct_fpr"])

            del model, probs
            gc.collect()
        except Exception as e:
            logger.warning(f" Fold {i} failed: {e}")
    
    if len(scores) < 3:
        raise ValueError(f"Only {len(scores)} valide folds - need at leadt 3")
    

    return {
        'mean_recall': float(np.mean(scores)),
        'std_recall': float(np.std(scores)),
        'mean_threshold': float(np.mean(thresholds)),
        'all_scores': scores,
        'n_valid_iterations': len(scores)
    }


def get_model_configs(imbalance_ratio: float) -> Dict[str, Dict]:
    """
    fixed configurations for imbalanced data
    """
    return {
    "XGBoost": {
        "class": XGBClassifier,
        "params": {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "scale_pos_weight": imbalance_ratio * 0.8, 
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "preprocessor": None
    },
    "LightGBM": {
        "class": lgb.LGBMClassifier,
        "params": {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "scale_pos_weight": imbalance_ratio * 0.6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        },
        "preprocessor": None
    },
    "RandomForest": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 10,
            "class_weight": "balanced",
            "min_samples_split": 50,
            "max_samples": 200_000,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "preprocessor": None
    },
    "LogisticRegression": {
        "class": LogisticRegression,
        "params": {
            "class_weight": "balanced",
            "max_iter": 1000,
            "C": 0.1,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "preprocessor": RobustScaler()
    }
}

from sklearn.base import clone
def build_model(config: dict):
    """wrap model in scaliing pipeline"""
    if config.get("preprocessor") is not None:
        return Pipeline([
            ("scaler", clone(config["preprocessor"])),
            ("clf", config["class"](**config["params"]))
        ])
    return config["class"](**config["params"])


def select_best_model(mccv_results: Dict[str, Dict]) -> Tuple[str, Dict]:
    """
    pick the model with the highest conservation score (mean - 2*std)
    this penalises models that occasionally spike on lucky foldes
    """
    valid_models = {
        name: res for name, res in mccv_results.items()
        if res.get("n_valid_iterations", 0) >= 3
    }
    
    if not valid_models:
        raise ValueError("No model produced 3+ valid MCCV folds")
    
    conservative = {name: res["mean_recall"] - 2 * res["std_recall"] for name, res in valid_models.items()}

   
    best_name = max(conservative, key=conservative.get)
    best_result = valid_models[best_name]
    
    logger.info(
        f"Best Model: {best_name} | "
        f"recall {best_result['mean_recall']:.4f} +- {best_result['std_recall']:.4f} | "
        f"conservative: {conservative[best_name]:.4f}"
    )
    return best_name, best_result


def train_model(model_config, X_train, y_train, X_val, y_val):
    """train on full training  data."""
    logger.info(f"Training on full dataset...")
    
    model = build_model(model_config)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:,1]
    val_metrics = calculate_metrics(y_val, val_probs)

    logger.info(f"Validation — Recall@1%FPR: {val_metrics['recall_at_1pct_fpr']:.4f} | "
                f"PR-AUC: {val_metrics['pr_auc']:.4f}")
    
    return model, val_metrics, val_probs


def check_mccv_val_alignment(best_mccv: Dict, val_metrics: Dict, model_name: str):
    """
    confirm the model mccv selected actually performs similaralyy on the validation set.
    a gap > 2% means the mccv folds were not representative.
    """
     
    gap = abs(best_mccv["mean_recall"] - val_metrics["recall_at_1pct_fpr"])
    if gap > MCCV_VAL_TOLERANCE:
        logger.warning(
            f"{model_name}: MCCV recall {best_mccv['mean_recall']:.4f} vs "
            f"val recall {val_metrics['recall_at_1pct_fpr']:.4f} - gap={gap:.4f}. "
            "Check for temporal drift or data leakage in splits"
        )
    else:
        logger.info(f"MCCV/val aligned - gap={gap:.4f} (within {MCCV_VAL_TOLERANCE})")


def check_calibration(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> float:
    """
    reliability diagram checks whether 'model says 80% fraud' means
    roughly 80% of those cases actually are fraud. Run on validation data
    (not test) so calibration stays out of the final evaluation.
    quantile binning ensures each bin has enough positive cases to be meaningful.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, "o-", label=model_name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraud rate")
    plt.title(f"Reliability Diagram — {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / f"{model_name}_calibration.png", dpi=150)
    plt.close()

    cal_error = float(np.mean(np.abs(prob_true - prob_pred)))
    logger.info(f"Calibration error: {cal_error:.4f} (< 0.05 good, > 0.10 concerning)")
    if cal_error > 0.10:
        logger.warning(
            "Calibration error > 0.10 — probability scores may be misleading. "
            "Consider Platt scaling before surfacing scores to analysts."
        )
    return cal_error


def calibration_model(model, X_cal, y_cal):
    """
    calibrate the model using Platt scaling.
    """
    raw_scores = model.predict_proba(X_cal)[:,1].reshape(-1,1)
    platt = LR()
    platt.fit(raw_scores, y_cal)
    logger.info("Platt scaling calibrated model")
    return platt


def calibrated_predict(model, platt, X):
    """
    predict probabilities using calibrated model.
    """
    raw_scores = model.predict_proba(X)[:,1].reshape(-1,1)
    return platt.predict_proba(raw_scores)[:,1]


def evaluate_fusion(y: np.ndarray, model_probs: np.ndarray, anomaly_scores: np.ndarray,
                    high: float, low: float, anomaly_thresh: float) -> Dict:
    """flag a transaction if model is very confident, or moderately confident + anomaly flag."""
    flagged = (
        (model_probs >= high) |((model_probs >= low) & (anomaly_scores >= anomaly_thresh))
    ).astype(int)

    tp = int(((flagged == 1) & (y == 1)).sum())
    fp = int(((flagged == 1) & (y == 0)).sum())
    tn = int(((flagged == 0) & (y == 0)).sum())
    fn = int(((flagged == 0) & (y == 1)).sum())

    pseudo_prob = np.where(model_probs >= high, 0.95,
                  np.where(anomaly_scores >= anomaly_thresh, 0.75, model_probs * 0.5))

    fpr, tpr, _ = roc_curve(y, pseudo_prob)
    prec, rec, _ = precision_recall_curve(y, pseudo_prob)

    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "fpr":fp / max(fp + tn, 1),
        "pr_auc":float(auc(rec, prec)),
        "roc_auc": float(roc_auc_score(y, pseudo_prob)),
        "recall_at_1pct_fpr": recall_at_target_fpr(fpr, tpr),
        "high_threshold": high,
        "low_threshold": low,
        "anomaly_threshold": anomaly_thresh,
    }


def run_fusion(y_cal, probs_cal, anomaly_cal, y_eval, probs_eval, anomaly_eval) -> Dict:
    """
    tune thresholds on the calibration slice (first half of val).
    measure final fusion performance on the eval slice (second half of val).
    this keeps threshold selection separate from performance reporting —
    the same way we keep training data separate from test data.
    """
    logger.info("Tuning fusion thresholds on calibration slice...")

    best_recall, best_config = 0.0, None

    for high in [0.4, 0.5, 0.6]:
        for low in [0.15, 0.2, 0.25]:
            if low >= high:
                continue
            for a_thresh in [0.6, 0.7, 0.8]:
                result = evaluate_fusion(y_cal, probs_cal, anomaly_cal, high, low, a_thresh)
                if result["fpr"] <= 0.02 and result["recall_at_1pct_fpr"] > best_recall:
                    best_recall = result["recall_at_1pct_fpr"]
                    best_config = result

    if best_config is None:
        best_config = evaluate_fusion(y_cal, probs_cal, anomaly_cal, 0.5, 0.2, 0.7)

    logger.info(
        f"Best thresholds (cal): high={best_config['high_threshold']}, "
        f"low={best_config['low_threshold']}, anomaly={best_config['anomaly_threshold']}"
    )

    # now measure on the eval slice the thresholds never saw
    fusion_eval = evaluate_fusion(
        y_eval, probs_eval, anomaly_eval,
        best_config["high_threshold"],
        best_config["low_threshold"],
        best_config["anomaly_threshold"],
    )
    logger.info(f"Fusion eval recall@1%FPR: {fusion_eval['recall_at_1pct_fpr']:.4f}")
    return fusion_eval

 
def explain_model(model,X_sample: np.ndarray,features: List[str],model_name: str) -> np.ndarray:
    """
    generate SHAP plots and return mean absolute shap values per feature.
    ysed downstream for SHAP-based feature selection.
    """
    logger.info(f"Creating SHAP plots for {model_name}...")
 
    try:
        act_model = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
        X_input = model.named_steps['scaler'].transform(X_sample) if hasattr(model, 'named_steps') else X_sample
 
        explainer = shap.TreeExplainer(act_model)
        shap_values = explainer.shap_values(X_input)
 
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
 
        # summary plot (dots)
        shap.summary_plot(shap_values, X_input, feature_names=features, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"{model_name}_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
 
        # bar plot (importance)
        shap.summary_plot(shap_values, X_input, feature_names=features, show=False, max_display=20, plot_type="bar")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"{model_name}_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
 
        logger.info(f"Saved SHAP plots to {SHAP_DIR}")
        return np.abs(shap_values).mean(axis=0)
 
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        return np.ones(len(features))
 
 
def shap_feature_selection(mean_shap: np.ndarray,features: List[str],threshold: float = 0.001) -> List[str]:
    """
    drop features whose mean absolute SHAP value is below threshold
    threshold=0.001 is conservative only removes truly zero-contribution features.
    """
    k = [f for f, s in zip(features, mean_shap) if s >= threshold]
    r =[f for f, s in zip(features, mean_shap) if s < threshold]
 
    if r:
        logger.info(f"SHAP selection removed {len(r)} near zero features: {r}")
    logger.info(f"SHAP selection kept {len(k)}/{len(features)} features")
 
    return k


def plot_thresholds(y_true, y_prob, model_name):
    """Show where to set decision threshold. Useful for business discussions."""
    
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Precision-Recall vs threshold
    ax1.plot(thresh, prec[:-1], label='Precision')
    ax1.plot(thresh, rec[:-1], label='Recall')
    ax1.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.set_title(f'{model_name}: Threshold vs precison.recall')
    ax1.grid(True, alpha=0.3)
    
  
    ax2.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_prob):.3f}')
    ax2.axvline(x=0.01, color='r', linestyle='--', alpha=0.5, label='1% FPR target')
    ax2.fill_between([0, FPR_THRESHOLD], 0, 1, alpha=0.1, color='green')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    ax2.set_title('ROC Curve')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = SHAP_DIR / f"{model_name}_thresholds.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def log_results(results: List[Dict], best_name: str, fusion: Dict, cal_error: float, model_path: Path):
    """Log everything to MLflow if available."""
    
    try:
        mlflow.set_tracking_uri("https://dagshub.com/virajdeshmukh080818/AML.mlflow")
        dagshub.init(repo_owner="virajdeshmukh080818", repo_name="AML", mlflow=True)
        mlflow.set_experiment("AML_Supervised_Exploration")
        
        # Log each model
        for r in results:
            with mlflow.start_run(run_name=r["name"]):
                mlflow.log_param("is_best", r["name"] == best_name)
                mccv = r["mccv"]
                mlflow.log_metric("conservative_score", mccv["mean_recall"] -2* mccv["std_recall"])
                mlflow.log_metric("mccv_mean_recall", mccv["mean_recall"])
                mlflow.log_metric("mccv_std_recall", mccv["std_recall"])
                
                if r.get("val_metrics"):
                    for k, v in r["val_metrics"].items():
                        mlflow.log_metric(f"val_{k}", v)
         
        # Log fusion
        with mlflow.start_run(run_name="fusion_and_final"):
            mlflow.log_param("base_model", best_name)
            mlflow.log_metric("calibration_error", cal_error)
            for k, v in fusion.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            
            # Attach SHAP plots
            if SHAP_DIR.exists():
                for plot in SHAP_DIR.glob("*.png"):
                    mlflow.log_artifact(str(plot))
                mlflow.log_artifact(str(model_path))
        
        logger.info("Logged to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


def print_results(results: List[Dict], fusion: Dict, best_name: str):
    """Print a nice summary table."""
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'Conservative':>14} {'Mean Recall':>12} {'Std':>10} {'Val PR-AUC':>12} {'Valid Folds':>11}")
    print("-"*80)
    
    # Sort by MCCV mean recall
    for r in sorted(results, key=lambda x: x["mccv"]["mean_recall"] -2 * x["mccv"]["std_recall"], reverse=True):
        
        mccv = r["mccv"]
        val = r.get("val_metrics") or {}
        score = mccv["mean_recall"] - 2 * mccv["std_recall"]
        tag = " <- BEST" if r["name"] == best_name else ""

        print(f"{r['name']:<20} {score:>14.4f} {mccv['mean_recall']:>12.4f} {mccv['std_recall']:>10.4f} {val.get('pr_auc', 0):>12.4f} {mccv['n_valid_iterations']:>11d}{tag}")
    print("-"*80)
    print(f"{'FUSION (eval)':<20} {fusion.get('recall_at_1pct_fpr', 0):>14.4f} {'N/A':>12} {'N/A':>10} {fusion.get('pr_auc', 0):>12.4f} {'N/A':>11}")
    print("="*80 + "\n")


def main():
    logger.info("Starting AML supervised training pipeline")
    setup()
    #load schema and features
    train_file = INPUT_DIR / "train_features.parquet"
    features = get_features(pl.scan_parquet(train_file).schema)

    X_train, y_train, train_timestamps = load_data('train', features, max_rows=TRAIN_SAMPLE_SIZE)
    X_val, y_val, _ = load_data("val", features, max_rows=VAL_SAMPLE_SIZE)

    #remove low var features
    variances = np.var(X_train, axis=0)
    var_mask = variances > 0.001
    removed = [f for f, keep in zip(features, var_mask) if not keep]
    if removed:
        logger.warning(f"Removed {len(removed)} low-var features: {removed}")

    X_train = X_train[:, var_mask]
    X_val = X_val[:, var_mask]
    features = [f for f, keep in zip(features, var_mask) if keep]
    logger.info(f"Kept {len(features)} features after variance filter")

    n_fraud = int(y_train.sum())
    imbalance_ratio = (len(y_train) - n_fraud) / max(n_fraud, 1)
    logger.info(f"Class imbalance: 1:{imbalance_ratio:.0f}")

    # split val into calibration slice (fusion tuning) and eval slice (reporting)
    # This prevents fusion threshold search from leaking into reported metrics.
    mid = len(X_val) // 2
    X_cal, y_cal = X_val[:mid], y_val[:mid]
    X_eval, y_eval = X_val[mid:], y_val[mid:]

    has_anomaly = 'anomaly_score' in features
    if has_anomaly:
        a_idx = features.index('anomaly_score')
        anomaly_cal = X_cal[:,a_idx].copy()
        anomaly_eval = X_eval[:, a_idx].copy()

    #mccv model selection
    logger.info('=' * 60)
    logger.info('MCCV Model Selection')
    logger.info('=' * 60)

    model_configs = get_model_configs(imbalance_ratio)
    all_results, mccv_results = [], {}

    for name, config in model_configs.items():
        try:
            logger.info(f"Evaluating {name}...")
            mccv = mccv_evaluate(config, X_train, y_train, train_timestamps)
            mccv_results[name] = mccv
            all_results.append({"name": name, "mccv": mccv, "val_metrics": None})
            score = mccv["mean_recall"] - 2 * mccv["std_recall"]
            logger.info(f"  {mccv['mean_recall']:.4f} ± {mccv['std_recall']:.4f} "
                        f"(conservative: {score:.4f})")
        except Exception as e:
            logger.error(f"  {name} failed during MCCV: {e}")

    if not mccv_results:
        logger.error("All models failed — cannot continue")
        sys.exit(1)

    best_name, best_mccv = select_best_model(mccv_results)
    best_config = model_configs[best_name]

    #finalm model training
    logger.info('=' * 60)
    logger.info('Final Training')
    logger.info("=" * 60)

    # tain on all of X_train; evaluate on X_eval (clean half of val)
    final_model, val_metrics, probs_eval = train_model(
        best_config, X_train, y_train, X_eval, y_eval
    )
    platt = calibration_model(final_model, X_cal, y_cal)
    probs_eval = calibrated_predict(final_model, platt, X_eval)
    check_mccv_val_alignment(best_mccv, val_metrics, best_name)

    # calibration on validation data — not test data — to keep test fully clean
    cal_error = check_calibration(y_eval, probs_eval, best_name)

    #optimal threshold
    optimal_threshold = find_optimal_threshold(y_eval, probs_eval)

    for r in all_results:
        if r["name"] == best_name:
            r["val_metrics"] = val_metrics

    #fusion   
    logger.info("=" * 60)
    logger.info("Fusion")
    logger.info("=" * 60)

    if has_anomaly:
        probs_cal = final_model.predict_proba(X_cal)[:, 1]
        fusion_results = run_fusion(
            y_cal, probs_cal, anomaly_cal,
            y_eval, probs_eval, anomaly_eval,
        )
    else:
        fusion_results = {"recall_at_1pct_fpr": 0.0, "pr_auc": 0.0}

    del X_train, y_train, X_val, y_val, X_cal, y_cal, X_eval, y_eval
    gc.collect()

    #test evaluation
    logger.info("=" * 60)
    logger.info("Test Evaluation")
    logger.info("=" * 60)

    X_test, y_test, _ = load_data("test", features, max_rows=TEST_SAMPLE_SIZE)
    test_probs = calibrated_predict(final_model, platt, X_test)
    test_metrics = calculate_metrics(y_test, test_probs)

    logger.info(f"Test recall@1%FPR: {test_metrics['recall_at_1pct_fpr']:.4f} | "
                f"PR-AUC: {test_metrics['pr_auc']:.4f}")

    # Explainability on a test sample
    mean_shap = explain_model(final_model, X_test[:1000], features, best_name)
    shap_kept = shap_feature_selection(mean_shap, features, threshold=0.001)
    if len(shap_kept) < len(features):
        logger.info(
            f"SHAP suggests {len(features)- len(shap_kept)} features can be dropped"
            f"in the next pipeline run. See logs above for features names"
        )
    del X_test
    gc.collect()


    #save
    model_path = MODEL_DIR / f"{best_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":final_model,
            "platt": platt,
            "features": features,
            "optimal_threshold": optimal_threshold,
            "shap_kept_features": shap_kept,
            "imbalance_ratio": imbalance_ratio,
            "mccv_results": best_mccv,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "calibration_error": cal_error,
            "fusion_config": fusion_results if has_anomaly else None,
        }, f)
    logger.info(f"Model saved: {model_path} ({os.path.getsize(model_path)/1024**2:.1f} MB)")


    plot_thresholds(y_test, test_probs, best_name)
    print_results(all_results, fusion_results, best_name)
    log_results(all_results, best_name, fusion_results, cal_error, model_path)

    logger.info("Pipeline Ciomplete")


if __name__ == "__main__":
    main()
