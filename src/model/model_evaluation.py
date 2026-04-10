import os
import gc
import logging
import yaml
import pickle
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import polars as pl
import mlflow
import dagshub
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    classification_report, confusion_matrix
)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

try:
    from src.logger import logging
except Exception:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


#params
def load_params(params_path: str='params.yaml') -> dict:
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info(f"parameters loaded from {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"params.yaml not found at {params_path}")
        raise


#load model and data
def load_model_artifact(model_dir: Path) -> dict:
    """
    load the model pickle saved by model_building stage
    return full artifact dict - model, platt, features, metrics, thrshold
    """
    model_files = list(model_dir.glob("*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model .pkl found in {model_dir}")

    model_path = model_files[0]
    logger.info(f"Loading model from {model_path}")

    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
    
    logger.info(f"Model loaded: {artifact['best_model_name']}")
    logger.info(f"Features: {len(artifact['features'])}")
    logger.info(f'Optimal Theshold: {artifact['optimal_threshold']:.4f}')

    return artifact, model_path

 
def load_test_data(
    test_path: Path,
    features: list,
    target_col: str,
    time_col: str,
    variance_mask: np.ndarray,
    max_rows: int,
    random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """load test features and apply same variance mask used during training."""
    cols_to_load = list(set(features + [target_col]))
    n_rows = pl.scan_parquet(test_path).select(pl.len()).collect().item()
 
    if max_rows and n_rows > max_rows:
        lazy  = pl.scan_parquet(test_path).select(cols_to_load)
        fraud_df = lazy.filter(pl.col(target_col) == 1).collect()
        n_legit= min(max_rows - len(fraud_df), n_rows - len(fraud_df))
        rng= np.random.RandomState(random_state)
        legit_idx = sorted(rng.choice(n_rows - len(fraud_df), n_legit, replace=False).tolist())
        legit_df = (
            lazy.filter(pl.col(target_col) == 0)
            .with_row_index("_idx")
            .filter(pl.col("_idx").is_in(legit_idx))
            .drop("_idx")
            .collect()
        )
        df = pl.concat([fraud_df, legit_df]).sample(fraction=1.0, shuffle=True, seed=random_state)
        del fraud_df, legit_df
        gc.collect()
    else:
        df = pl.scan_parquet(test_path).select(cols_to_load).collect()
 
    y = df[target_col].to_numpy().astype(np.int32)
    X = df.select(features).to_numpy().astype(np.float32)
    del df
    gc.collect()
 
    #impute nulls
    ZERO_IMPUTE = [
        "burst_score_1h", "burst_count_24h", "txn_in_hour",
        "toxic_corridor_count_28d", "toxic_corridor_volume_28d",
        "txn_count_28d", "txn_count_total",
        "total_amount_paid_28d", "total_amount_received_28d",
    ]
    for col_idx, col_name in enumerate(features):
        row_indices = np.where(np.isnan(X[:, col_idx]))[0]
        if len(row_indices) == 0:
            continue
        fill = 0.0 if col_name in ZERO_IMPUTE else float(np.nanmean(X[:, col_idx]))
        X[row_indices, col_idx] = fill
 
    # apply variance mask from training
    if variance_mask is not None and X.shape[1] > variance_mask.sum():
        X = X[:, variance_mask]
 
    logger.info(f"Test data: {len(y):,} rows | fraud={int(y.sum()):,} ({y.mean():.4%})")
    return X, y
 
 
# evaluation
def calibrated_predict(model, platt, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)[:, 1].reshape(-1, 1)
    return platt.predict_proba(raw)[:, 1]
 
 
def evaluate(y_true: np.ndarray,y_prob: np.ndarray,threshold: float,fpr_threshold: float) -> dict:
    """
    compute full evaluation suite for AML model.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
 
    # recall at target FPR
    recall_at_target = float(np.interp(fpr_threshold, fpr, tpr))
 
    # metrics at operational threshold
    y_pred = (y_prob >= threshold).astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp =int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
 
    precision_at_thresh = tp / max(tp + fp, 1)
    recall_at_thresh = tp / max(tp + fn, 1)
    fpr_at_thresh = fp / max(fp + tn, 1)
    f1_at_thresh= (2 * precision_at_thresh * recall_at_thresh) / max(precision_at_thresh + recall_at_thresh, 1e-9)
 
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(auc(recall, precision)),
        "recall_at_1pct_fpr": recall_at_target,
        "precision_at_threshold": precision_at_thresh,
        "recall_at_threshold": recall_at_thresh,
        "fpr_at_threshold": fpr_at_thresh,
        "f1_at_threshold": f1_at_thresh,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "threshold_used": threshold,
    }
 
 
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    reports_dir: Path
) -> None:
    """
    confusion matrix at operational threshold.
    more meaningful for compliance review than raw probability distributions.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
 
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"]
    )
    plt.title(f"{model_name} — Confusion Matrix (threshold={threshold:.3f})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(reports_dir / f"confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("Confusion matrix saved")
 
 
def plot_pr_curve(y_true: np.ndarray,y_prob: np.ndarray,model_name: str,reports_dir: Path) -> None:
    """
    Precision-Recall curve  the primary evaluation plot for imbalanced fraud data.
    ROC-AUC can be misleading at high class imbalance (1:104).
    PR-AUC is the honest metric.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    baseline = y_true.mean()
 
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.axhline(y=baseline, color="r", linestyle="--", label=f"Baseline (fraud rate = {baseline:.4%})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} — Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(reports_dir / f"pr_curve.png", dpi=150)
    plt.close()
    logger.info("PR curve saved")
 

# MLflow logging
def setup_mlflow(params: dict) -> None:
    """Configure MLflow tracking to DagsHub."""
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    dagshub_username = os.getenv("DAGSHUB_USERNAME")
 
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TOKEN not set in env")
    if not dagshub_username:
        raise EnvironmentError("DAGSHUB_USERNAME not set in env")
 
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
 
    repo_owner = params["dagshub"]["repo_owner"]
    repo_name = params["dagshub"]["repo_name"]
    tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
 
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
 
    logger.info(f"MLflow tracking URI: {tracking_uri}")
 
 
def log_to_mlflow(
    artifact: dict,
    test_metrics: dict,
    model_path: Path,
    reports_dir: Path,
    shap_dir: Path,
    params: dict
) -> str:
    """
    log everything to MLflow for experiment tracking.
    returns run_id for model_registry stage.
 
    what gets logged:
    - all test metrics (primary performance)
    - MCCV metrics (stability / model selection record)
    - calibration error (probability reliability)
    - optimal threshold (operational decision boundary)
    - model hyperparameters
    - model artifact (.pkl)
    - all plots (SHAP, calibration, pr curve, cm)
    """
    model_name = artifact["best_model_name"]
    mccv = artifact["mccv_results"]
 
    with mlflow.start_run(run_name=f"AML_{model_name}") as run:
 
        # test metrics
        mlflow.log_metrics({
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_recall_at_1pct_fpr": test_metrics["recall_at_1pct_fpr"],
            "test_precision_at_threshold": test_metrics["precision_at_threshold"],
            "test_recall_at_threshold": test_metrics["recall_at_threshold"],
            "test_f1_at_threshold":test_metrics["f1_at_threshold"],
            "test_fpr_at_threshold": test_metrics["fpr_at_threshold"],
        })
 
        # mccv metrics
        mlflow.log_metrics({
            "mccv_mean_recall": mccv["mean_recall"],
            "mccv_std_recall": mccv["std_recall"],
            "mccv_conservative": mccv["conservative_score"],
            "mccv_n_valid_folds": mccv["n_valid_folds"],
        })
 
        # model quality metrics
        mlflow.log_metrics({
            "calibration_error": artifact["calibration_error"],
            "optimal_threshold": artifact["optimal_threshold"],
            "mccv_val_gap":abs(mccv["mean_recall"] - artifact["val_metrics"]["recall_at_1pct_fpr"]),
            "n_features":len(artifact["features"]),
            "n_shap_kept_features": len(artifact.get("shap_kept_features", artifact["features"])),
        })
 
        # model params
        mlflow.log_params({
            "best_model": model_name,
            "imbalance_ratio": round(artifact["imbalance_ratio"], 2),
            "fpr_threshold": params["model"].get("fpr_threshold", 0.01),
            "mccv_iterations": params["model"].get("mccv_iterations", 5),
            "train_sample_size": params["model"].get("train_sample_size", 1_500_000),
        })
 
        # model artifact
        mlflow.log_artifact(str(model_path), artifact_path="model")
 
        # plots
        for plot_dir in [reports_dir, shap_dir]:
            if plot_dir.exists():
                for plot in plot_dir.glob("*.png"):
                    mlflow.log_artifact(str(plot), artifact_path="plots")
 
        run_id = run.info.run_id
        logger.info(f"MLflow run logged | run_id: {run_id}")
 
    return run_id
 

# main
def main():
    logger.info("=" * 60)
    logger.info("Stage 5: Model Evaluation")
    logger.info("=" * 60)
 
    try:
        params = load_params("params.yaml")
        model_cfg = params["model"]
        target_col= params["data_ingestion"].get("target_col", "Is Laundering")
        time_col= params["data_ingestion"].get("timestamp_col", "Timestamp")
        fpr_threshold = model_cfg.get("fpr_threshold", 0.01)
 
        model_dir= Path(params["storage"]["model_build_dir"])
        features_dir = Path(params["storage"]["anomaly_dir"])
        reports_dir= Path(params["storage"]["reports_dir"])
        shap_dir= model_dir / "shap"
        reports_dir.mkdir(parents=True, exist_ok=True)
 
        #load model 
        artifact, model_path = load_model_artifact(model_dir)
        model_name= artifact["best_model_name"]
        features= artifact["features"]
        variance_mask = artifact.get("variance_mask")
        threshold= artifact["optimal_threshold"]
 
        # load test data 
        test_path = features_dir / "test_features.parquet"
        X_test, y_test = load_test_data(
            test_path, features, target_col, time_col,
            variance_mask=variance_mask,
            max_rows=model_cfg.get("test_sample_size", 750_000),
            random_state=model_cfg.get("random_state", 42)
        )
 
        #evaluate 
        test_probs= calibrated_predict(artifact["model"], artifact["platt"], X_test)
        test_metrics = evaluate(y_test, test_probs, threshold, fpr_threshold)
 
        logger.info("=" * 60)
        logger.info("Test Evaluation Results")
        logger.info("=" * 60)
        logger.info(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"PR-AUC:{test_metrics['pr_auc']:.4f}")
        logger.info(f"Recall@1%FPR: {test_metrics['recall_at_1pct_fpr']:.4f}")
        logger.info(f"Precision@threshold: {test_metrics['precision_at_threshold']:.4f}")
        logger.info(f"Recall@threshold:{test_metrics['recall_at_threshold']:.4f}")
        logger.info(f"F1@threshold: {test_metrics['f1_at_threshold']:.4f}")
        logger.info(f"TP={test_metrics['tp']:,} | FP={test_metrics['fp']:,} | "
                    f"TN={test_metrics['tn']:,} | FN={test_metrics['fn']:,}")
 
        #plots 
        plot_confusion_matrix(y_test, test_probs, threshold, model_name, reports_dir)
        plot_pr_curve(y_test, test_probs, model_name, reports_dir)
 
        #save metrics.json for DVC
        import json
        metrics_path = reports_dir / "eval_metrics.json"
        
        from src.model.explainability import AMLExplainer
    
        explainer = AMLExplainer(model=artifact['model'], feature_names=features)
        explainer.get_global_importance(X_test[:1000], save_dir=str(shap_dir))

        del X_test
        gc.collect()

        with open(metrics_path, "w") as f:
            json.dump({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))}, f, indent=4)
        logger.info(f"Metrics saved -> {metrics_path}")
 
        #MLflow logging
        setup_mlflow(params)
        run_id = log_to_mlflow(artifact, test_metrics, model_path, reports_dir, shap_dir, params)
 
        #experiment_info.json for registry
        experiment_info = {
            "run_id": run_id,
            "model_name":model_name,
            "model_path": str(model_path),
            "test_roc_auc":test_metrics["roc_auc"],
            "test_pr_auc":test_metrics["pr_auc"],
            "test_recall_at_1pct_fpr": test_metrics["recall_at_1pct_fpr"],
            "optimal_threshold": threshold,
            "calibration_error": artifact["calibration_error"],
        }
        exp_info_path = reports_dir / "experiment_info.json"
        with open(exp_info_path, "w") as f:
            json.dump(experiment_info, f, indent=4)
        logger.info(f"Experiment info saved → {exp_info_path}")
 
        logger.info("=" * 60)
        logger.info("Model evaluation complete")
        logger.info(f"Run ID: {run_id}")
        logger.info("=" * 60)
 
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise
 
 
if __name__ == "__main__":
    main()
 