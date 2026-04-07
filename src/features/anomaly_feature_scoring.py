import gc
import os
import pickle
import yaml
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest

try:
    from src.logger import logging
except Exception:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

#config
RANDOM_STATE = 42
CONTAMINATION = "auto"
CHUNK_SIZE = 100_000
TRAIN_SAMPLE_SIZE = 2_500_000
EXCLUDE_COLS = [
    "Is Laundering", "is_laundering",
    "Account_HASHED", "Account", "account_id",
    "transaction_id", "tx_id",
    "Timestamp", "timestamp", "Date", "date",
]


def load_params(params_path: str = "params.yaml") -> dict:
    try:
        with open(params_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"params.yaml not found at {params_path}")
        raise


def get_feature_columns(features_dir: Path, split: str = "train") -> List[str]:
    """extract numeric features from schema without loading data."""
    path = features_dir / f"{split}_features.parquet"
    schema = pl.scan_parquet(path).collect_schema()
    
    numeric_types = (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64, pl.Boolean,
    )
    
    features = []
    for col, dtype in schema.items():
        if any(excl.lower() in col.lower() for excl in EXCLUDE_COLS):
            continue
        if isinstance(dtype, numeric_types):
            features.append(col)
    
    logger.info(f"Found {len(features)} usable features for anomaly detection")
    return features


def compute_train_medians(features_dir: Path, feature_cols: List[str], target_col: str) -> np.ndarray:
    """compute medians from training data (legit only) for imputation."""
    path = features_dir / "train_features.parquet"
    logger.info("Computing feature medians from training data")
    
    medians_df = (
        pl.scan_parquet(path)
        .filter(pl.col(target_col) == 0)  # Legit only
        .select([pl.col(c).median() for c in feature_cols])
        .collect()
    )
    
    meds = medians_df.to_numpy().flatten().astype(np.float32)
    meds = np.nan_to_num(meds, nan=0.0)
    
    zero_cols = int((meds == 0).sum())
    logger.info(f"Medians computed — {zero_cols} columns have zero median")
    return meds


def clean_chunk(chunk_df: pl.DataFrame, feature_cols: List[str], medians: np.ndarray) -> np.ndarray:
    """convert chunk to numpy array with imputation."""
    X = chunk_df.select(feature_cols).to_numpy().astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    
    #column wise imputation using pre computed medians
    for col_idx in range(X.shape[1]):
        nan_mask = np.isnan(X[:, col_idx])
        if nan_mask.any():
            X[nan_mask, col_idx] = medians[col_idx]
    
    return X


def train_anomaly_model(features_dir: Path, feature_cols: List[str], 
                       medians: np.ndarray, target_col: str) -> IsolationForest:
    """train isolation forest on sampled legit transactions only."""
    path = features_dir / "train_features.parquet"
    
    # Load legit rows only
    legit_df = (
        pl.scan_parquet(path)
        .select(feature_cols + [target_col])
        .filter(pl.col(target_col) == 0)
        .select(feature_cols)
        .collect()
    )
    
    n_legit = len(legit_df)
    sample_size = min(TRAIN_SAMPLE_SIZE, n_legit)
    
    #random sample for training
    rng = np.random.default_rng(RANDOM_STATE)
    row_indices = sorted(rng.choice(n_legit, size=sample_size, replace=False).tolist())
    train_sample = legit_df[row_indices]
    
    logger.info(f"Training sample: {sample_size:,} rows x {len(feature_cols)} features")
    
    X_train = clean_chunk(train_sample, feature_cols, medians)
    
    del legit_df, train_sample
    gc.collect()
    
    model = IsolationForest(
        n_estimators=200,
        max_samples=10_050,
        contamination=CONTAMINATION,
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    
    model.fit(X_train)
    logger.info("Isolation Forest training complete")
    
    del X_train
    gc.collect()
    return model


def score_and_save_split(model: IsolationForest, features_dir: Path, output_dir: Path,
                        split: str, feature_cols: List[str], medians: np.ndarray):
    """score one split in chunks and write augmented parquet."""
    input_path = features_dir / f"{split}_features.parquet"
    output_path = output_dir / f"{split}_features.parquet"
    
    n_rows = pl.scan_parquet(input_path).select(pl.len()).collect().item()
    n_chunks = (n_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    logger.info(f"Scoring {split}: {n_rows:,} rows in {n_chunks} chunks")
    
    lazy_reader = pl.scan_parquet(input_path)
    parquet_writer = None
    all_scores = []
    
    for chunk_idx, row_start in enumerate(range(0, n_rows, CHUNK_SIZE)):
        chunk_df = lazy_reader.slice(row_start, CHUNK_SIZE).collect()
        
        X = clean_chunk(chunk_df, feature_cols, medians)
        #flip sign higher score = more anomalous
        scores = -model.score_samples(X)
        
    
        chunk_df = chunk_df.with_columns(pl.Series("anomaly_score", scores))
        arrow_chunk = chunk_df.to_arrow()
        
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
                str(output_path), arrow_chunk.schema, compression="snappy"
            )
        parquet_writer.write_table(arrow_chunk)
        
        all_scores.append(scores)
        
        del chunk_df, arrow_chunk, X, scores
        gc.collect()
        
        if (chunk_idx + 1) % 10 == 0 or (chunk_idx + 1) == n_chunks:
            logger.info(f"  {split}: {chunk_idx + 1}/{n_chunks} chunks complete")
    
    if parquet_writer:
        parquet_writer.close()
    
    scores_arr = np.concatenate(all_scores)
    logger.info(
        f"Anomaly score distribution [{split}] — "
        f"mean: {scores_arr.mean():.4f}, std: {scores_arr.std():.4f}, "
        f"max: {scores_arr.max():.4f}, p99: {np.percentile(scores_arr, 99):.4f}"
    )
    
    del all_scores, scores_arr
    gc.collect()
    
    file_mb = os.path.getsize(output_path) / 1024**2
    logger.info(f"Written: {output_path} ({file_mb:.1f} MB)")


def main():
    logger.info("=" * 60)
    logger.info("AML Pipeline — Stage 3.5: Anomaly Scoring")
    logger.info("=" * 60)
    
    try:
        params = load_params("params.yaml")
        target_col = params["data_ingestion"].get("target_col", "Is Laundering")
        
        features_dir = Path(params["storage"]["features_dir"])
        output_dir = Path(params["storage"].get("anomaly_features_dir", "data/processed_with_anomaly"))
        model_dir = Path(params["storage"].get("model_dir", "models"))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        #verify inputs exist
        for split in ["train", "val", "test"]:
            path = features_dir / f"{split}_features.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}. Run feature_engineering first.")
        
        #get features
        feature_cols = get_feature_columns(features_dir, "train")
        if not feature_cols:
            raise ValueError("No valid feature columns found")
        
        #vompute medians from train (prevents leakage)
        medians = compute_train_medians(features_dir, feature_cols, target_col)
        
        #yrain model
        model = train_anomaly_model(features_dir, feature_cols, medians, target_col)
        
        #save model artifact
        model_path = model_dir / "anomaly_detector.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": model,
                "feature_cols": feature_cols,
                "medians": medians,
                "contamination": CONTAMINATION,
                "random_state": RANDOM_STATE,
            }, f)
        logger.info(f"Anomaly model saved: {model_path}")
        
        #score all splits
        for split in ["train", "val", "test"]:
            logger.info("-" * 60)
            score_and_save_split(model, features_dir, output_dir, split, feature_cols, medians)
        
    
        logger.info("=" * 60)
        logger.info("Anomaly scoring complete")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Anomaly scoring failed: {e}")
        raise


if __name__ == "__main__":
    main()