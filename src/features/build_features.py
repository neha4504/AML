"""
Integrated AML Feature Engineering Pipeline

This module orchestrates the complete feature engineering pipeline:
1. Base features (temporal, benford, lifecycle)
2. Advanced rolling features (burst, time-gaps, velocity)
3. Counterparty entropy and network metrics
4. Unsupervised anomaly detection (Isolation Forest)
5. Feature validation and output

Usage:
    from src.features.build_features import build_all_features
    
    build_all_features(
        transactions_path='data/raw/transactions.csv',
        accounts_path='data/raw/accounts.csv',
        output_dir='aml_features',
        compute_anomaly_scores=True
    )
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import polars as pl
import warnings
import gc
warnings.filterwarnings('ignore')

# Import feature modules
from src.features.experimental.base_features import add_base_features
from src.features.experimental.precompute_entity_stats import precompute_entity_stats

from src.features.experimental.rolling_features_v2 import compute_rolling_features
from src.features.experimental.ratio_features import compute_advanced_features
from src.features.experimental.derived_features import compute_derived_features
from src.features.experimental.advanced_rolling_features_v2 import (
    add_advanced_rolling_features
)
from src.features.experimental.counterparty_entropy_features_v2 import (
    add_counterparty_entropy_features
)

from src.utils.hashing import hash_pii_column
from src.features.experimental.network_features import add_network_features
from src.features.experimental.toxic_corridors import apply_toxic_corridor_features, derive_toxic_corridors


logger = logging.getLogger(__name__)


def optimize_dtypes(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns([
        pl.col('Amount Paid').cast(pl.Float32),
        pl.col('Amount Received').cast(pl.Float32)
    ])

import shutil
def process_spilts_in_batches(
    df: pl.LazyFrame,
    split_name: str,
    entity_stats_lazy: Optional[pl.LazyFrame],
    output_dir: Path, 
    batch_size: int = 10000,
    toxic_corridors=None
) -> Path:

    logger.info(f"BATCH PROCESSING: {split_name.upper()}")
    
    #create temp directory for split batches
    temp_dir = output_dir / f"temp_batches_{split_name}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    #get unique accounts
    unique_acc = df.select('Account_HASHED').unique().collect().to_series().to_list()

    n_acc = len(unique_acc)
    n_batches = (n_acc + batch_size -1)// batch_size

    logger.info(f"   Total accounts: {n_acc:,}")
    logger.info(f"   Batch size: {batch_size:,} accounts")
    logger.info(f"   Number of batches: {n_batches}")
    logger.info(f"   Estimated time: {n_batches * 15}-{n_batches * 25} minutes")

    batch_results = []

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_acc)
        batch_acc = unique_acc[start_idx:end_idx]

        logger.info(f"BATCH {batch_idx+1}/{n_batches} ({len(batch_acc):,} accounts)")

        #filter to batch accounts
        df_batch = df.filter(pl.col('Account_HASHED').is_in(batch_acc))

        # 0. Optimize dtypes
        df_batch = optimize_dtypes(df_batch)
        
        # 1. Sort for rolling features
        logger.info("   Step 1: Sorting(maintained through pipeline)...")
        df_batch = df_batch.sort(['Account_HASHED', 'Timestamp'])
        
        # 2. Base features
        logger.info("   Step 2: Base features...")
        df_batch = add_base_features(df_batch)

        # 2.1 Join precompute entity/accounts stats if available
        if entity_stats_lazy is not None:
            logger.info("   Step 2.5: Joining entity/accounts stats into transactions....")
            df_batch = df_batch.join(entity_stats_lazy, left_on='Account_HASHED', right_on='Account Number_HASHED', how='left')

            assert 'Account_HASHED' in df.columns or isinstance(df, pl.LazyFrame), "Expected Account_HASHED in transaction df"
            if entity_stats_lazy is not None:
                assert 'Account Number_HASHED' in entity_stats_lazy.columns, "entity_stats must contain Account Number_HASHED for join"

        # 3. Standard rolling features (from original pipeline)
        logger.info("   Step 3: Standard rolling features...")
        df_batch = compute_rolling_features(df_batch)
   
        # 4. Derived/ratio features
        logger.info("   Step 4: Ratio and Derived features...")
        df_batch = compute_advanced_features(df_batch)
        df_batch = compute_derived_features(df_batch)

        # 5. Advanced rolling features
        logger.info("   Step 5: Advanced rolling features...")
        df_batch = add_advanced_rolling_features(df_batch)
        
        # 6. Counterparty entropy
        logger.info("   Step 6: Counterparty entropy features...")
        df_batch = add_counterparty_entropy_features(df_batch)
        
        # 7. Network features
        logger.info("   Step 7: Network features...")
        df_batch = add_network_features(df_batch)
        
        # 8. Toxic corridors
        logger.info("   Step 8: Toxic corridor features...")
        df_batch = apply_toxic_corridor_features(df_batch, toxic_corridors=toxic_corridors)
        
        # Collect this batch
        logger.info(f"   Collecting batch {batch_idx+1} (streaming)...")
        df_batch_collected = df_batch.collect(engine='streaming')
        
        logger.info(f"   Batch {batch_idx+1} collected: {len(df_batch_collected):,} rows, {len(df_batch_collected.columns)} columns")
        
        batch_file = temp_dir / f"batch_{batch_idx:04d}.parquet"
        df_batch_collected.write_parquet(batch_file)
        logger.info(f"   Batch {batch_idx+1} saved to disk: {len(df_batch_collected):,} rows")
        #batch_results.append(df_batch_collected)

        del df_batch, df_batch_collected
        gc.collect()

    logger.info(f"CONCATENATING {n_batches} BATCHES")
    
    #final output path for split
    final_op_path = output_dir / f"{split_name}_features.parquet"

    pl.scan_parquet(temp_dir / "*.parquet").sink_parquet(final_op_path)
    logger.info(f"  Final {split_name} saved directly to: {final_op_path}")

    shutil.rmtree(temp_dir)
    # del batch_results
    # gc.collect()
    return final_op_path


def build_training_features(
    train_df: pl.LazyFrame,
    val_df: pl.LazyFrame,
    test_df: pl.LazyFrame,
    accounts: Optional[pl.DataFrame] = None,
    output_dir: Path = Path('./aml_features'),
    toxic_corridors=None
) -> Tuple[Path, Path, Path]:

    """
    Build all features for training, validation, and test sets.
    Processes splits sequentially to manage memory.
    
    Args:
        train_df, val_df, test_df: Lazy DataFrames by split
        accounts: Accounts reference data
    
    Returns:
        Tuple of (train_features, val_features, test_features) as eager DataFrames
    """
    logger.info("="*70)
    logger.info("BUILDING AML FEATURES")
    logger.info("="*70)

    entity_stats_lazy = None
    if accounts is not None:
        logger.info("   Precomputing entity-level stats from accounts...")
        entity_stats = precompute_entity_stats(accounts)

        if 'Account HASHED' not in entity_stats.columns and 'Account Number' in entity_stats.columns:
            logger.info("   Hashing Account Number in entity_stats to produce Account_HASHED...")
            entity_stats = hash_pii_column(entity_stats.lazy(), 'Account Number').collect()

        entity_stats_lazy = entity_stats.lazy()
        del entity_stats

        import gc
        gc.collect()

    # Process each split
    splits = [
        ('train', train_df),
        ('val', val_df),
        ('test', test_df)
    ]
    
    processed_splits = {}
    
    for split_name, df in splits:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {split_name.upper()} split")
        logger.info(f"{'='*70}")
        
        output_path = process_spilts_in_batches(
            df=df, split_name=split_name,
            entity_stats_lazy=entity_stats_lazy,
            output_dir=output_dir,
            batch_size=15000,
            toxic_corridors=toxic_corridors
        )
        processed_splits[split_name] = output_path
        logger.info(f"\n{split_name.upper()} split complete")
        gc.collect()

    
    train_features = processed_splits['train']
    val_features = processed_splits['val']
    test_features = processed_splits['test']
    
    # Add anomaly scores (fitted on train, applied to all) - DISABLED FOR NOW
    # logger.info("\n" + "="*70)
    # logger.info("Adding Unsupervised Anomaly Scores (Isolation Forest)")
    # logger.info("="*70)
    
    # train_features, val_features, test_features = add_isolation_forest_scores(
    #     train_features,
    #     val_features,
    #     test_features,
    #     contamination=0.10
    # )
    
    return train_features, val_features, test_features


def validate_features(file_path: Path) -> Dict:
    """
    Validate feature engineering output.
    
    Checks:
    - No NaNs in critical features
    - Feature distribution reasonableness
    - Class balance
    - Feature diversity
    """
    logger.info(f"Validating feature quality for {file_path.name}...")
    
    #load as lazyframe
    df_lazy = pl.scan_parquet(file_path)
    cols = df_lazy.columns

    validation_report = {
        'num_rows': df_lazy.select(pl.len()).collect().item(),
        'num_features': len(cols),
    }
    
    # Check critical features for missing values
    critical_features = [
        col for col in cols
        if 'rolling' in col or 'burst' in col or 'entropy' in col or 'anomaly' in col
    ]
    
    if critical_features:
        exprs = [pl.col(col).is_null().sum().alias(col) for col in critical_features[:5]]
        missing_counts = df_lazy.select(exprs).collect().row(0)

        for col, missing in zip(critical_features[:5], missing_counts):
            if missing > 0:
                logger.warning(f"   {col}: {missing} missing values")
    
    # Basic statistics
    logger.info(f"\nFeature Statistics:")
    logger.info(f"  Total rows: {validation_report['num_rows']}")
    logger.info(f"  Total features: {validation_report['num_features']}")
    
    # Class balance
    target_col = 'Is Laundering'
    if target_col:
        class_counts = df_lazy.group_by(target_col).len().collect()
        logger.info(f"\n Class Distribution: ")
        for row in class_counts.iter_rows(named=True):
            logger.info(f"   Class {row[target_col]}: {row['len']:,}samples")
    
    return validation_report


def create_temporal_splits(
    trans: pl.LazyFrame,
) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Splits transactions chronologically using only dense days (1-16).

    Why days 1-16 only:
        The IBM HI-Medium dataset has a hard cliff after day 16.
        Days 17-28 contain ~11,000 rows total vs ~2M+ per day in
        the dense window. Using sparse days as test produces a 60%
        fraud rate and 882 rows — statistically meaningless for
        any evaluation metric.

    Why sparse tail goes into train:
        Those ~848 fraud cases in days 17-28 are real signal.
        They should contribute to model learning, not contaminate
        evaluation. Folding them into train preserves the signal
        without distorting metrics.

    Split:
        Train  : days 1-11  + sparse tail (days 17-28)
        Val    : days 12-13
        Test   : days 14-16
    """
    dense_train = trans.filter(
        pl.col("Timestamp").dt.day() <= 11
    )

    val_df = trans.filter(
        (pl.col("Timestamp").dt.day() >= 12) &
        (pl.col("Timestamp").dt.day() <= 13)
    )

    test_df = trans.filter(
        (pl.col("Timestamp").dt.day() >= 14) &
        (pl.col("Timestamp").dt.day() <= 16)
    )

    # sparse tail folded into train — too few rows for evaluation
    # but valid as additional training signal
    sparse_tail = trans.filter(
        pl.col("Timestamp").dt.day() > 16
    )
    train_df = pl.concat([dense_train, sparse_tail])

    # log what each split actually contains so you can verify
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        stats = df.select([
            pl.len().alias("rows"),
            pl.col("Is Laundering").sum().alias("fraud"),
            pl.col("Timestamp").dt.day().min().alias("day_start"),
            pl.col("Timestamp").dt.day().max().alias("day_end"),
        ]).collect()
        logger.info(
            f"{name}: {stats['rows'].item():,} rows | "
            f"fraud={int(stats['fraud'].item()):,} | "
            f"days {stats['day_start'].item()}- {stats['day_end'].item()}"
        )

    return train_df, val_df, test_df
    

def build_all_features(
    transactions_path: Path,
    accounts_path: Path,
    output_dir: Path = Path('./aml_features'),
    compute_anomaly_scores: bool = False,
    sample_fraction: Optional[float] = None
) -> Tuple[Path, Path, Path]:
    """
    Complete feature engineering pipeline.
    
    Args:
        transactions_path: CSV file with transactions
        accounts_path: CSV file with accounts
        output_dir: Directory for output
        compute_anomaly_scores: Whether to compute Isolation Forest scores
        sample_fraction: Optional - use fraction of data for testing
    
    Returns:
        Tuple of (train_features_path, val_features_path, test_features_path)
    """
    logger.info("="*70)
    logger.info("AML FEATURE ENGINEERING PIPELINE (COMPLETE)")
    logger.info("="*70)
    
    # convert transactions data
    trans_parquet_path = transactions_path.with_suffix('.parquet')

    if trans_parquet_path.exists():
        logger.info(f"   Found cached Parquet: {trans_parquet_path}")    
    else:
        logger.info(f"   Converting {transactions_path.name}...")
    
        pl.scan_csv(
            transactions_path, 
            try_parse_dates=True,
            dtypes={
                'From Bank': pl.Utf8,
                'To Bank': pl.Utf8,
                'Amount Paid': pl.Float32,
                'Amount Received': pl.Float32,   }
        ).sink_parquet(trans_parquet_path, compression='snappy')

    # covert accounts data
    acc_parquet_path = accounts_path.with_suffix('.parquet')

    if acc_parquet_path.exists():
        logger.info(f"   Found cached Parquet: {acc_parquet_path}")
    else:
        logger.info(f"   Converting {accounts_path.name}...")
    
        df_acc = pl.read_csv(accounts_path)
        df_acc.write_parquet(acc_parquet_path)

        del df_acc

    logger.info("   Loading from Parquet....")
    # Load data from parquet files
    logger.info(f"\nLoading transactions from {trans_parquet_path}")
    trans = pl.scan_parquet(trans_parquet_path)

    logger.info(f"Loading accounts from {acc_parquet_path}")
    accounts = pl.read_parquet(acc_parquet_path)
    
    if sample_fraction:
        logger.info(f"Sampling {sample_fraction*100}% of transactions...")
        trans = trans.collect().sample(fraction=sample_fraction, seed=42).lazy()
    
    # Hash PII
    logger.info("Hashing PII columns...")
    trans = hash_pii_column(trans, 'Account')
    trans = trans.with_columns(pl.col('Account_HASHED').cast(pl.Utf8))
    
    logger.info("Creating temporal splits...")
    train_df, val_df, test_df = create_temporal_splits(trans)
    del trans
    import gc
    gc.collect()

    logger.info("Deriving toxic corridors from training data...")
    toxic_corridors = derive_toxic_corridors(
    train_df=train_df,   # training data only  no leakage
    threshold=0.02
)

    # Build features
    train_path, val_path, test_path = build_training_features(
        train_df, val_df, test_df, accounts, output_dir, toxic_corridors=toxic_corridors
    )
    
    # Validate
    validate_features(train_path)
   
    logger.info("\n" + "="*70)
    logger.info(" FEATURE ENGINEERING COMPLETE")
    logger.info("="*70)
   
    return train_path, val_path, test_path

if __name__ == '__main__':
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Default paths (adjust as needed)
    trans_path = Path('data/raw/HI-Medium_Trans.csv')
    acc_path = Path('data/raw/HI-Medium_accounts.csv')
    output_path = Path('aml_features')
    
    if not trans_path.exists() or not acc_path.exists():
        logger.error(f"Data files not found at {trans_path} or {acc_path}")
        sys.exit(1)
    
    build_all_features(trans_path, acc_path, output_path)
