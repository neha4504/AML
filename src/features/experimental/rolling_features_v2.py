import polars as pl
import logging
logger = logging.getLogger(__name__)

def compute_rolling_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    rolling feature computation for AML datasets.
    """
    
    if not isinstance(df, pl.LazyFrame):
        raise TypeError("compute_rolling_features requires LazyFrame input.")

    logger.info("Computing rolling features...")

    return df.with_columns([
        #  Batch 1: Counts 
        
        # 1h equivalent (Cumulative count - immediate activity)
        pl.col('Timestamp').count().over('Account_HASHED').cast(pl.UInt32).alias('txn_count_total'),
        
        # 24h equivalent (Last 24 transactions)
        # Using integer window 24 aligns with the memory-efficient strategy.
        pl.lit(1, dtype=pl.UInt32)
            .rolling_sum(window_size=24)
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0)
            .alias('txn_count_24h'),
            
        # 7d equivalent (Last 168 transactions)  
        pl.lit(1, dtype=pl.UInt32)
            .rolling_sum(window_size=168)
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0)
            .alias('txn_count_7d'),
        
        # 28d equivalent (Last 500 transactions)
        pl.lit(1, dtype=pl.UInt32)
            .rolling_sum(window_size=500)
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0)
            .alias('txn_count_28d'),

        #  Batch 2: Volume (Window 500) 
        
        pl.col('Amount Paid')
            .rolling_sum(window_size=500)
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0.0)
            .alias('total_amount_paid_28d'),

        pl.col('Amount Received')
            .rolling_sum(window_size=500)
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0.0)
            .alias('total_amount_received_28d'),

        #  Batch 3: Statistics (Window 500) 
        
        pl.col('Amount Paid')
            .rolling_mean(window_size=500)
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0.0)
            .alias('mean_amount_paid_28d'),

        pl.col('Amount Paid')
            .rolling_std(window_size=500)
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0.0)
            .alias('std_amount_paid_28d'),

        pl.col('Amount Paid')
            .rolling_quantile(window_size=500, quantile=0.5)
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0.0)
            .alias('median_amount_paid_28d'),

        pl.col('Amount Paid')
            .rolling_max(window_size=500)
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0.0)
            .alias('max_amount_paid_28d'),


        # Batch 4: Ratio features...
        # current amount vs rolling baseline (structuring singnal)
        # $9500 transaction is normal alone
        # $9500 when yor 28d average is #200 is very suspicious
        (pl.col('Amount Paid') / (pl.col('Amount Paid')
        .rolling_mean(window_size=500)
        .over('Account_HASHED')
        .shift(1)
        .fill_null(1.0) + 0.000001))
        .cast(pl.Float32)
        .alias('amount_vs_baseline_ratio')
    ])