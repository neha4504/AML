"""
Base Foundational and Time Features
Benford's Law Features
Detects:
Fake numbers: Identifies if transaction amounts look "made up" by humans rather than occurring naturally.
Hiding under the limit: Spots people trying to avoid "red flags" by keeping amounts just under the $10,000 reporting threshold.
Copy-paste patterns: Finds unusual repetition of the same specific dollar amounts.

Temporal Timing (Sin/Cos) Features
Detects:
Strange timing: Flags activity happening at weird hours, like a retail business suddenly moving money at 3:00 AM.
Robot-like behavior: Spots transfers that happen on a perfect, repetitive schedule (often a sign of automated laundering).
Weekend surges: Detects sudden spikes in activity during times when a specific customer is usually quiet.

Lifecycle Features
Detects:
"Burner" accounts: New accounts that immediately start moving massive amounts of money.
Sudden wake-ups: Old, empty accounts that suddenly "come to life" to move suspicious funds.
Behavior changes: Flags when a customer who usually spends $500 a month suddenly starts moving $50,000.
"""

import polars as pl
import numpy as np
import logging

logger = logging.getLogger(__name__)

def add_base_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add foundational temporal, Benford, and account lifecycle features.
    Optimized for AML risk detection on LazyFrames.
    """
    logger.info("Adding base temporal features....")

    # 1. Cyclical Time Encoding
    # Captures periodic nature of time (e.g., 23:00 is close to 00:00)
    cyclical_features = [
        (2 * np.pi * pl.col('Timestamp').dt.hour() / 24).sin().cast(pl.Float32).alias('hour_sin'),
        (2 * np.pi * pl.col('Timestamp').dt.hour() / 24).cos().cast(pl.Float32).alias('hour_cos'),
        
        (2 * np.pi * pl.col('Timestamp').dt.weekday() / 7).sin().cast(pl.Float32).alias('day_of_week_sin'),
        (2 * np.pi * pl.col('Timestamp').dt.weekday() / 7).cos().cast(pl.Float32).alias('day_of_week_cos'),
        
    ]

    df = df.with_columns(cyclical_features)

    # 2. Benford's Law & Rounding Features
    logger.info("Adding Benford's Law features...")
    
    benford_features = [
        # Extract first non-zero digit using Regex
        # We take absolute value first so negative amounts are handled correctly
        pl.col('Amount Paid')
        .abs()
        .cast(pl.Utf8)
        .str.extract(r"[1-9]")  # Finds the first character between 1 and 9
        .cast(pl.Int32, strict=False)
        .alias('first_digit'),

        # Round number flags (AML red flag: structuring)
        (pl.col('Amount Paid') % 100 == 0).cast(pl.Int8).alias('is_round_100'),
        (pl.col('Amount Paid') % 1000 == 0).cast(pl.Int8).alias('is_round_1000')
    ]

    df = df.with_columns(benford_features)

    # 3. Account Lifecycle Features
    df = df.sort('Account_HASHED', 'Timestamp')

    first_txn = pl.col('Timestamp').min().over('Account_HASHED')

    lifecycle_features = [
        # Anchor
        first_txn.alias('account_first_txn'),

        # Days since account's first transaction (Tenure)
        (pl.col('Timestamp') - first_txn).dt.total_days().cast(pl.Float32).alias('account_tenure_days'),

        # Transaction Sequence (Ordinal Rank)
        pl.col('Timestamp')
        .rank(method='ordinal')
        .over('Account_HASHED')
        .alias('txn_rank_in_account_history'),

        # Days since previous transaction (Velocity)
        pl.col('Timestamp')
        .diff()
        .over('Account_HASHED')
        .dt.total_days()
        .fill_null(0)
        .cast(pl.Float32) # First transaction gets 0
        .alias('days_since_last_txn'),

        # Flags for account maturity
        ((pl.col('Timestamp') - first_txn) >= pl.duration(days=7))
        .cast(pl.Int8)
        .alias('has_7d_history'),

        ((pl.col('Timestamp') - first_txn) >= pl.duration(days=28))
        .cast(pl.Int8)
        .alias('has_28d_history')
    ]

    df = df.with_columns(lifecycle_features)

    return df