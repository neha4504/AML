"""
Counterparty entropy and network analysis features - Polars-compatible version.

Detects:
- Money mule behavior (high inflow, low outflow)
- Network concentration (few counterparties)
- Pass-through laundering (balanced flow)
- Hub-and-spoke networks (central node)

All using Polars-compatible syntax (no time-based rolling).
"""

import polars as pl
from typing import Tuple
import logging
import gc

logger = logging.getLogger(__name__)

def compute_counterparty_entropy(df: pl.LazyFrame, chunk_size: int = 10000) -> pl.LazyFrame:
    """
    shannon entropy-like diversity metrics for counterparties.
    """
    logger = logging.getLogger(__name__)
    
    # Single-pass aggregation (NO chunking, NO filtering)
    entropy_features = (
        df.lazy()
        .group_by(['Account_HASHED', 'Account_duplicated_0'])
        .agg(pl.count().cast(pl.UInt32).alias('txn_count'))
        .with_columns([
            pl.col('txn_count').sum().over('Account_HASHED').alias('total_txns'),
            (pl.col('txn_count') / pl.col('txn_count').sum().over('Account_HASHED'))
                .cast(pl.Float32)
                .alias('probability')
        ])
        .with_columns([
            (pl.col('probability') * pl.col('probability').log())
                .fill_null(0.0)
                .sum()
                .over('Account_HASHED')
                .cast(pl.Float32)
                .neg()
                .alias('entropy_value')
        ])
        .select(['Account_HASHED', 'entropy_value'])
        .unique()

    )
    
    # Join back (now both are eager DataFrames)
    df = df.join(entropy_features, on='Account_HASHED', how='left')
    
    df = df.with_columns([
        pl.col('entropy_value').fill_null(0.0).cast(pl.Float32).alias('counterparty_entropy_28d')
    ])
    
    return df.drop('entropy_value')


def compute_counterparty_switching_metrics(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    detect round-robin money laundering (switching between counterparties).
    if A->X->A->Y->A pattern, that's suspicious.
    """

    # 1. Counterparty switches
    df = df.with_columns([
        (pl.col('Account_duplicated_0').ne(
            pl.col('Account_duplicated_0').shift(1).over('Account_HASHED')
        ))
        .fill_null(False)
        .cast(pl.Int8)
        .alias('is_counterparty_switch')
    ])
    
    # 2. Switching rate (per 50 txns and 200 txns)
    df = df.with_columns([
        (pl.col('is_counterparty_switch') 
            .rolling_mean(window_size=50)
            .over('Account_HASHED')
            .cast(pl.Float32)
            .alias('counterparty_switch_rate_50txns')),
        
        (pl.col('is_counterparty_switch')
            .rolling_sum(window_size=500)
            .over('Account_HASHED')
            .cast(pl.UInt32)
            .alias('total_counterparty_switches_28d')),
    ])
    
    # 3. Unique receiver diversity
    df = df.with_columns([
        pl.col('Account_duplicated_0')
        .n_unique()
        .over('Account_HASHED')
        .cast(pl.UInt32)
        .alias('num_unique_receivers_28d'),
    ])
    
    # 4. Recycling ratio (do we reuse the same N counterparties?)
    df = df.with_columns([
        (pl.col('num_unique_receivers_28d').cast(pl.Float32) / 
         (pl.col('total_counterparty_switches_28d').cast(pl.Float32) + 1.0))
        .cast(pl.Float32)
        .alias('counterparty_recycling_ratio_28d')

    ])
    
    return df


def compute_network_balance_ratios(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    inflow vs outflow balance (money mule and pass-through detection).
    
    - Mule: High inflow, low outflow
    - Pass-through: inflow ≈ outflow
    - Collector: Many inflows, few outflows
    """
   
    required = ['total_amount_received_28d', 'total_amount_paid_28d', 'txn_count_28d']
    schema = df.collect_schema()
    missing = [c for c in required if c not in schema]
    if missing:
        raise ValueError(
            f"compute_network_balance_ratios requires rolling features first. "
            f"Missing: {missing}"
        )
    # 1. Balance metrics
    df = df.with_columns([
        (pl.col('total_amount_received_28d') / 
        (pl.col('total_amount_paid_28d') + 0.000001))
        .cast(pl.Float32)
        .alias('inflow_outflow_balance_28d'),
        
        (pl.col('total_amount_received_28d') - pl.col('total_amount_paid_28d'))
        .cast(pl.Float32)
        .alias('net_flow_28d'),
        
        ((pl.col('total_amount_received_28d') + pl.col('total_amount_paid_28d')) / 2.0)
        .cast(pl.Float32)
        .alias('average_flow_magnitude_28d'),
    ])
    
    # 2. Transaction count balance
    df = df.with_columns([
        (pl.col('txn_count_28d') / 
         (pl.col('txn_count_28d') + 0.000001))
        .cast(pl.Float32)
        .alias('inflow_outflow_txn_ratio_28d')
    ])
    
    # 3. Pass-through detection (ratio close to 1)
    df = df.with_columns([
        (1.0 - 
         ((pl.col('inflow_outflow_balance_28d') - 1.0).abs() /
          (pl.col('inflow_outflow_balance_28d') + 1.0)))
        .cast(pl.Float32)
        .alias('passthrough_likelihood_28d'),
        
        (pl.col('inflow_outflow_balance_28d') -
         pl.col('inflow_outflow_balance_28d').shift(1).over('Account_HASHED'))
        .fill_null(0.0)
        .abs()
        .cast(pl.Float32)
        .alias('balance_volatility_daily'),
    ])
    
    return df


def compute_temporal_counterparty_patterns(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    time-of-day patterns in counterparty selection.
    automated systems have consistent patterns (suspicious).
    """
  
    
    # 1. Hour of day patterns
    df = df.with_columns([
        pl.col('Timestamp').dt.hour().cast(pl.Int8).alias('hour_of_day'),
        (pl.col('Timestamp').dt.hour() >= 17).cast(pl.Int8).alias('is_end_of_day_txn'),
    ])
    
    # 2. End-of-day clearing pattern (smurfing)
    df = df.with_columns([
        pl.col('is_end_of_day_txn')
            .rolling_sum(window_size=200)
            .over('Account_HASHED')
            .cast(pl.UInt32)
            .alias('end_of_day_txn_count_7d')
    ])
    
    return df


def compute_relationship_asymmetry(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    detect one-way relationships (A→B but B doesn't send to A).
    typical of money mules and launderers.
    """
   
    
    # 1. Create directed pairs
    df = df.with_columns([
        pl.concat_str([
            pl.col('Account_HASHED'),
            pl.col('Account_duplicated_0')
        ], separator="|").alias('account_pair_directed'),
        
        # Create undirected pair (for reverse lookup)
        pl.concat_str([
            pl.when(pl.col('Account_HASHED') < pl.col('Account_duplicated_0'))
                .then(pl.col('Account_HASHED'))
                .otherwise(pl.col('Account_duplicated_0')),
            pl.when(pl.col('Account_HASHED') < pl.col('Account_duplicated_0'))
                .then(pl.col('Account_duplicated_0'))
                .otherwise(pl.col('Account_HASHED')),
        ], separator="|").alias('account_pair_undirected')
    ])
    
    # 2. Count transactions in each direction
    df = df.with_columns([
        pl.col('Timestamp')
        .count()
        .over('account_pair_directed')
        .alias('txns_in_directed_pair')
    ])
    
    # 3. Detect one-way relationships
    df = df.with_columns([
        (pl.col('txns_in_directed_pair') > 5).cast(pl.Int8)
            .rolling_sum(window_size=500)
            .over('account_pair_directed')
            .cast(pl.UInt32)
            .alias('asymmetric_pair_evidence_28d')
    ])
    
    # 4. Volume asymmetry
    df = df.with_columns([
        (pl.col('Amount Paid') / 
         (pl.col('Amount Paid').shift(1).over('account_pair_directed') + 0.000001))
        .fill_null(1.0)
        .cast(pl.Float32)
        .alias('amount_asymmetry_with_counterparty')
    ])
    
    return df


def compute_network_centrality_proxy(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    proxy for hub-and-spoke network detection.
    hubs (mules) have many connections as in/out nodes.
    """
    
    
    # 1. In-degree and out-degree approximations
    df = df.with_columns([
        pl.col('Account_duplicated_0').n_unique()
        .over('Account_HASHED')
        .cast(pl.UInt32)
        .alias('out_degree_approximation'),
        
        pl.col('Account_HASHED').n_unique()
        .over('Account_duplicated_0')
        .cast(pl.UInt32)
        .alias('in_degree_approximation'),
    ])
    
    # 2. Hub score (high both in and out)
    df = df.with_columns([
        (pl.col('out_degree_approximation').cast(pl.Float64) * 
         pl.col('in_degree_approximation').cast(pl.Float64)).sqrt()
        .cast(pl.Float32)
        .alias('betweenness_centrality_proxy')
    ])
    
    return df


def add_counterparty_entropy_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add all counterparty-based network features.
    
    Applies transformations in order and returns enhanced DataFrame.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not isinstance(df, pl.LazyFrame):
        raise TypeError("add_counterparty_entropy_features requires LazyFrame input.")
    
    logger.info("  Computing counterparty entropy...")
    df = compute_counterparty_entropy(df)
    
    logger.info("  Computing counterparty switching metrics...")
    df = compute_counterparty_switching_metrics(df)
    
    logger.info("  Computing network balance ratios...")
    df = compute_network_balance_ratios(df)
    
    logger.info("  Computing temporal counterparty patterns...")
    df = compute_temporal_counterparty_patterns(df)
    
    logger.info("  Computing relationship asymmetry...")
    df = compute_relationship_asymmetry(df)
    
    logger.info("  Computing network centrality proxy...")
    df = compute_network_centrality_proxy(df)
    
    return df
