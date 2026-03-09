"""
Network Graph Features for AML Detection

Extract network-level patterns from transaction flows:
- Node centrality (degree, betweenness, closeness)
- Clustering coefficients
- PageRank scores
- Community patterns
"""

import polars as pl
import networkx as nx
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)



def compute_bank_centrality_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate bank-level centrality and join back to transactions.
    REFACTORED: Pre-aggregate before NetworkX to reduce memory 100x.
    """
    logger.info("Computing bank centrality features...")
    
    # PRE-AGGREGATE: Reduce 32M rows → ~10K bank pairs
    bank_edges = (
        df.group_by(['From Bank', 'To Bank']).agg([
            pl.count().alias('weight'),
            pl.col('Amount Paid').sum().alias('total_amount_paid')
        ]).collect(engine='streaming')
    )
    logger.info(f"  Aggregated to {len(bank_edges)} bank pairs")
      
    # Build network from aggregated edges (10K iterations vs 32M)
    bank_network = nx.DiGraph()
    for row in bank_edges.iter_rows(named=True):
        from_bank = row['From Bank']
        to_bank = row['To Bank']
        weight = row['weight']
        
        if bank_network.has_edge(from_bank, to_bank):
            bank_network[from_bank][to_bank]['weight'] += weight
        else:
            bank_network.add_edge(from_bank, to_bank, weight=weight)
    
    # Calculate centrality metrics (unchanged)
    try:
        pagerank = nx.pagerank(bank_network, weight='weight')
    except:
        pagerank = {node: 0.0 for node in bank_network.nodes()}
        logger.warning("PageRank calculation failed, using zeros")
    
    # Convert to Polars dataframes for efficient joining
    from_bank_features = pl.DataFrame({
        'From Bank': list(bank_network.nodes()),
        'from_bank_out_degree': [bank_network.out_degree(n) for n in bank_network.nodes()],
        'pagerank_from_bank': [pagerank.get(n, 0.0) for n in bank_network.nodes()],
    })

    to_bank_features = pl.DataFrame({
        'To Bank': list(bank_network.nodes()),
        'to_bank_in_degree': [bank_network.in_degree(n) for n in bank_network.nodes()],
        'pagerank_to_bank': [pagerank.get(n, 0.0) for n in bank_network.nodes()],
    })

    # Join back to original dataframe
    df = df.join(from_bank_features.lazy(), on='From Bank', how='left')
    df = df.join(to_bank_features.lazy(), on='To Bank', how='left')
        
    return df


def compute_account_network_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate account-level network features within rolling windows.
    
    Features:
    - account_counterparty_diversity_7d: Distinct 'To Bank' values in 7d window
    - account_counterparty_diversity_28d: Distinct 'To Bank' values in 28d window
    - account_bank_repeat_rate_7d: % of transactions to repeat banks in 7d window
    """
    logger.info("Computing account-level network features...")
    
    
    # Counterparty diversity: count distinct 'To Bank' per account

    df = df.with_columns([
        pl.col('To Bank')
        .eq(pl.col('To Bank').shift(1).over('Account_HASHED'))
        .cast(pl.Int8)
        .fill_null(0)
        .alias('is_same_bank')
    ])

    # rolling sum of repeats (integer window)
    # Bank repeat rate: proportion of repeat banks (using 200-row window ≈ 7d)
    df = df.with_columns([
        # 7dayproxy (120 rows)
        (pl.col('is_same_bank')
            .rolling_sum(window_size=120)
            .over('Account_HASHED') / 120.0)
            .shift(1)
            .fill_null(0)
            .cast(pl.Float32)
            .alias('bank_repeat_rate_7d'),
        
        (pl.col('is_same_bank')
            .rolling_sum(window_size=500)
            .over('Account_HASHED') /120.0)
            .shift(1)
            .fill_null(0)
            .cast(pl.Float32)
            .alias('bank_repeat_rate_28d'),
    ])
    
    # diversity proxy
    # if repeat rate is low (e.g 0.1), diversity is high 0.9
    df = df.with_columns([
        (1.0 - pl.col('bank_repeat_rate_7d')).alias('bank_diversity_proxy_7d'),
        (1.0 - pl.col('bank_repeat_rate_28d')).alias('bank_diversity_proxy_28d')
    ])
    return df.drop('is_same_bank')


def compute_corridor_risk_score(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Create corridor-level risk aggregation feature.
    
    Combines From Bank -> To Bank corridor information with transaction specifics.
    """
    logger.info("Computing corridor-level features...")
    
    df = df.with_columns([
        # create a corridor identifier
        (pl.col('From Bank').cast(pl.Utf8) + '_to_' + pl.col('To Bank').cast(pl.Utf8))
            .alias('corridor')
    ])
    
    # Compute corridor statistics in rolling windows (500-row window ≈ 28d)
    df = df.with_columns([
        pl.col('Amount Paid')
            .rolling_mean(window_size=500)
            .over('corridor')
            .shift(1)
            .fill_null(0)
            .alias('corridor_mean_amount_28d'),
        
        pl.col('Amount Paid')
            .rolling_std(window_size=500)
            .over('corridor')
            .shift(1)
            .fill_null(0)
            .alias('corridor_std_amount_28d'),
    ])
    
    return df


def add_network_features(df: pl.DataFrame) -> pl.DataFrame:
    """ 
    Add all network features
    """
    # defensive check: ensure the input is a Polars DataFrame or LazyFrame to
    # catch regressions where a different type (e.g., a tuple) is passed in.
    assert isinstance(df, pl.LazyFrame), (
        "add_network_features expects a Polars DataFrame or LazyFrame"
    )

    logger.info("  Computing bank centrality...")
    df = compute_bank_centrality_features(df)

    logger.info("  Computing account network....")
    df = compute_account_network_features(df)

    logger.info("  Computing corridor risk score...")
    df = compute_corridor_risk_score(df)

    # Return the enhanced DataFrame. Keep networks in local variables in case
    # they are needed later; if needed, consider returning them as well.
    return df