import polars as pl
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def derive_toxic_corridors(
    train_df: pl.LazyFrame,
    threshold: float = 0.02
) -> pl.DataFrame:
    """
    Compute fraud rate per From Bank → To Bank corridor
    from the raw training LazyFrame.
    Only needs three columns — From Bank, To Bank, Is Laundering.
    Must use training data only to avoid leakage into val/test.
    """
    corridors = (
        train_df
        .group_by(['From Bank', 'To Bank'])
        .agg([
            pl.len().alias('total_txns'),
            pl.col('Is Laundering').sum().alias('fraud_txns'),
        ])
        .with_columns([
            (pl.col('fraud_txns') / pl.col('total_txns'))
            .cast(pl.Float32)
            .alias('fraud_rate')
        ])
        .filter(pl.col('total_txns') >= 100)
        .with_columns([
            (pl.col('fraud_rate') >= threshold)
            .cast(pl.Int8)
            .alias('is_toxic_corridor')
        ])
        .collect()
    )

    n_toxic = corridors.filter(pl.col('is_toxic_corridor') == 1).height
    n_total = corridors.height
    logger.info(
        f"Derived {n_toxic} toxic corridors out of {n_total} "
        f"(threshold={threshold:.1%}, min_txns=100)"
    )
    return corridors
    

def apply_toxic_corridor_features(df: pl.LazyFrame, toxic_corridors: pl.DataFrame | None=None) -> pl.LazyFrame:
  """
  Add features measuring exposure to toxic corridors.
  """
  #join with toxic corridor 
  
  if not isinstance(df, pl.LazyFrame):
        raise TypeError("apply_toxic_corridor_features requires LazyFrame input.")
  
  if toxic_corridors is None:
        logger.info("No toxic corridors provided. creating features")
    
        df = df.with_columns([
            pl.lit(0).cast(pl.Int8).alias('is_toxic_corridor'),
            pl.lit(0.0).cast(pl.Float32).alias('corridor_risk_score'),
            pl.lit(0).cast(pl.UInt32).alias('toxic_corridor_count_28d'),
            pl.lit(0.0).cast(pl.Float32).alias('toxic_corridor_volume_28d'),
            pl.lit(0.0).cast(pl.Float32).alias('pct_volume_via_toxic_corridors'),
        ])

        return df
    
  logger.info(f"Joining with {len(toxic_corridors)} toxic corridors...")

  required_cols = {'From Bank', 'To Bank'}
  if not required_cols.issubset(set(toxic_corridors.columns)):
    logger.warning("  Toxic corridors must contain 'From Bank' and 'To Bank' columns.")
    return apply_toxic_corridor_features(df, None)

  if 'is_toxic_corridor' not in toxic_corridors.columns:
    toxic_corridors = toxic_corridors.with_columns([
        pl.lit(0.5).cast(pl.Float32).alias('fraud_rate')
    ])


  df = df.join(
    toxic_corridors.lazy().select(['From Bank', 'To Bank', 'is_toxic_corridor', 'fraud_rate']),
    on=['From Bank', 'To Bank'],
    how='left'
  )  

  df = df.with_columns([
    pl.col('is_toxic_corridor').fill_null(0).cast(pl.Int8),
    pl.col('fraud_rate').fill_null(0.0).cast(pl.Float32).alias('corridor_risk_score'),
    ])

  # Use 500-row window (≈ 28 days based on typical transaction frequency)
  df = df.with_columns([
      pl.col('is_toxic_corridor')
          .cast(pl.Int64)
          .rolling_sum(window_size=500)
          .over('Account_HASHED')
          .shift(1)
          .fill_null(0)
          .cast(pl.UInt32)
          .alias('toxic_corridor_count_28d'),

      (pl.col('Amount Paid') * pl.col('is_toxic_corridor'))
          .rolling_sum(window_size=500)
          .over('Account_HASHED')
          .shift(1)
          .fill_null(0)
          .cast(pl.Float32)
          .alias('toxic_corridor_volume_28d'),
  ])

  df = df.with_columns([
      (pl.col('toxic_corridor_volume_28d') /
        pl.when(pl.col('total_amount_paid_28d') > 0.0)
          .then(pl.col('total_amount_paid_28d'))
          .otherwise(1.0))
      .cast(pl.Float32)
      .alias('pct_volume_via_toxic_corridors'),
  ])

  return df
