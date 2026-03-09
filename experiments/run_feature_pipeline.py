
"""Lightweight, standalone feature-engineering pipeline.

This script orchestrates feature generation only by calling
`src.features.build_features.build_all_features`.

- Loads raw transactions and accounts
- Builds features (base, rolling, derived, counterparty, network, etc.)
- Saves train/val/test feature Parquet files to the specified output directory
- Does NOT perform model training or anomaly detection

Usage:
    python experiments/run_feature_pipeline.py
"""

from pathlib import Path
import argparse
import logging
import sys

# Ensure repo package imports work when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import build_all_features


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main(
    transactions_path: Path,
    accounts_path: Path,
    output_dir: Path,
    sample_fraction: float | None = None,
):
    logger.info("Starting feature-only pipeline")

    transactions_path = Path(transactions_path)
    accounts_path = Path(accounts_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Basic validation
    if not transactions_path.exists():
        logger.error(f"Transactions file not found: {transactions_path}")
        raise SystemExit(1)
    if not accounts_path.exists():
        logger.error(f"Accounts file not found: {accounts_path}")
        raise SystemExit(1)

    logger.info(f"Transactions: {transactions_path}")
    logger.info(f"Accounts: {accounts_path}")
    logger.info(f"Output dir: {output_dir}")
    if sample_fraction:
        logger.info(f"Sampling fraction: {sample_fraction}")

    # Build features
    train_path, val_path, test_path = build_all_features(
        transactions_path=transactions_path,
        accounts_path=accounts_path,
        output_dir=output_dir,
        compute_anomaly_scores=False,
        sample_fraction=sample_fraction
    )

    logger.info("Feature engineering completed")
    logger.info(f"Train features written to: {train_path}")
    logger.info(f"Val features written to: {val_path}")
    logger.info(f"Test features written to: {test_path}")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature-only pipeline')
    parser.add_argument('--trans-path', type=Path, default=Path('data/raw/HI-Medium_Trans.csv'), help='Path to transactions CSV (default: data/raw/HI-Medium_Trans.csv)')
    parser.add_argument('--accounts-path', type=Path, default=Path('data/raw/HI-Medium_accounts.csv'), help='Path to accounts CSV (default: data/raw/HI-Medium_accounts.csv)')
    parser.add_argument('--output-dir', type=Path, default=Path('aml_features'), help='Output directory for features (default: aml_features)')
    parser.add_argument('--sample', type=float, default=None, help='Optional sampling fraction')

    args = parser.parse_args()
    raise SystemExit(main(args.trans_path, args.accounts_path, args.output_dir, args.sample))