import yaml
import os
import gdown
import logging
import polars as pl
from src.logger import logging
from src.connections.s3_connection import s3_operations
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

#load params
def load_params(params_path: str) -> dict:
    """
    Loads parameters from params.yaml file
    """
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logging.info(f"Loaded parameters from {params_path}")
        return params
    except Exception as e:
        logging.exception(f"Failed to load parameters from {params_path}: {e}")
        raise


#load data from local
def load_data_local(params: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    
    cfg = params['data_ingestion']
    trans_path = Path(cfg['trans_local_path'])
    acc_path =Path(cfg['acc_local_path'])

    logger.info(f"Loading data from local...")
    try:
        if trans_path.suffix == '.parquet':
            transactions = pl.read_parquet(trans_path)
        else:
            transactions = pl.read_csv(trans_path)
        
        if acc_path.suffix == '.parquet':
            accounts = pl.read_parquet(acc_path)
        else:
            accounts = pl.read_csv(acc_path)
        
        logger.info(f"Transactions loaded: {len(transactions):,} rows | {len(transactions.columns)} cols")
        logger.info(f"Accounts loaded: {len(accounts):,} rows | {len(accounts.columns)} cols")

        return transactions, accounts
    
    except FileNotFoundError as e:
        logger.error(f"Local file not found: {e}")
        raise
    

#load data from s3
def load_data_from_s3(params: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    pull transactions and accounts from s3
    """
    cfg = params['data_ingestion']
    s3 = s3_operations(
        bucket_name=cfg['s3_bucket'],
        aws_access_key=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=cfg.get['region_name', 'us-east-1']
    )
    logger.info(f"Connecting to S3 bucket: {cfg['s3_bucket']}")

    #transactions
    trans_key = cfg['transactions_s3_key']
    logger.info(f"Fetching transactions: {trans_key}")
    transactions = s3.fetch_file_from_s3(trans_key)
    logger.info(f"Transactions: {len(transactions):,} rows | {len(transactions.columns)} columns")

    #acccounts
    acc_key = cfg['accounts_s3_key']
    logger.info(f"Fetching accounts: {acc_key}")
    accounts = s3.fetch_file_from_s3(acc_key)
    logger.info(f'Accounts: {len(accounts):,} rows | {len(accounts)} columns')

    return transactions, accounts

def load_data_from_gdrive(params: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """ download data from google drive"""

    cfg = params['data_ingestion']
    raw_dir = Path(params['storage']['raw_data_dir'])
    raw_dir.mkdir(parents=True, exist_ok=True)

    trans_file_id = cfg['gdrive_trans_id']
    acc_file_id = cfg['gdrive_acc_id']

    trans_path = raw_dir / "transactions_data"
    acc_path = raw_dir / "accounts_data"

    logger.info("Downloading data from google drive...")
    gdown.download(f"https://drive.google.com/uc?id={trans_file_id}", str(trans_path), quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={acc_file_id}", str(acc_path), quiet=False)

    #load using polars
    if trans_path.suffix == 'parquet':
        transactions = pl.read_parquet(trans_path)
    else:
        transactions = pl.read_csv(trans_path)

    if acc_path.suffix == 'parquet':
        accounts = pl.read_parquet(acc_path)
    else:
        accounts = pl.read_csv(acc_path)

    logger.info(f"Transactions loaded: {len(transactions):,}")
    logger.info(f"Accounts loaded: {len(accounts):,}")

    return transactions, accounts


def save_raw_data(transactions: pl.DataFrame, accounts: pl.DataFrame, params: dict) -> tuple[Path, Path]:
    """
    save raw data as parquet for downstream DVC stages
    returns (transactions_path, accounts_path)
    """
    raw_dir = Path(params['storage']['raw_data_dir'])
    raw_dir.mkdir(parents=True, exist_ok=True)

    trans_path = raw_dir / 'transactions_raw.parquet'
    acc_path = raw_dir / 'accounts_raw.parquet'

    transactions.write_parquet(trans_path, compression='snappy')
    accounts.write_parquet(acc_path, compression='snappy')

    logger.info(f"Transactions saved -> {trans_path}")
    logger.info(f"Accounts saved -> {acc_path}")
    return trans_path, acc_path


def main():
    logger.info('=' * 60)
    logger.info('AML Pipeline - Stage 1: Data Ingestion')
    logger.info('='*60)

    try:
        params = load_params('params.yaml')
        #local data loading
        #transactions, accounts = load_data_local(params)
        #s3 data loading
        #transactions, accounts = load_data_from_s3(params)
        transactions, accounts = load_data_from_gdrive(params)
        trans_path, acc_path = save_raw_data(transactions, accounts, params)

        logger.info('=' * 60)
        logger.info('Data Ingestion Complete')
        logger.info('='* 60)

    except Exception as e:
        logger.error(f"data ingestion failed: {e}")
        raise


if __name__ == '__main__':
    main()