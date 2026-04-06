from pathlib import Path
from src.features.build_features import build_all_features

# paths
trans_path = Path('data/raw/HI-Medium_Trans.csv')
acc_path = Path('data/raw/HI-Medium_accounts.csv')
output_path = Path('aml_features')

# run feature engineering
build_all_features(trans_path, acc_path, output_path)