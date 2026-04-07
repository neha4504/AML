import boto3
import polars as pl
import logging
from io import BytesIO
from src.logger import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class s3_operations:
    def __init__(self, bucket_name:str, aws_access_key: str, aws_secret_key: str, region_name: str ="us-east-1"):
        """
        Initialize the s3_operations class with AWS credentials"""
        
        self.bucket_name = bucket_name
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region_name
            )
            logging.info(f"S3 connection initialize for bucket {bucket_name}")
        except Exception as e:
            logging.exception(f"Failed to initialize S3 connection: {e}")
            raise
    
    def fetch_file_from_s3(self, file_key: str) -> pl.DataFrame:
        """
        Fetches a CSV file from the s3 bucket and return it as a pandas Dataframe
        :param file_key: S3 file path (e.g. 'data/data.csv')
        :return Polars Dataframe
        """
        try:
            logging.info(f"Fetching file '{file_key}' from s3 bucket '{self.bucket_name}'...")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            buffer = BytesIO(obj['Body'].read())
            df = pl.read_csv(buffer, try_parse_dates=True, infer_schema_length=10000)
            logging.info(f"Successfully fetched '{file_key}' from S3 with {len(df)} records.")
            return df
        except Exception as e:
            logging.exception(f"Failed to fetch '{file_key}' from s3: {e}")
            raise

    def upload_file_to_s3(self, s3_key: str, file_path: str):
        """
        Uploads a file to the s3 bucket
        """
        try:
            logging.info(f"Uploading file '{file_path}' to s3 bucket '{self.bucket_name}'...")
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            logging.info(f"Uploade complete: s3://{self.bucket_name}/{s3_key}")
        except Exception as e:
            logging.exception(f"Failed to upload '{file_path}' to s3: {e}")
            raise