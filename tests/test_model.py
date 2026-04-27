import unittest
import mlflow
import os
import tempfile
import pickle
import logging
import polars as pl
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import recall_score, f1_score

from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAMLModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = "virajdeshmukh080818"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(
            "https://dagshub.com/virajdeshmukh080818/AML.mlflow"
        )

        cls.model_name = "AML_Laundering_Detector"

        client = mlflow.MlflowClient()
        version_obj = client.get_model_version_by_alias(cls.model_name, "staging")
        run_id = version_obj.run_id

        # download artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="model/model.pkl",
                dst_path=tmpdir
            )
            with open(path, "rb") as f:
                artifacts = pickle.load(f)

        cls.model = artifacts["model"]
        cls.platt = artifacts["platt"]
        cls.feature_names = artifacts["features"]
        cls.threshold = artifacts['optimal_threshold']

        logger.info(f"Model loaded: {type(cls.model)}")

        #load holdout safely
        df = pl.read_csv(
            "data/test/test_data_sample.csv",
            schema_overrides={'Account_HASHED': pl.Utf8}
        )

        #ensure feature alignment
        available_features = [c for c in cls.feature_names if c in df.columns]
        df = df.select(available_features + ["Is Laundering"])

        cls.X = df.select(available_features).to_pandas()          
        cls.y = df["Is Laundering"].to_numpy()
        cls.available_features = available_features

        # load production model
        try:
            prod_version = client.get_model_version_by_alias(cls.model_name, "production")
            cls.prod_model = mlflow.sklearn.load_model(
                f"models:/{cls.model_name}/{prod_version.version}"
            )
        except Exception:
            cls.prod_model = None

    def test_model_load(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
      
        dummy = pd.DataFrame(
            np.zeros((1, len(self.feature_names)), dtype=np.float32),
            columns=self.feature_names
        )

        raw_prob = self.model.predict_proba(dummy)[0][1]
        final_prob = self.platt.predict_proba([[raw_prob]])[0][1]

        self.assertTrue(0 <= final_prob <= 1)

    def test_model_performance(self):
        raw_probs = self.model.predict_proba(self.X)[:, 1]
        final_probs = self.platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

        y_pred = (final_probs >= self.threshold).astype(int)

        recall_new = recall_score(self.y, y_pred, zero_division=0)
        f1_new = f1_score(self.y, y_pred, zero_division=0)

        logger.info(f"\n--- NEW model metrics (threshold={self.threshold:.4f}) ---")
        logger.info(f"Recall : {recall_new:.4f}")
        logger.info(f"F1  : {f1_new:.4f}")

        if self.prod_model is not None:
            
            prod_X = self.X if isinstance(self.X, pd.DataFrame) else pd.DataFrame(
                self.X, columns=self.available_features
            )
            prod_probs = self.prod_model.predict_proba(prod_X)[:, 1]
            prod_pred = (prod_probs >= self.threshold).astype(int)

            recall_prod = recall_score(self.y, prod_pred, zero_division=0)
            f1_prod = f1_score(self.y, prod_pred, zero_division=0)

            logger.info(f"Production Recall : {recall_prod:.4f} | New Recall : {recall_new:.4f}")
            logger.info(f"Production F1  : {f1_prod:.4f} | New F1 : {f1_new:.4f}")

            self.assertGreaterEqual(
                f1_new, f1_prod,
                f"F1 dropped: {f1_prod:.4f} → {f1_new:.4f}"
            )
            self.assertGreaterEqual(
                recall_new, recall_prod,
                f"Recall dropped: {recall_prod:.4f} → {recall_new:.4f}"
            )

        else:
            self.assertGreaterEqual(recall_new, 0.70)
            self.assertGreaterEqual(f1_new, 0.55)


if __name__ == "__main__":
    unittest.main(verbosity=2)