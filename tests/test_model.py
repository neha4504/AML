import unittest
import mlflow
import os
import polars as pl
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


class TestAMLModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN variable is not set in .env")

        os.environ["MLFLOW_TRACKING_USERNAME"] = "virajdeshmukh080818"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        repo_owner = "virajdeshmukh080818"
        repo_name = "AML"
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

        cls.model_name = "AML_Laundering_Detector"
        cls.new_model_version = cls.get_latest_model_version(cls.model_name)

        if not cls.new_model_version:
            raise ValueError("No model found in mlflow registry to test")

        cls.new_model_uri = f"models:/{cls.model_name}/{cls.new_model_version}"
        cls.new_model = mlflow.sklearn.load_model(cls.new_model_uri)

        cls.feature_names = [
            'Amount Received', 'Amount Paid', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'is_round_100', 
            'account_tenure_days', 'txn_rank_in_account_history', 'days_since_last_txn', 'has_7d_history', 'entity_account_count', 
            'entity_bank_count', 'txn_count_total', 'total_amount_paid_28d', 'total_amount_received_28d', 'mean_amount_paid_28d', 
            'std_amount_paid_28d', 'median_amount_paid_28d', 'max_amount_paid_28d', 'amount_vs_baseline_ratio', 
            'inflow_outflow_ratio_28d', 'is_counterparty_switch', 
            'counterparty_diversity_7d', 'counterparty_diversity_28d', 'prev_amount_paid', 'is_rush_txn', 'prev_was_sender', 
            'amount_deviation_from_avg', 'amount_vs_max_ratio', 'avg_txn_size_28d', 'amount_cv_28d', 'txn_in_hour', 
            'burst_score_1h', 'burst_count_24h', 'minutes_since_last_txn', 'avg_timegap_minutes_28d', 'min_timegap_minutes_28d', 
            'max_timegap_minutes_28d', 'std_timegap_minutes_28d', 'timegap_consistency_28d', 'timegap_acceleration', 
            'amount_paid_last_100', 'txn_velocity_delta', 'amount_velocity_pct_change', 
            'amount_acceleration_2nd_order', 'concentration_ratio_7d', 'amount_concentration_cv_28d', 
            'is_multiple_100', 'flag_high_burst', 'flag_large_gap', 'flag_extreme_consistency', 'flag_high_concentration', 
            'flag_heavy_structuring', 'anomaly_cascade_score', 'cascade_frequency_28d', 'counterparty_entropy_28d', 
            'counterparty_switch_rate_50txns', 'total_counterparty_switches_28d', 'num_unique_receivers_28d', 
            'counterparty_recycling_ratio_28d', 'inflow_outflow_balance_28d', 'average_flow_magnitude_28d', 'passthrough_likelihood_28d', 
            'hour_of_day', 'is_end_of_day_txn', 'end_of_day_txn_count_7d', 'txns_in_directed_pair', 
            'amount_asymmetry_with_counterparty', 'out_degree_approximation', 'in_degree_approximation', 
            'betweenness_centrality_proxy', 'from_bank_out_degree', 'to_bank_in_degree', 'bank_repeat_rate_7d', 'bank_repeat_rate_28d', 
            'bank_diversity_proxy_7d', 'bank_diversity_proxy_28d', 'corridor_mean_amount_28d', 'corridor_std_amount_28d', 
            'is_toxic_corridor', 'toxic_corridor_count_28d', 'toxic_corridor_volume_28d', 'pct_volume_via_toxic_corridors', 
            'Payment Format_target_enc', 'Receiving Currency_target_enc', 'Payment Currency_target_enc', 
            'From Bank_freq_enc', 'To Bank_freq_enc', 'corridor_freq_enc', 'anomaly_score'
        ]
        
        cls.holdout_data = pl.read_csv('data/test/test_data.csv')

        try:
            client = mlflow.MlflowClient()
            prod_version = client.get_model_version_by_alias(cls.model_name, "production")
            cls.prod_model = mlflow.sklearn.load_model(
                f"models:/{cls.model_name}/{prod_version.version}"
            )
        except Exception:
            cls.prod_model = None


    @staticmethod
    def get_latest_model_version(model_name):
        client = mlflow.MlflowClient()
        try:
            latest_version = client.get_model_version_by_alias(model_name, "staging")
            return latest_version.version if latest_version else None
        except Exception:
            return None

    def test_model_load(self):
        self.assertIsNotNone(self.new_model)
    
    def test_model_signature(self):
        dummy_input = pl.DataFrame({f: [0.0] for f in self.feature_names})

        #predictusing new model
        pred = self.new_model.predict(dummy_input.to_numpy())
        self.assertEqual(len(pred), 1)
        self.assertIn(int(pred[0]), [0,1])

    def test_model_performance(self):
        x_holdout = self.holdout_data[self.feature_names].to_numpy()
        y_holdout = self.holdout_data['Is Laundering'].to_numpy()

        y_probs_new = self.new_model.predict_proba(x_holdout)[:, 1]

        optimal_threshold = 0.1621619
        y_pred_new_cust = (y_probs_new >= optimal_threshold).astype(int)
        
        recall_new = recall_score(y_holdout, y_pred_new_cust, zero_division=0)
        f1_new = f1_score(y_holdout, y_pred_new_cust, zero_division=0)

        print(f"\n --NEW mdel metrics (At threshold {optimal_threshold:.4f})")
        print(f"Recall: {recall_new:.4f}")
        print(f"F1 score: {f1_new:.4f}")

        if self.prod_model is not None:
            y_probs_prod = self.prod_model.predict_proba(x_holdout)[:, 1]
            y_pred_prod_cust = (y_probs_prod >= optimal_threshold).astype(int)

            recall_prod = recall_score(y_holdout, y_pred_prod_cust, zero_division=0)
            f1_prod = f1_score(y_holdout, y_pred_prod_cust, zero_division=0)

            print(f"Production recall: {recall_prod:.4f} | New recall: {recall_new:.4f}")

            self.assertGreaterEqual(f1_new, f1_prod, f"Rejected: F1 at threshold dropped from {f1_prod:.4f} tp {f1_new:.4f}")
            self.assertGreaterEqual(recall_new, recall_prod, f"Rejected: Recall at threshold dropped from {recall_prod:.4f} to {recall_new:.4f}")
        
        else:
            self.assertGreaterEqual(recall_new, 0.70, 'First model recall is critically low')
            self.assertGreaterEqual(f1_new, 0.55, 'First model f1 is critically low')



if __name__ == "__main__":
    unittest.main()