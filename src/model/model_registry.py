import os
import json
import pickle
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from mlflow import MlflowClient

load_dotenv()

try:
    from src.logger import logging
except Exception:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


#params
def load_params(params_path: str='params.yaml') -> dict:
    try:
        with open(params_path, 'r') as x:
            params = yaml.safe_load(x)
        logger.info(f"Parameters loaded from {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"params.yaml not found at {params_path}")
        raise


#mlflow setup
def setup_mlflow(params: dict) -> MlflowClient:
    """configure mlflow and retur tracking client"""
    
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    dagshub_username = os.getenv("DAGSHUB_USERNAME")

    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TOKEN not set in .env")
    if not dagshub_username:
        raise EnvironmentError("DAGSHUB_USERNAME not set in env")

    os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

    repo_owner = params['dagshub']['repo_owner']
    repo_name = params['dagshub']['repo_name']
    tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    logger.info(f"Mlflow client connected: {tracking_uri}")
    return client


#load experiment info
def load_experiment_info(reports_dir: Path) ->dict:
    exp_info_path = reports_dir / 'experiment_info.json'
    if not exp_info_path.exists():
        raise FileNotFoundError(f"experiment_info.json not found at {exp_info_path}")

    with open(exp_info_path, 'r') as x:
        info = json.load(x)
    logger.info(f"Experiment info loaded | run_id: {info['run_id']}")
    logger.info(f"New model: {info['model_name']}")
    logger.info(f" PR-AUC: {info['test_pr_auc']:.4f}")
    logger.info(f" Recall@1%FPR: {info['test_recall_at_1pct_fpr']:.4f}")
    logger.info(f" Calibration error: {info['calibration_error']:.4f}")

    return info


def get_production_metrics(client: MlflowClient, registered_model_name: str) ->dict | None:
    """
    fetch metrics of the current production model from mLflow registry.
    returns None if no production model exists yet.
    """
    try:
        versions = client.get_latest_versions(registered_model_name, stages=["Production"])
        if not versions:
            logger.info("No production model found — new model will be first staging candidate")
            return None
 
        prod_version = versions[0]
        run_data= client.get_run(prod_version.run_id)
        metrics= run_data.data.metrics
 
        logger.info(f"Current production model | version: {prod_version.version}")
        logger.info(f"  PR-AUC:{metrics.get('test_pr_auc', 0):.4f}")
        logger.info(f"  Recall@1%FPR: {metrics.get('test_recall_at_1pct_fpr', 0):.4f}")
 
        return {
            "version":prod_version.version,
            "pr_auc": metrics.get("test_pr_auc", 0),
            "recall_at_1pct_fpr": metrics.get("test_recall_at_1pct_fpr", 0),
            "roc_auc":metrics.get("test_roc_auc", 0),
            "calibration_error": metrics.get("calibration_error", 1.0),
        }
 
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            logger.info("Model not yet registered — will register as new")
            return None
        raise
 
 
def should_promote(new_info: dict, prod_metrics: dict | None, params: dict) -> bool:
    """
    determine if new model should be promoted to staging.
 
    promotion criteria (both must be met):
    - PR-AUC improves by at least min_improvement threshold
    - Recall@1%FPR improves by at least min_improvement threshold
    if no production model exists always promote.
    """
    if prod_metrics is None:
        logger.info("No production model to compare against — promoting to staging")
        return True
 
    min_improvement = params.get("model_registry", {}).get("min_improvement", 0.005)
 
    pr_auc_gain = new_info["test_pr_auc"] - prod_metrics["pr_auc"]
    recall_gain= new_info["test_recall_at_1pct_fpr"] - prod_metrics["recall_at_1pct_fpr"]
 
    logger.info("Model comparison:")
    logger.info(f"  PR-AUC:{prod_metrics['pr_auc']:.4f} -> {new_info['test_pr_auc']:.4f} (Δ {pr_auc_gain:+.4f})")
    logger.info(f"  Recall@1%FPR: {prod_metrics['recall_at_1pct_fpr']:.4f} -> {new_info['test_recall_at_1pct_fpr']:.4f} (Δ {recall_gain:+.4f})")
 
    if pr_auc_gain >= min_improvement and recall_gain >= min_improvement:
        logger.info(f"New model improves on both metrics by > {min_improvement}  promoting to staging")
        return True
    elif pr_auc_gain < 0 or recall_gain < 0:
        logger.warning("New model regresses on one or more metrics  NOT promoting")
        return False
    else:
        logger.info(
            f"Improvement below threshold ({min_improvement})  NOT promoting. "
            "Current production model retained."
        )
        return False
 

#registration
def register_model(
    client: MlflowClient,
    run_id: str,
    model_name: str,
    registered_model_name: str,
    new_info: dict,
    params: dict
) -> str:
    """
    register model version in MLflow registry.
    tags version with all relevant metrics for audit trail.
    transitions to staging if promotion criteria met.
 
    returns registered model version number.
    """
    logger.info(f"Registering model '{registered_model_name}'...")
    
    model_build_dir = Path(params["storage"]["model_build_dir"])
    pkl_files = list(model_build_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No. pkl found in {model_build_dir}")
    
    model_path = pkl_files[0]
    logging.info(f"Logging model from {model_path}")

    with open(model_path, "rb") as x:
        model = pickle.load(x)
    
    with mlflow.start_run(run_id=run_id):
        mlflow.lightgbm.log_model(model['model'], "model")
    logger.info("Model artifact logged")
    
    model_uri = f"runs:/{run_id}/model"
    try:
        client.create_registered_model(registered_model_name)
    except Exception:
        pass

    model_version = client.create_model_version(
        name=registered_model_name,
        source=model_uri, 
        run_id=run_id)
    

    logger.info(f"Registered: {registered_model_name} v{model_version.version}")
    
    #tag with full metric audit trail
    tags = {
        "model_type":model_name,
        "test_pr_auc": str(round(new_info["test_pr_auc"], 4)),
        "test_roc_auc": str(round(new_info["test_roc_auc"], 4)),
        "test_recall_at_1pct_fpr":str(round(new_info["test_recall_at_1pct_fpr"], 4)),
        "calibration_error": str(round(new_info["calibration_error"], 4)),
        "optimal_threshold":str(round(new_info["optimal_threshold"], 4)),
        "dataset":params.get("model_registry", {}).get("dataset_name", "IBM_HI_Medium"),
        "promoted_by": "pipeline",
        "approval_required_for_prod": "true",  # staging only prod needs human approval
    }
 
    for key, value in tags.items():
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version.version,
            key=key,
            value=value
        )
 
    return model_version.version
 
 
def transition_to_staging(
    client: MlflowClient,
    registered_model_name: str,
    version: str
) -> None:
    """
    transition model version to Staging.
    staging = validated, ready for human review before production.
    production transition is manual — requires model risk sign-off.
    """
    try:
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="staging",
            version=version
        )
        logger.info(
            f"Model '{registered_model_name}' v{version} → Staging ✓"
        )
        logger.info(
            "Next step: Human model risk review required before promoting to Production. "
            "Use: mlflow models promote --name <model> --version <version> --stage Production"
        )
    except Exception as e:
        logger.warning(f"Could not set staging alias: {e}")
        logger.warning("Model registered but not aliased — check MLflow registry manually")
 
 
def save_registry_output(
    version: str,
    registered_model_name: str,
    new_info: dict,
    promoted: bool,
    reports_dir: Path
) -> None:
    """Save registry result for CI/CD pipeline to read."""
    output = {
        "registered_model_name": registered_model_name,
        "version": version,
        "run_id": new_info["run_id"],
        "promoted_to_staging":promoted,
        "test_pr_auc":new_info["test_pr_auc"],
        "test_recall_at_1pct_fpr": new_info["test_recall_at_1pct_fpr"],
        "optimal_threshold": new_info["optimal_threshold"],
    }
    out_path = reports_dir / "registry_output.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    logger.info(f"Registry output saved -> {out_path}")
 
 

# main
def main():
    logger.info("=" * 60)
    logger.info(" Stage 6: Model Registry")
    logger.info("=" * 60)
 
    try:
        params= load_params("params.yaml")
        reports_dir = Path(params["storage"].get("reports_dir", "reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)
 
        registered_model_name = params.get("model_registry", {}).get("reg_model_name")
 
        #load experiment info from evaluation stage
        new_info = load_experiment_info(reports_dir)
 
        # setup MLflow
        client = setup_mlflow(params)
 
        # compare against current production model
        prod_metrics = get_production_metrics(client, registered_model_name)
        promote = should_promote(new_info, prod_metrics, params)
 
        # always register  even if not promoting
        # every trained model should have a version record for audit
        version = register_model(
            client,
            run_id=new_info["run_id"],
            model_name=new_info["model_name"],
            registered_model_name=registered_model_name,
            new_info=new_info,
            params=params
        )
 
        if promote:
            transition_to_staging(client, registered_model_name, version)
        else:
            logger.info(
                f"Model registered as v{version} but NOT promoted to staging. "
                "Current production model retained."
            )
 
        save_registry_output(version, registered_model_name, new_info, promote, reports_dir)
 
        logger.info("=" * 60)
        logger.info("Model registry complete")
        logger.info(f"Registered: {registered_model_name} v{version}")
        logger.info(f"Promoted to staging: {promote}")
        logger.info("=" * 60)
 
    except Exception as e:
        logger.error(f"Model registry failed: {e}")
        raise
 
 
if __name__ == "__main__":
    main()
 