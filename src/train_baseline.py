import os
import logging
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Loads dataset and performs initial structural cleaning."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        df.dropna(subset=["Total Charges"], inplace=True)
        
    return df

def build_preprocessing_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Creates a Scikit-Learn pipeline for scaling and encoding."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def evaluate_model(model, X, y, cv_splits=5):
    """Evaluates a model using Stratified K-Fold Cross Validation."""
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    
    metrics = {"roc_auc": [], "pr_auc": [], "f1": []}
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        
        metrics["roc_auc"].append(roc_auc_score(y_val, probs))
        metrics["pr_auc"].append(average_precision_score(y_val, probs))
        metrics["f1"].append(f1_score(y_val, preds))
        
    return {k: np.mean(v) for k, v in metrics.items()}

def run_experiment():
    data_path = os.path.join("data", "raw", "telco_churn.csv")
    df = load_and_clean_data(data_path)
    
    target = "Churn Value"
    drop_cols = [
        target, "Churn Label", "Churn Reason", "CustomerID", "Count", 
        "Lat Long", "Latitude", "Longitude", "Country", "State", "City", "Zip Code"
    ]
    
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[target]
    
    numeric_features = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV", "Churn Score"]
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    logger.info("Building preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    
    models = {
        "Baseline_Dummy": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "Baseline_LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    }
    
    mlflow.set_experiment("Telco_Churn_Baselines")
    
    for model_name, classifier in models.items():
        logger.info(f"Training and evaluating {model_name}...")
        
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("cv_splits", 5)
            
            # Evaluate metrics using Cross Validation
            avg_metrics = evaluate_model(pipeline, X, y)
            
            # Log metrics
            mlflow.log_metric("val_roc_auc", avg_metrics["roc_auc"])
            mlflow.log_metric("val_pr_auc", avg_metrics["pr_auc"])
            mlflow.log_metric("val_f1_score", avg_metrics["f1"])
            
            logger.info(f"{model_name} Results -> ROC-AUC: {avg_metrics['roc_auc']:.4f} | PR-AUC: {avg_metrics['pr_auc']:.4f} | F1: {avg_metrics['f1']:.4f}")
            
            # ML ENGINEERING BEST PRACTICE: Fit on the entire dataset before saving the artifact
            logger.info(f"Retraining {model_name} on full dataset to save artifact...")
            pipeline.fit(X, y)
            

            mlflow.sklearn.log_model(
                sk_model=pipeline, 
                artifact_path="model", 
                input_example=X.iloc[:5],
                serialization_format="cloudpickle"
            )

if __name__ == "__main__":
    run_experiment()
    logger.info("Baseline training completed successfully.")