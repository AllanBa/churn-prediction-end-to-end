import os
import logging
import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int):
        super(ChurnMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Dropout(0.3), 
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.network(x)

def load_and_preprocess_data(filepath: str):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        df.dropna(subset=["Total Charges"], inplace=True)
        
    target = "Churn Value"
    drop_cols = [target, "Churn Label", "Churn Reason", "CustomerID", "Count", "Lat Long", "Latitude", "Longitude", "Country", "State", "City", "Zip Code"]
    
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[target].values
    
    num_features = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV", "Churn Score"]
    cat_features = [col for col in X.columns if col not in num_features]
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_features)
    ])
    
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor

def train_and_evaluate_mlp(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metrics = {"roc_auc": [], "pr_auc": [], "f1": []}
    
    epochs = 100
    batch_size = 64
    learning_rate = 0.001
    patience = 10 
    
    input_dim = X.shape[1]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"--- Training Fold {fold + 1} ---")
        
        X_train, y_train = torch.FloatTensor(X[train_idx]), torch.FloatTensor(y[train_idx]).unsqueeze(1)
        X_val, y_val = torch.FloatTensor(X[val_idx]), torch.FloatTensor(y[val_idx]).unsqueeze(1)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        
        model = ChurnMLP(input_dim)
        criterion = nn.BCELoss() 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            probs = model(X_val).numpy()
            preds = (probs > 0.5).astype(int)
            
            metrics["roc_auc"].append(roc_auc_score(y_val.numpy(), probs))
            metrics["pr_auc"].append(average_precision_score(y_val.numpy(), probs))
            metrics["f1"].append(f1_score(y_val.numpy(), preds))
            
    return {k: np.mean(v) for k, v in metrics.items()}

def run_nn_experiment():
    data_path = os.path.join("data", "raw", "telco_churn.csv")
    X, y, preprocessor = load_and_preprocess_data(data_path)
    
    mlflow.set_experiment("Telco_Churn_Baselines")
    
    logger.info("Starting PyTorch MLP Training...")
    with mlflow.start_run(run_name="MLP_PyTorch"):
        mlflow.log_param("model_type", "PyTorch_MLP")
        mlflow.log_param("hidden_layers", [64, 32])
        mlflow.log_param("activation", "ReLU")
        mlflow.log_param("optimizer", "Adam")
        
        avg_metrics = train_and_evaluate_mlp(X, y)
        
        mlflow.log_metric("val_roc_auc", avg_metrics["roc_auc"])
        mlflow.log_metric("val_pr_auc", avg_metrics["pr_auc"])
        mlflow.log_metric("val_f1_score", avg_metrics["f1"])
        
        logger.info(f"MLP PyTorch Results -> ROC-AUC: {avg_metrics['roc_auc']:.4f} | PR-AUC: {avg_metrics['pr_auc']:.4f} | F1: {avg_metrics['f1']:.4f}")
        

        final_model = ChurnMLP(X.shape[1])
        final_model.eval() 
        
        example_input = X[:5].astype(np.float32) 
        
        mlflow.pytorch.log_model(
            pytorch_model=final_model, 
            artifact_path="model",
            input_example=example_input,
            serialization_format="pickle"
        )
        os.makedirs("models", exist_ok=True)
        
        # 1. Save the fitted Scikit-Learn preprocessor
        joblib.dump(preprocessor, "models/preprocessor.pkl")
        
        # 2. Save the PyTorch model state and expected input dimension
        model_data = {
            "input_dim": X.shape[1],
            "state_dict": final_model.state_dict()
        }
        torch.save(model_data, "models/churn_mlp.pth")
        logger.info("Productions artifacts successfully saved to /models directory.")
if __name__ == "__main__":
    run_nn_experiment()