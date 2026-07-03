# Model Card: Deep Learning for Telecom Customer Churn Prediction

## 1. Model Details
* **Developer:** Allanderson Barros
* **Model Date:** July 2026
* **Model Version:** 1.0.0
* **Model Type:** Multi-Layer Perceptron (MLP) / Feedforward Artificial Neural Network.
* **Framework:** PyTorch 2.x
* **Architecture Design:** * Input Layer: Dynamically sized based on feature engineering output.
  * Hidden Layer 1: 64 neurons, Batch Normalization (`BatchNorm1d`), ReLU activation, 30% Dropout.
  * Hidden Layer 2: 32 neurons, Batch Normalization (`BatchNorm1d`), ReLU activation, 20% Dropout.
  * Output Layer: 1 neuron, Sigmoid activation (outputs probability $P \in [0, 1]$).
* **Training Hyperparameters:** Optimizer: Adam | Learning Rate: 0.001 | Batch Size: 64 | Maximum Epochs: 100.
* **Regularization:** Automated Early Stopping (Patience = 10 epochs) to prevent overfitting on the training set.

## 2. Intended Use
* **Primary Use Case:** Predict the probabilistic risk of a telecommunications customer canceling their service contract within the next billing cycle.
* **Primary Consumers:** * **Marketing Microservices:** Automated systems consuming the API to trigger personalized retention discounts.
  * **Customer Success Dashboard:** Real-time evaluation during customer support calls to empower agents with retention tools.
* **Out-of-Scope Use Cases:** This model must **not** be used for automated credit denial, service degradation, or price discrimination based on demographic features. 

## 3. Data Lineage & Feature Engineering
* [cite_start]**Source Dataset:** Telecom Customer Churn dataset[cite: 71, 74].
* **Volume:** 7,043 historical customer profiles.
* [cite_start]**Target Definition:** `Churn Value` (Binary: 1 = Churned, 0 = Retained)[cite: 72].
* **Data Leakage Prevention:** The `Churn Reason` and `Churn Label` columns were strictly removed prior to pipeline execution to ensure the model does not train on post-event indicators.
* **Preprocessing Pipeline (Scikit-Learn):**
  * Numerical attributes (`Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV`, `Churn Score`) scaled via `StandardScaler` to stabilize gradient descent.
  * Categorical attributes (`Gender`, `Contract`, `Internet Service`, etc.) transformed via `OneHotEncoder`.
  * Geolocation data dropped in version 1.0 to reduce dimensionality noise.

## 4. Evaluation Strategy & Performance Metrics
Due to the inherent class imbalance of the dataset (~27% overall churn rate), pure Accuracy was discarded as a metric. The model was evaluated using **5-Fold Stratified Cross-Validation**, ensuring proportional representation of churners in every training fold.

**Final Cross-Validation Results (PyTorch MLP):**
* **ROC-AUC:** `0.9745` (Excellent ability to distinguish between classes at various thresholds).
* **PR-AUC (Precision-Recall AUC):** `0.9362` (Highly robust performance even on the minority class).
* **F1-Score (Macro):** `0.8483` (Strong harmonic mean of Precision and Recall).

**Baseline Comparison:** The deep learning architecture vastly outperformed the linear baselines (Logistic Regression F1-Score: ~0.24), demonstrating the non-linear complexity of customer behavioral triggers.

## 5. Ethical Considerations & Fairness
* **Demographic Sensitivity:** The training data includes protected/sensitive attributes such as `Gender` and age indicators (`Senior Citizen`). 
* **Disparate Impact Risk:** There is a theoretical risk that the network might correlate specific demographic profiles with a lower likelihood of receiving retention incentives (False Negatives).
* **Mitigation Recommendation:** Marketing execution teams must establish a "Fairness Loop," randomly auditing a subset of False Positives and False Negatives quarterly to ensure retention budgets are distributed equitably.

## 6. Caveats and Limitations
* **Concept Drift Vulnerability:** The network places heavy mathematical weight on financial inputs like `Monthly Charges` and `CLTV`. If the business alters its pricing tiers, introduces new products, or faces a major macroeconomic shift, the model's accuracy will rapidly degrade.
* **Explainability (Black Box):** Unlike linear regression, the PyTorch MLP does not natively output feature coefficients. If stakeholders require exact reasoning for a prediction, post-hoc explainability tools (e.g., SHAP or LIME) must be integrated into the inference pipeline.
* **Thresholding:** The default classification threshold is set to `0.5`. Production usage should implement a dynamic threshold optimized against the actual business cost of a False Positive (wasted discount) versus a False Negative (lost customer revenue).