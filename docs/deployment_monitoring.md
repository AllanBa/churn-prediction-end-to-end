# Deployment Architecture & Monitoring Plan

## 1. Deployment Architecture: Real-Time Inference
Given the business requirement to empower Customer Support Agents during active calls and trigger instant marketing incentives, a batch-processing architecture is insufficient. We have opted for a **Real-Time Synchronous Inference Architecture**.

### 1.1 Components & Flow
1. **Client Application:** A Customer Success CRM or Marketing Automation tool sends an HTTP POST request containing customer data.
2. **API Gateway / Load Balancer:** Routes the request, ensuring high availability and handling traffic spikes.
3. **Inference Service (FastAPI):** - Validates incoming payload using strict **Pydantic** schemas.
   - Converts the JSON payload into a Pandas DataFrame.
   - Loads the serialized PyTorch MLP model (via MLflow `pickle` artifact).
   - Executes the forward pass and returns the binary churn prediction and probability score.
4. **Telemetry Logging:** An asynchronous background task logs the input features, the prediction, and the API latency to a central logging system (e.g., Elasticsearch or AWS CloudWatch).

### 1.2 Cloud Infrastructure (Optional Extension)
To deploy this in a cloud environment (e.g., AWS):
* **Compute:** Containerized via Docker and deployed on AWS ECS (Elastic Container Service) or AWS App Runner for fully managed scaling.
* **Model Registry:** The trained PyTorch model resides in an S3 bucket, pulled by the container during startup via MLflow tracking URI.

---

## 2. Monitoring Plan (Observability)

Machine Learning models degrade silently over time. To ensure reliability, we will monitor three distinct pillars: System Health, Data Quality, and Model Performance.

### Pillar 1: System Health (Infrastructure SLOs)
* **Metric:** API Latency (Inference time).
  * **Alert Threshold:** $P_{95}$ latency > 100ms.
* **Metric:** API Error Rate (HTTP 500s or Validation 422s).
  * **Alert Threshold:** Error rate > 1% over a 5-minute rolling window.
* **Playbook Action:** Scale up containers, check memory utilization, or rollback to the previous stable API version.

### Pillar 2: Data Drift (Input Degradation)
* **Metric:** Population Stability Index (PSI) on critical features (`Monthly Charges`, `Tenure Months`, `CLTV`).
* **Trigger:** Compare the distribution of incoming API requests (over a 7-day window) against the original training dataset.
* **Alert Threshold:** PSI > 0.2 (Significant drift detected).
* **Playbook Action:** Investigate business changes (e.g., Did a new pricing tier launch?). If the business reality changed, trigger a model retraining pipeline.

### Pillar 3: Concept Drift (Performance Degradation)
* **Metric:** F1-Score and PR-AUC.
* **Mechanism:** Requires a "Feedback Loop". Once a month, the CRM database is queried to establish ground truth (did the predicted customers actually churn?).
* **Alert Threshold:** F1-Score drops below 0.75 (Baseline acceptable performance).
* **Playbook Action:** Immediate automated retraining (CI/CD pipeline) using the latest 3 months of data, tracking the new experiment in MLflow before shadow deployment.

---

## 3. Incident Response Playbook
If a critical alert is triggered (e.g., Model predicting 100% churn due to a pipeline bug):
1. **Acknowledge:** On-call ML Engineer acknowledges the alert via PagerDuty/Slack.
2. **Mitigate (Circuit Breaker):** Switch the API response to a safe heuristic fallback (e.g., "Return a baseline churn probability of 0.2 for all users based on historical averages") to prevent catastrophic marketing spend.
3. **Investigate:** Analyze the structured logs to identify if the issue is a data schema change or an internal PyTorch execution error.
4. **Resolve & Post-Mortem:** Deploy the fix and document the root cause to prevent recurrence.