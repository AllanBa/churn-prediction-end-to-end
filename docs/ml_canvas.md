# Strategic Machine Learning Canvas: End-to-End Churn Prediction

## 1. Value Proposition & Business Context
* [cite_start]**Business Problem:** A telecommunications provider is experiencing accelerated customer attrition (churn), leading to directly compounding revenue loss[cite: 10].
* [cite_start]**Solution:** An end-to-end predictive system capable of identifying high-risk customers before contract termination, allowing proactive retention campaigns[cite: 11, 12].
* [cite_start]**Primary Stakeholders:** C-Level Executives (Business Impact Tracking), Marketing & Customer Retention Teams (End Users of Predictions), and Platform/ML Platform Engineers (API Consumers).

## 2. Value Creation & Operationalization
* **Downstream Action:** The marketing team triggers automated, personalized retention incentives (e.g., contract upgrades, targeted discounts) based on prediction segments.
* [cite_start]**Prediction Consumption:** Batch processing triggers a weekly scoring pipeline for high-value cohorts, while the real-time FastAPI endpoint handles on-demand evaluations during customer service interactions[cite: 57, 61].

## 3. Data Sources & Ingestion Topology
* **Source Systems:** Telecom CRM, billing applications, and historical customer interaction logs.
* **Data Volume:** 7,043 historical customer profiles with 33 comprehensive attributes spanning demographic, geographic, and behavioral data.
* **Target Integrity:** Ground truth is derived from the historical `Churn Value` indicator (binary 1/0), capturing verified service cancellations.

## 4. Feature Engineering & Prevention of Data Leakage
* **Predictive Signals:** Key continuous variables include `Tenure Months`, `Monthly Charges`, `Total Charges`, and business-enriched metrics like `CLTV` and real-time `Churn Score`.
* **Data Leakage Mitigation:** * The column `Churn Reason` must be completely dropped during ingestion since it is a post-hoc attribute only populated *after* the churn event occurs.
  * `CustomerID` and `Churn Label` (string text) are stripped to maintain features clean and prevent target encoding duplication.
  * Geographic attributes (`Latitude`, `Longitude`, `Zip Code`) are omitted in the baseline layer to minimize noise and variance.

## 5. Machine Learning Task Formulation
* **Task Type:** Supervised Learning — Binary Classification.
* **Model Objective:** Output a continuous probability score $P(\text{Churn} = 1 \mid X) \in [0, 1]$.
* **Decision Framework:** A dynamic thresholding mechanism maps the probability into a binary action flag (1 or 0). This threshold is optimized continuously against the business cost function rather than defaulting to 0.5.

## 6. Evaluation Framework (Technical vs. Business Metrics)
* [cite_start]**Technical Metrics:** Due to class imbalance (~27% churn rate), optimization prioritizes Area Under the Receiver Operating Characteristic (AUC-ROC), Precision-Recall AUC (PR-AUC), and the F1-Score over simple accuracy.
* [cite_start]**Business Metric (Cost of Avoided Churn):** Maximizing financial savings by minimizing False Negatives (unseen customers who leave, costing full contract value) and controlling False Positives (spending unnecessary retention budgets on loyal customers)[cite: 46, 50].

## 7. Deployment Constraints & Service Level Objectives (SLOs)
* [cite_start]**Serving Infrastructure:** High-performance REST API developed using FastAPI, structured with strict Pydantic validation schemas and robust error handling middleware[cite: 57].
* **Defined SLOs:**
  * [cite_start]**Latency:** $P_{95}$ response time must remain under 50 milliseconds per individual inference request under baseline production load.
  * **Availability:** Service uptime target of $\ge 99.9\%$.
  * **Throughput Capacity:** Designed to sustain up to 200 concurrent requests per second without degradation.

## 8. Observability & Monitoring Plan
* **Structured Logging:** Zero print statements. [cite_start]Implements a json-structured cloud logging middleware capturing latency, pipeline execution paths, and validation errors[cite: 38, 57].
* **Data Drift Tracking:** Continual monitoring of distribution shifts in critical inputs (e.g., changes in the mean of `Monthly Charges` or sudden skewness in `Tenure Months`) using Population Stability Index (PSI).
* **Concept Drift Tracking:** Monitoring degradation of the production F1-Score as customer behavior paradigms shift over quarters.

## 9. Feedback Loop & Retraining Strategy
* **Data Refresh:** Monthly ingestion of new ground-truth CRM logs to expand the training pool.
* [cite_start]**Automated CI/CD Pipeline:** Triggered retraining scheduled when performance drops below an F1-Score baseline or automatically on a 30-day cycle, tracking all parameters, artifacts, and comparative graphs inside MLflow[cite: 46, 50].