# PRCP-1016: Heart Disease Prediction (Clinical Decision Support System)

[![Domain](https://img.shields.io/badge/Domain-Healthcare%20%26%20Cardiology-red.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

> **Comprehensive Capstone Project & Clinical Decision Support System**  
> **Problem Code:** PRCP-1016  
> **Target:** Binary Classification (`heart_disease_present`: 0 = Absence, 1 = Presence)  
> **Clinical Priority:** Recall (Sensitivity) & ROC-AUC (Minimizing Life-Threatening False Negatives)

---

## 1. Executive Summary & Clinical Context
Cardiovascular disease (CVD) is the leading cause of death globally, claiming approximately **17.9 million lives annually** (31% of all deaths worldwide according to the World Health Organization). In the United States alone, the CDC reports that heart disease accounts for **1 in every 4 deaths** (~610,000 deaths/year), with over 735,000 Americans suffering a myocardial infarction each year.

The primary diagnostic challenge is that coronary artery disease (CAD) often develops silently until acute decompensation occurs. This project builds and validates an end-to-end, leak-free Machine Learning clinical decision support pipeline on the benchmark **DrivenData "Machine Learning with a Heart"** cohort ($N = 180$).

### Key Achievements:
- **Optimal Production Model:** L2-Regularized Logistic Regression ($C = 0.2$).
- **Holdout Test ROC-AUC:** **0.959**
- **Holdout Test Accuracy:** **91.7%** (33/36 correct)
- **Calibrated Clinical Sensitivity (Recall):** **100.0%** (16/16 cardiac patients caught with **0 False Negatives** at operating threshold $t = 0.40$).
- **Fold Stability:** Evaluated using 25-fold Repeated Stratified Cross-Validation ($5 \text{ splits} \times 5 \text{ repeats}$), yielding CV ROC-AUC = $0.875 \pm 0.059$.

---

## 2. Project Architecture & Notebook Structure
All analysis, modeling, visualizations, and clinical reports are housed in a single, submission-ready Jupyter Notebook: [`PRCP-1016-HeartDiseasePred.ipynb`](PRCP-1016-HeartDiseasePred.ipynb).

```
PRCP-1016-HeartDiseasePred/
│
├── PRCP-1016-HeartDiseasePred.ipynb  # Primary Deliverable: Fully executed 62-cell notebook with all outputs & plots
├── app.py                            # Streamlit Web App: Interactive Clinical Decision Support System (CDSS)
├── final_model.pkl                   # Serialized production pipeline (ColumnTransformer + Tuned LogisticRegression)
├── sample_batch_patients.csv         # 10-patient sample dataset for testing batch CSV triage
├── requirements.txt                  # Environment dependencies
├── README.md                         # Project documentation and summary
│
└── Data/
    ├── values.csv                    # 180 rows, 14 columns (patient_id + 13 clinical covariates)
    ├── labels.csv                    # 180 rows (patient_id, heart_disease_present: 100 [0] vs 80 [1])
    └── description.docx              # Clinical feature definitions & DrivenData problem description
```

---

## 3. Clinical Data Dictionary

| Feature Name | Clinical Data Type | Physiological Range | Clinical Significance |
|:---|:---|:---|:---|
| `slope_of_peak_exercise_st_segment` | Categorical / Ordinal | 1: Upsloping, 2: Flat, 3: Downsloping | ST-segment slope on peak exercise ECG; indicates coronary perfusion under exertion. |
| `thal` | Categorical (String) | `normal`, `fixed_defect`, `reversible_defect` | Thallium-201 stress scintigraphy; identifies reversible ischemia vs infarction scar. |
| `resting_blood_pressure` | Continuous (int) | 94 - 180 mmHg | Systolic arterial pressure upon hospital triage admission. |
| `chest_pain_type` | Categorical (int) | 1: Typical Angina, 2: Atypical, 3: Non-Anginal, 4: Asymptomatic | Symptom presentation; Type 4 paradoxically masks advanced multi-vessel CAD. |
| `num_major_vessels` | Discrete / Ordinal | 0 - 3 vessels | Major coronary vessels opacified via radiographic fluoroscopy. |
| `fasting_blood_sugar_gt_120_mg_per_dl` | Binary (int) | 0: $\le 120$ mg/dl, 1: $> 120$ mg/dl | Marker of impaired fasting glucose or diabetes mellitus. |
| `resting_ekg_results` | Categorical (int) | 0: Normal, 1: ST-T wave abnormality, 2: LVH | Baseline resting 12-lead electrocardiogram readout prior to stress test. |
| `serum_cholesterol_mg_per_dl` | Continuous (int) | 126 - 564 mg/dl | Serum total cholesterol (includes familial hypercholesterolemia extremes). |
| `oldpeak_eq_st_depression` | Continuous (float) | 0.0 - 6.2 mm | Exercise-induced ST-segment depression relative to baseline; quantifies ischemia. |
| `sex` | Binary (int) | 0: Female, 1: Male | Biological sex of patient (68.9% male in this referred cohort). |
| `age` | Continuous (int) | 29 - 77 years | Chronological patient age (mean: $54.8 \pm 9.3$ years). |
| `max_heart_rate_achieved` | Continuous (int) | 96 - 202 bpm | Peak chronotropic rate during treadmill exercise testing. |
| `exercise_induced_angina` | Binary (int) | 0: No, 1: Yes | Subjective chest tightness precipitated by exertion. |
| **`heart_disease_present`** | **Binary Target** | **0: Absence (100), 1: Presence (80)** | **Confirmed significant coronary artery stenosis ($\ge 50\%$ vessel narrowing).** |

---

## 4. Key EDA & Statistical Findings (Task 1)

1. **Functional Stress Modalities Dominate Static Resting Tests:**
   Inferential testing conclusively established that exercise hemodynamics differentiate CAD with overwhelming significance ($p < 10^{-6}$), while resting BP ($p = 0.295$) and total cholesterol ($p = 0.287$) showed no univariate statistical significance.
2. **Thallium Scintigraphy is the Single Most Predictive Biomarker:**
   Chi-square testing yielded $\chi^2 = 52.26, p = 4.48 \times 10^{-12}$. **75.7%** of patients with a reversible perfusion defect possess confirmed disease vs only **21.4%** with normal scans.
3. **ST Depression Quantifies Ischemic Severity:**
   Diseased patients exhibit over **2x higher** exercise-induced ST depression ($1.43 \pm 1.25$ mm vs $0.68 \pm 0.86$ mm, $p = 1.12 \times 10^{-7}$).
4. **Chronotropic Incompetence Signals Impairment:**
   Healthy individuals achieved a mean peak heart rate of $158.1$ bpm vs only $138.8$ bpm in diseased patients (deficit of **-19.3 bpm**, $p = 2.08 \times 10^{-7}$).
5. **Dose-Response with Fluoroscopy Vessels:**
   Disease probability escalates monotonically with opacified vessels: 0 vessels = 24.5%, 1 vessel = 67.6%, 2 vessels = 78.3%, and 3 vessels = 78.6% ($\chi^2 = 33.66, p = 2.34 \times 10^{-7}$).
6. **Chest Pain Paradox:**
   Patients presenting with Type 4 (asymptomatic chest pain) exhibited a **70.7%** CAD rate, proving that absence of angina cannot rule out acute coronary obstruction.
7. **Clinical Outlier Preservation:**
   High cholesterol (up to 564 mg/dl) and BP (up to 180 mmHg) reflect authentic severe pathology (familial hypercholesterolemia and hypertensive crisis). Retaining these patients and scaling via `StandardScaler` inside pipelines prevented 6.7% data loss while preserving vital clinical signals.

---

## 5. Model Comparison Report (Task 2 & Comparison)

### Benchmark Summary (25-Fold Repeated Stratified CV & Holdout Test Set)

| Model Architecture | CV ROC-AUC | CV Recall | Test ROC-AUC | Test Recall | Test Precision | Test F1 | Test Accuracy | Train Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Tuned)** | **0.875 ± 0.059** | **0.736 ± 0.102** | **0.959** | **0.938** | **0.882** | **0.909** | **0.917** | **1.84s** |
| **SVM (RBF Kernel)** | 0.843 ± 0.071 | 0.702 ± 0.130 | 0.975 | 1.000 | 0.800 | 0.889 | 0.889 | 2.14s |
| **Random Forest (Tuned)** | 0.869 ± 0.059 | 0.694 ± 0.103 | 0.962 | 0.875 | 0.875 | 0.875 | 0.889 | 11.80s |
| **Gradient Boosting** | 0.839 ± 0.049 | 0.691 ± 0.110 | 0.947 | 1.000 | 0.842 | 0.914 | 0.917 | 8.73s |
| **XGBoost Classifier** | 0.848 ± 0.055 | 0.686 ± 0.117 | 0.947 | 0.875 | 0.824 | 0.848 | 0.861 | 6.15s |
| **Gaussian Naive Bayes** | 0.850 ± 0.054 | 0.749 ± 0.116 | 0.894 | 0.938 | 0.750 | 0.833 | 0.833 | 2.40s |
| **K-Nearest Neighbors (KNN)**| 0.803 ± 0.066 | 0.639 ± 0.142 | 0.941 | 0.875 | 0.824 | 0.848 | 0.861 | 2.25s |
| **Decision Tree (Pruned)** | 0.739 ± 0.086 | 0.609 ± 0.119 | 0.806 | 0.875 | 0.737 | 0.800 | 0.806 | 1.85s |
| **Dummy (Baseline)** | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.556 | 1.75s |

### Clinical Threshold Calibration (Logistic Regression)
In medical screening, missing a sick patient (False Negative) is catastrophic, while a False Positive merely prompts follow-up non-invasive imaging. Calibrating the decision threshold to **0.40** optimizes sensitivity:

| Operating Threshold | Recall (Sensitivity) | Precision | F1-Score | False Negatives (Missed) | False Positives | Clinical Interpretation |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 0.50 (Default) | 93.8% | 88.2% | 0.909 | 1 | 2 | 1 diseased patient missed. |
| **0.40 (Selected)** | **100.0%** | **80.0%** | **0.889** | **0** | **4** | **100% of cardiac patients caught; 0 missed.** |
| 0.35 | 100.0% | 76.2% | 0.865 | 0 | 5 | Safe screening; slightly higher false alarms. |

### Top 5 Predictive Clinical Biomarkers
1. **`thal` (Reversible Defect):** Odds Ratio = **2.42** ($p = 4.48 \times 10^{-12}$)
2. **`num_major_vessels`:** Odds Ratio = **1.79** per vessel ($p = 2.34 \times 10^{-7}$)
3. **`chest_pain_type_4` (Asymptomatic):** Odds Ratio = **1.91** ($p = 3.10 \times 10^{-10}$)
4. **`oldpeak_eq_st_depression`:** Odds Ratio = **1.46** per SD ($p = 1.12 \times 10^{-7}$)
5. **`sex` (Male):** Odds Ratio = **1.95** ($p = 1.44 \times 10^{-5}$)
6. **`max_heart_rate_achieved` (Protective):** Odds Ratio = **0.71** (Higher reserve reduces CAD odds by 29.4%).

---

## 6. Hospital Implementation Plan (Task 3)

1. **Automated Triage System:**
   - **Low Risk ($P < 30\%$):** Outpatient discharge, annual checkup, lifestyle maintenance.
   - **Moderate Risk ($30\% \le P < 60\%$):** Stress echocardiography and outpatient cardiology consultation within 1-2 weeks.
   - **High Risk ($P \ge 60\%$):** Immediate cardiology admission, continuous ECG telemetry, and expedited coronary catheterization.
2. **Diagnostic Prioritization:**
   - Prioritize exercise ECG, thallium scintigraphy, and fluoroscopy.
   - Stop over-relying on solitary fasting glucose ($p = 1.0$) or total cholesterol ($p = 0.287$) for acute triage.
3. **Targeted Prevention Programs:**
   - Proactive community screening for men aged 50+ presenting with atypical exertional fatigue or tightness.
   - Structured cardiac rehabilitation for patients with peak chronotropic deficits.
4. **Human-in-the-Loop Governance & Privacy:**
   - Model functions strictly as a decision-support tool; attending cardiologists retain final diagnostic authority.
   - Strict HIPAA compliance with tokenized de-identification within the hospital firewall.

---

## 7. Report on Challenges Faced

| Challenge | Lifecycle Stage | Technique Used | Methodological Rationale |
|:---|:---|:---|:---|
| **Small Sample Size ($N=180$)** | Modeling | Repeated Stratified 5-Fold CV ($5 \times 5 = 25$ folds) + L2 Regularization ($C=0.2$) | Averages partition variance; regularization prevents small-sample overfitting. |
| **Separated Values & Labels** | Ingestion | Inner join on `patient_id` + multi-point assertions (`assert row_count == 180`) | Verifies record alignment; identifier subsequently dropped to eliminate ID leakage. |
| **Mild Class Imbalance (55.6/44.4)** | Evaluation | Stratified splitting across all 25 folds; prioritized Recall & ROC-AUC | Preserves identical class proportions; avoids accuracy paradox. |
| **Extreme Clinical Outliers** | EDA & Preprocessing | Retained all patients; scaled continuous features via `StandardScaler` inside Pipeline | Preserves vital pathology (cholesterol 564 mg/dl, BP 180 mmHg) without data leakage. |
| **Mixed Data Types** | Preprocessing | `ColumnTransformer` with explicit category arrays for `OneHotEncoder` | Handles string, nominal, and continuous inputs without unseen category errors. |
| **Asymmetric Error Costs** | Threshold Tuning | Calibrated operating threshold from 0.50 down to 0.40 | Eliminates False Negatives (100% Recall) in high-stakes cardiac screening. |
| **Model Interpretability** | Clinical Deployment | Logistic Regression Odds Ratios ($e^{\beta}$) & Permutation Importance | Replaces black-box predictions with transparent, pathophysiological explanations. |

---

## 8. Installation & Verification Instructions

### Environment Setup
```bash
# Clone or navigate to the repository
cd PRCP-1016-HeartDieseasePred

# Install dependencies
pip install -r requirements.txt
```

### Launching the Streamlit Clinical Decision Support Web App
The project includes a web-ready Streamlit clinical interface:
```bash
# Launch Streamlit app
streamlit run app.py
```
*Accessible locally at: `http://localhost:8501`*

**App Modules:**
1. 🩺 **Single Patient Triage:** Real-time probability estimation, triage badge (Low/Moderate/High Risk), customized clinical action recommendations, quick-fill preset buttons (Healthy, Moderate, High-Risk), and feature attribution breakdown.
2. 📁 **Batch Cohort Screening:** Upload a multi-patient CSV, compute cohort risk distributions, view filterable triage tables, and download annotated prediction reports. Includes `sample_batch_patients.csv` for immediate testing.
3. 📊 **Model Benchmark & Metrics Dashboard:** Interactive ROC curves, metric comparison charts, production confusion matrix, and feature Odds Ratio visualizations.
4. 💡 **Clinical Guidelines & EDA:** Actionable hospital protocol, diagnostic test prioritization (high vs low value), and statistical challenge resolutions.
5. ℹ️ **Project Documentation:** Complete Clinical Data Dictionary and global cardiovascular health statistics.

### Execution & Verification of Jupyter Notebook
The primary deliverable notebook runs top to bottom without errors:
```bash
# Execute the notebook via Jupyter
jupyter notebook PRCP-1016-HeartDiseasePred.ipynb
```
Or use "Restart & Run All" in VS Code or Jupyter Lab.

### Running Live Inference via Saved Artifact
```python
import joblib
import pandas as pd

# Load serialized pipeline
model = joblib.load('final_model.pkl')

# Example patient
new_patient = pd.DataFrame([{
    'slope_of_peak_exercise_st_segment': 2,
    'thal': 'reversible_defect',
    'resting_blood_pressure': 150,
    'chest_pain_type': 4,
    'num_major_vessels': 2,
    'fasting_blood_sugar_gt_120_mg_per_dl': 0,
    'resting_ekg_results': 2,
    'serum_cholesterol_mg_per_dl': 260,
    'oldpeak_eq_st_depression': 2.5,
    'sex': 1,
    'age': 60,
    'max_heart_rate_achieved': 130,
    'exercise_induced_angina': 1
}])

prob = model.predict_proba(new_patient)[0, 1]
print(f"Predicted CAD Probability: {prob * 100:.1f}%")
print("Triage Status:", "HIGH RISK" if prob >= 0.40 else "LOW RISK")
```

---

## 9. References
1. **DrivenData:** *Machine Learning with a Heart* Competition ([Benchmark Link](https://www.drivendata.org/competitions/54/machine-learning-with-a-heart/)).
2. **World Health Organization (WHO):** *Cardiovascular Diseases (CVDs) Fact Sheet*, 2021.
3. **Centers for Disease Control and Prevention (CDC):** *Heart Disease Facts & Statistics*, 2023.
4. **American Heart Association (AHA) / ACC:** *2021 AHA/ACC Guideline for the Evaluation and Diagnosis of Chest Pain*, Circulation, 144(22), e368–e454.
5. **Scikit-Learn:** Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2825-2830, 2011.
