# Promotion Eligibility and Bias Analysis.
---

## Executive summary
The promotion dataset is highly imbalanced: only 8.46% of staff were promoted (3,241 promoted vs 35,071 not promoted; imbalance ≈ 10.8:1). Most important predictors of promotion (consistent across correlation, regression coefficients, and SHAP):  
* Training_score_average (top predictor)
* Targets_met
* Previous_Award (past recognition)
* Last_performance_score
* Division (Commercial Sales & Marketing and People/HR positively associated; several divisions negatively associated).  

Models:
* Gradient Boosting gives the best precision (0.94) and accuracy (0.94), but low recall (0.34).
* SVC and Random Forest give high recall (0.84), but poor precision (~ 0.20–0.22).  

This means there is a trade-off between avoiding false positives (GB) and capturing most promotable staff (SVC/RF).  

Fairness audits performed for Gender, Marital status, and State of Origin show no evidence of a legally significant adverse impact under commonly-used thresholds (4/5ths rule):  
* Gender AIR ≈ 0.930 (female selection_rate 0.985, male 0.916) → above 0.8 threshold.  
* Marital AIR ≈ 0.992 → very close to parity.  
* State of Origin AIR ≈ 0.922 → very close to parity.  

However, EDA shows skewed counts by Division, State, Gender, Qualifications, and Recruitment Channel (agency vs referred), which may create perceptions of bias. Absolute counts can be misleading — selection rates matter.  

In conclusion, based on the available metrics, there is no strong statistical evidence of systematic bias against Gender, Marital Status, or State of Origin under standard tests — but important signs of disparate outcomes across divisions, locations and recruitment channels warrant procedural fixes, better data, and continued monitoring.

---

## Table of contents  
[Problem statement](#problem-statement)  
[Aim and Objectives](#aim-and-objectives)  
[Skills Demonstrated](#skills-demonstrated)  
[Data and preprocessing](#data-and-preprocessing)  
[Highlighted results from EDA](#highlighted-results-from-eda)    
[Modeling and evaluation](#modeling-and-evaluation)  
[Feature importance](#feature-importance)  
[Fairness audit and interpretation](#fairness-audit-and-interpretation)  
[Business interpretation](#business-interpretation)  
[Limitations and caveats](#limitations-and-caveats)  
[Recommendations](#recommendations)  
[Conclusion](#conclusion) 

---

## Problem statement
Employees raised concerns that promotions are skewed and biased. The business question: “Is the current promotion process biased? Which features should be used to make fair, explainable promotion eligibility decisions?”

---

## Aim and Objectives
To answer staff concerns about whether promotions are biased, identify the most important features that should drive promotion eligibility, explain models and findings, and give concrete recommendations to make promotions fairer, transparent, and defensible.  

We address this by:
* Descriptive EDA to map promotion distributions by division, state, gender, qualification, staff type, etc.
* Supervised models to identify predictive features and produce explainability (coefficients + SHAP).
* Fairness audit on protected attributes available (Gender, Marital Status, and State of Origin).
* Operational recommendations for policy, transparency, and monitoring.

---

## Skills Demonstrated
* Python
* Pandas
* Seaborn
* XGBoost
* SHAP
* Fairness auditing
* Scikit-learn
* Imbalanced classification

---

## Data and preprocessing
* Dataset shape (38312, 19)
* Target: 'Promoted_or_Not'
* Out of 38,312 staff records, 35,071 were not promoted,  while  3,241 were promoted.
* 1,679 null values from the Qualifications column was replaced with the mode of that column.
* No duplicates found
* All values were standardized
* No outliers were detected
* Categorical variables encoded using Target or OneHotEncoder

---

## Highlighted results from EDA

### 1. Distribution of Promotions
* Not Promoted: 35,071 (91.54%)
* Promoted: 3,241 (8.46%)
> [!NOTE]
> **Imbalance ratio: ~ 10.8 : 1**

![target distribution](/img/target_distr.png)

### 2. Promotions by Division
* Commercial Sales and Marketing: 864 promotions (highest)
* Regulatory and Legal Services: 41 promotions (lowest)

### 3. Promotions By State of Origin
* Lagos: 534 promotions (highest)
* Jigawa: 18 promotions (lowest)  

![Promotion by division](/img/prm_by_div_sor.png)  

### 4. Promotions by Qualification / Gender / Staff type (distribution among promoted)
* Qualification: First Degree holders make up 67.2% of promotions (highest); non-university degree holders 16% (least).
* Gender among promoted: Males 68.6%, Females 31.4% (absolute counts).
* Staff category among promoted: Agency staff 55.4% (highest), Referred staff 1.2% (least).

### 5. Other notable EDA flags
* Foreign schooled staff: 91.6% (appearing in promoted group)
* Married staff among promoted: 81.3%
* Promoted staff with no past disciplinary action: 99.5%
* Promoted staff with no previous intra-department movement: 91.4%  

> [!IMPORTANT]
> **Absolute counts are heavily influenced by base rates (e.g., more males in workforce → more promoted males). These counts raise questions and justify further rate-based and statistical tests — which I further performed.**

---

## Modeling and evaluation
Main models evaluated: 
* Logistic Regression
* Gradient Boosting
* Random Forest
* Support Vector Classifier (SVC)
* Decision Tree
* KNN (SMOTE)  

![model comparison](/img/model_comp.png)

**Best precision/accuracy:** Gradient Boosting — Precision: 0.94, Accuracy: 0.94, Recall: 0.34.  
**Interpretation:** Very low false positive rate; but misses many actual promotions (low recall).  
**Best recall:** 
* SVC — Recall: 0.84, Precision: 0.22, Accuracy: 0.73.
* Random Forest — Recall: 0.84, Precision: 0.20, Accuracy: 0.70.  
**Interpretation:** Good at finding promotable staff but many false positives.  

Choice depends on business objective: Do we prefer a conservative rule (few false promotions — higher precision) or inclusive identification (find most promotable staff — higher recall)? I recommend a hybrid two-stage approach. See [Recommendations](#recommendations).

---

## Feature importance
### 1. From correlation with promotion  
Top positive correlations:
```
Targets_met                                  0.224518
Previous_Award                               0.201434
Training_score_average                       0.178448
Last_performance_score                       0.119690
Division_Information Technology...           0.031617
State_Of_Origin                              0.031455
Qualification_MSc/MBA/PhD                    0.026600
```
Lowest / negative correlations:
```
Trainings_Attended                          -0.024345
Qualification_First Degree or HND           -0.026664
Division_Commercial Sales and Marketing    -0.030213
```

**Interpretation:** Targets_met, Previous_Award, and Training_score_average show the strongest linear relationships with promotions. 
Some division dummies show small correlations (signs may differ from regression/SHAP).  

### 2. From regression model coefficients (global importance, signed)  
Top positive coefficients (encourage promotion):
```
Training_score_average                      15.260171
Division_Commercial Sales and Marketing      3.932522
Division_People/HR Management                3.416400
Targets_met                                  2.609909
```
Top negative coefficients (associated with lower promotion odds):
```
Division_Information Technology and Support  -3.131539
Division_Information and Strategy            -4.321002
Division_Research and Innovation             -4.862115
```
**Interpretation:** After accounting for other features, 
* Training_score_average has by far the largest positive coefficient in the regression — suggesting it is the most powerful numeric predictor.
* Division affiliation strongly affects odds in both directions.

![lr feature coeff](/img/lr_feat_coeff.png)  

### 3. From SHAP (XGBoost tree-based, richer explanation)  
Top features by average SHAP magnitude:
```
Training_score_average
Targets_met
Division_Commercial Sales and Marketing
Last_performance_score
```
**Interpretation:** SHAP confirms the same major drivers: training score, targets met, division, and last performance are the features with highest contribution to model predictions.  

![shap coeff](/img/shap_value_coeff.png)

Therefore, the primary features to recommend for promotion eligibility scoring (ordered):
* Training_score_average — most consistent, largest effect.
* Targets_met (achievement of targets/KPIs) — strong positive.
* Previous_Award / Recognition — signal of prior merit.
* Last_performance_score — direct performance appraisal.
* Division (not a merit metric but a strong predictor):
    - Commercial Sales & Marketing and People/HR positively associated
    - Some divisions (IT, Strategy, Research) negatively associated.
* Postgraduate Qualification (MSc/MBA/PhD) — modest positive effect.
* State of Origin — small signal but visible in EDA (with Lagos having the highest promotions).
* Recruitment Channel — with large differences in counts, should be normalised/corrected if policy shouldn’t favor one group.  

---

## Fairness audit and interpretation
### 1. Gender fairness
```
Overall selection rate: 0.936448
Overall TPR: 0.976852
Per-group selection_rate / tpr:
  Female: selection_rate = 0.985042, TPR = 0.985437
  Male: selection_rate = 0.915955, TPR = 0.972851
Selection rate ratio (min/max) = Adverse Impact Ratio (AIR): 0.9299
```
**Interpretation:**  
The AIR of ~0.93 is above the conservative 0.80 (4/5ths) threshold used in many legal and HR contexts to flag adverse impact. That implies no clear adverse impact against either gender based on the audited metric.  

Females actually appear to have a higher selection rate than males in the model's selection results — this explains perception vs reality: absolute counts may show more promoted males because there are more males employed, but rates show higher female selection probability.  

![selection rate by gender](/img/sel_rate_gender.png)  

### 2. Marital status fairness
```
Overall selection rate: 0.936448
Per-group selection_rate / tpr:
  Married: selection_rate = 0.937460, TPR = 0.973129
  Not_Sure: selection_rate = 0.939024, TPR = 1.000000
  Single: selection_rate = 0.931802, TPR = 0.991525
AIR: 0.9923
```
**Interpretation:**  
Very near parity — no adverse impact detected for marital status.  

![selection rate by marital status](/img/sel_rate_ms.png)  

### 3. State of Origin fairness
```
Overall selection rate: 0.93644
Overall TPR: 0.97685
Selection rate ratio (min / max): 0.9229947068403908
Adverse Impact Ratio (AIR): 0.9229947068403908
```
**Interpretation:**  
Very near parity — no adverse impact detected for State of Origin.  

![selection rate by marital status](/img/sel_rate_sor.png)

> [!CAUTION]
> Fairness is multi-dimensional. A pass on AIR does not mean the promotion process is perfectly fair — it means there is no statistical evidence (on audited attributes/content) of adverse impact by those attributes given available features and modeling choices.

---

## Business interpretation
This section explains how my results answer the business question “Is the promotion process biased? What should determine promotion?”  

* Key merit signals strongly associated with promotion are training performance, targets met, prior awards, and recent performance scores. These are good candidates to form an objective promotion-scoring engine because they map to observable performance and skill measures.
```
Business implication: Build a promotion eligibility score that weights these merit-based metrics heavily and transparently.
```
* Division and location have material influence on promotions. This may reflect real differences in promotion opportunities (some divisions have more roles, faster turnover, or clearer promotion pipelines), or it may reflect inconsistent application of criteria.
```
Business implication: If promotions should be equally accessible across divisions, then division-adjusted thresholds, quotas,
or normalized scoring may be needed.
```
* Gender and marital status fairness audits do not indicate an adverse impact under standard metrics; BUT absolute counts and perceptions (e.g., more promoted males in counts) fuel complaints. Perception matters.
```
Business implication: Communicate rates and normalized metrics (not just counts).
Publish the promotion criteria and the score components.
```
* Model performance trade-offs: If the manager choose a model like Gradient Boosting, they minimize false promotions (high precision) but miss candidates (low recall). If the manager's goal is to find every promotable candidate (high recall), they need a different approach or an intermediate two-stage process (screen broadly then apply higher-precision review).

---

## Limitations and caveats
* Class imbalance is extreme (≈ 10.8:1). This affects model training and evaluation. Metrics must be interpreted with imbalance in mind.
* Causality vs correlation: The models identify associations, not causal effects. For example, being in Commercial Sales may correlate with promotion because that division has more roles or different appraisal processes — not because the division itself "deserves" promotion.
* Missing protected attributes: I could only audit fairness for attributes present in the data. If race/ethnicity, disability, religion, or other protected traits are relevant but not in the dataset, hidden bias may exist undetected.
* Measurement quality: Training_score_average and Last_performance_score quality depends on consistent, objective scoring across divisions and raters. If scores are noisy or biased, the model will propagate that bias.
* Label issues: The “promotion” label reflects historical managerial decisions — if the historical process was biased, a model trained on that label can reproduce bias unless explicitly corrected.

---

## Recommendations
* Publish a clear promotion framework that states the primary objective criteria and relative weighting (example: Training score 35%, Targets met 30%, Last performance 20%, Previous awards 10%, Qualifications 5%). Transparency reduces perception of bias.
* Use rate-based reporting (not just counts) in internal communications: Report selection rates by protected group (gender, marital status, state of origin), division-normalized promotion rates, and promotion rates per 100 staff to illustrate fairness.
* Establish an "Appeals" & "Audit" process: Any denied candidate can request a review - an independent panel (cross-division) reviews borderline cases.
* Two-stage selection pipeline is recommended:
      - Stage 1 (recall-oriented): Use a higher-recall model (e.g., SVC/RF ensemble) or rule-based screener to assemble a longlist of promotable staff.
      - Stage 2 (precision + human review): Apply a high-precision model (Gradient Boosting) to create a shortlist, then have a calibrated human panel review. This balances finding promotable candidates and avoiding false promotions.
* Normalize for division/role availability: When promotion slots are limited by division/structure, report promotions as rate per eligible population, and (where possible) create cross-division mobility programs.
* Improve and standardize performance measurement (training scores, appraisal rubrics).
* Quantitative KPI for promotions: Establish a promotion-dashboard for leadership with key metrics, trends, and outlier alerts.
* Collect richer demographic & contextual data (only as legally permitted) to enable broader fairness audits (age, disability, ethnicity). Ensure data privacy and legal compliance.
* Create development pipelines: Make trainings and assignments available across divisions to reduce structural inequity. Track who gets development opportunities.  

For a more technical audience...
* Adopt the consensus feature set for any promotion scoring tool: i.e. training_score_average, targets_met, previous_award, last_performance_score, qualification_level. Use division only as an adjustment or tie-breaker (and never as a replacement for merit).
* Bias mitigation: If future audits show disparities, apply reweighting or post-processing fairness techniques (e.g. equalized odds post-processing) rather than naive removal of features.
* Model monitoring and periodic audits: Track selection rates, TPR, precision, recall, and fairness metrics monthly or quarterly. Keep model versioning and a log of decisions.
* Human-in-the-loop: Never fully automate promotions. Use models to recommend and prioritize, not to decide. All final promotions should be reviewed by a calibrated, diverse panel.
* Calibration & thresholding: For score-based decisions, calibrate model output to reflect actual probabilities of promotion and set thresholds transparently. Consider different thresholds per division if promotion dynamics differ, but document the business logic clearly.

---

## Conclusion
Final verdict — Was the promotion process biased?  

Short answer: No clear statistical evidence of bias was found for Gender, Marital status, or State of Origin when audited under standard selection-rate based tests (Adverse Impact Ratio > 0.8).  

However, there are meaningful disparities in promotion counts and rates across Division, State of Origin, Recruitment Channel, and Qualification, which are legitimate matters for investigation. These disparities can drive perceptions of unfairness even if protected-group AIRs are within accepted thresholds.  

Historical processes and measurement quality can hide biases. The model inherits historical managerial decisions — if those were biased in subtle ways, the model can reproduce them unless corrected.  

Finally, the evidence does not support a claim of outright, systemic (gender, marital status, or state of origin) discrimination in promotions based on the metrics audited, but the company should act to remove ambiguity, increase transparency, normalize division differences, improve measurement, and continue fairness monitoring. That combination will reduce both real unfairness and the perception of unfairness.
