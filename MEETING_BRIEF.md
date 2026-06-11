# Meeting Brief

This document is a meeting-ready overview of the capstone project. It is designed to help explain the project clearly to the professor, with enough detail to show the logic behind each milestone and the current modeling status.

## Project Title

`AI-Based Student Dropout Early Warning System for Higher Education: Reproduction and Cross-Dataset Generalization on UCI and OULAD`

## 1. Opening Summary

This project studies whether an explainable machine learning workflow for student dropout prediction can remain useful when moved across different educational datasets.

We are not only trying to build a prediction model. We are also trying to understand:

- whether the workflow can be reproduced on a benchmark dataset
- whether it can be extended to a richer and more realistic dataset
- whether predictive patterns and explanations remain stable across datasets

So the project is a reproduction-and-extension study, not just a standard one-dataset classification project.

## 2. Why This Project Matters

Student dropout is important because it affects:

- student retention and success
- advising and intervention workload
- institutional planning and resource allocation

An early-warning system is only useful if:

- it identifies at-risk students early enough for intervention
- it uses signals that are understandable to educators

That is why this project focuses on both prediction and explainability.

## 3. Main Research Logic

The logic of the project is:

1. Reproduce a defensible benchmark workflow on a clean structured dataset.
2. Move to a richer and more realistic educational dataset with behavioral signals.
3. Compare whether predictive signals remain useful across datasets.
4. Examine whether the explanations behind model predictions remain stable or change across data environments.

This means the project has moved beyond a simple single-dataset classification task. It is now a reproduction-and-extension study.

## 4. Core Research Questions

The project is trying to answer these questions:

1. Can the explainable machine learning workflow reported in the benchmark literature be reproduced on the UCI dataset?
2. Can a comparable early-warning modeling pipeline be built on OULAD?
3. If models are developed across these two datasets, do the main predictive patterns remain stable?
4. Are the most important features and explanations portable across datasets, or are they highly dataset-specific?

## 5. Why We Use Two Datasets

### UCI Dataset

The `UCI Predict Students' Dropout and Academic Success` dataset is used as the:

- reproduction dataset
- benchmark dataset

Why:

- it is compact
- it is already structured as a single flat table
- it is easier to use for baseline reproduction and benchmarking

### OULAD Dataset

The `Open University Learning Analytics Dataset (OULAD)` is used as the:

- primary evaluation dataset

Why:

- it contains richer behavioral and temporal information
- it includes assessment records and online learning activity
- it is more realistic for an early-warning system
- it allows a stronger test of robustness and transferability

## 6. Important Terms

### VLE

`VLE` means `Virtual Learning Environment`.

In OULAD, this is the online learning platform behavior data. It records how students interact with course resources, quizzes, pages, forums, and other learning materials.

Why it matters:

- it provides early engagement signals
- students who withdraw often show weaker VLE activity
- it gives OULAD a richer behavioral layer that does not exist in the UCI benchmark

### Shared Feature Schema

The `shared_feature_schema.json` file is a concept-level mapping between UCI and OULAD variables.

Why we need it:

- the two datasets do not share the same raw columns
- we still want to compare them fairly
- so we align them by feature families instead of raw field names

Examples of shared feature families:

- demographics
- prior preparation
- program setup
- early academic progress
- early engagement

This is important because later cross-dataset generalization should compare comparable concepts, not pretend that every column has a one-to-one match.

### F1

`F1` balances:

- `precision`: when the model flags a student as high risk, how often that is correct
- `recall`: of the truly high-risk students, how many the model actually catches

Why it matters here:

- an early-warning system should not miss too many truly at-risk students
- but it also should not flag too many low-risk students incorrectly

### ROC AUC

`ROC AUC` measures how well the model separates higher-risk from lower-risk students overall.

Why it matters here:

- it captures the model's general ability to rank risk correctly
- it is less dependent on one fixed classification threshold than F1

## 7. What We Did By Milestone

## Milestone 1

### Goal

Choose the topic, define the business problem, and propose a feasible capstone direction.

### What We Did

1. Defined the problem as student dropout early warning in higher education.
2. Framed the business objective as identifying at-risk students early enough for meaningful intervention.
3. Proposed an initial machine learning workflow using standard classification methods.
4. Included explainability as part of the design through SHAP.
5. Identified the UCI dataset as the original planned starting point.

### Why It Matters

Milestone 1 established the problem, the educational value, and the initial technical direction.

## Milestone 2

### Goal

Review the literature, identify datasets, and sharpen the research contribution.

### What We Did

1. Reviewed student-dropout prediction and explainable-AI literature.
2. Identified that many studies show strong within-dataset prediction performance.
3. Noted that much less work has been done on:
   - cross-dataset generalization
   - explanation stability
   - portability of learned patterns across educational settings
4. Reframed the project from a simple prediction system into a reproduction-plus-extension study.
5. Confirmed the roles of the two datasets:
   - UCI for reproduction
   - OULAD for richer evaluation

### Why It Matters

Milestone 2 is where the project became more research-oriented and more graduate-level.

## Milestone 3

### Goal

Collect the data, perform initial EDA, and determine whether the project is ready to move into preprocessing and baseline modeling.

### What We Did

1. Collected both datasets locally.
2. Verified row counts, structure, targets, missingness, and duplicates.
3. Confirmed that UCI contains:
   - `4,424` records
   - `37` columns
   - no missing values
   - no duplicate rows
4. Confirmed that OULAD includes:
   - student-level tables
   - registration history
   - assessment records
   - more than `10.6` million VLE interaction rows
5. Performed initial EDA on UCI.
   - reviewed target distribution
   - compared outcome groups
   - examined selected correlations
6. Performed initial EDA on OULAD.
   - reviewed `final_result` distribution
   - compared studied credits, scores, and VLE engagement
   - confirmed that withdrawn students show much lower participation and engagement
7. Tested whether OULAD could realistically support later modeling.
   - verified the join structure
   - confirmed chunked aggregation was feasible for the large `studentVle` table
8. Identified the main preprocessing risks.
   - UCI: coded variables, later-semester leakage concerns
   - OULAD: multi-table joins, aggregation, and label leakage through late behavior and withdrawal timing

### Why It Matters

Milestone 3 proved that:

- UCI was immediately ready for baseline modeling
- OULAD was usable, but only after aggregation and leakage-aware preprocessing

This was the key bridge into Milestone 4.

## Milestone 4

### Goal

Turn both datasets into model-ready forms and run baseline models.

### What We Did Step By Step

#### Step 1: Defined the modeling tasks

We set up three modeling tasks:

- `UCI multiclass reproduction`
- `UCI binary early warning`
- `OULAD binary early warning`

Why:

- the multiclass task preserves the original benchmark setting
- the binary tasks are more suitable for early-warning attrition analysis

#### Step 2: Prepared UCI for modeling

We built:

- a multiclass benchmark dataset
- an early-warning binary dataset

For the early-warning binary version, we removed second-semester variables.

Why:

- later-semester features may inflate performance
- a true early-warning system should rely on information available early enough for intervention

We also treated coded categorical variables as categorical during modeling rather than leaving them as ordinary continuous numbers.

#### Step 3: Prepared OULAD for modeling

OULAD is not a single-table dataset, so we had to build a student-level table first.

We used:

- `code_module`
- `code_presentation`
- `id_student`

as the main join keys.

We combined:

- `studentInfo`
- `studentRegistration`
- `studentAssessment`
- `studentVle`
- supporting metadata tables

Why:

- OULAD stores different parts of the student story across multiple tables
- raw tables are not directly suitable for baseline modeling

#### Step 4: Built early-course features

We used a fixed `75`-day cutoff to define the early-warning window.

For assessment activity, we aggregated features such as:

- early submission count
- early score summaries
- weighted-score summaries

For VLE behavior, we aggregated features such as:

- total clicks
- event counts
- active days
- unique site visits
- activity-type click totals

Why:

- raw event-level data are too granular for direct modeling
- early-course aggregation creates student-level signals that are much more usable

#### Step 5: Controlled leakage

We explicitly prevented obvious leakage by:

- removing `date_unregistration` from OULAD predictors
- limiting OULAD behavior features to the early-warning window
- removing second-semester variables from the UCI early-warning version

Why:

- otherwise the model could rely on information that would not be available at a true intervention point

#### Step 6: Created a shared feature schema

We exported:

- `shared_feature_schema.json`
- `shared_feature_schema.csv`

Why:

- UCI and OULAD do not have matching raw schemas
- later cross-dataset comparisons should align them by feature families, not raw column names

#### Step 7: Ran baseline models

For each task, we ran:

- logistic regression
- decision tree
- random forest
- gradient boosting
- XGBoost

For model selection, we used:

- stratified `60/20/20` train-validation-test splitting
- `5-fold` cross-validation on the selected model for each task
- train-only preprocessing pipelines for imputation, clipping, encoding, and scaling

We exported:

- metrics
- confusion matrices
- top-feature charts
- feature-importance CSV files

### Milestone 4 Results

The current main baseline results are:

| Task | Best model | Main metrics |
| --- | --- | --- |
| UCI multiclass reproduction | XGBoost | accuracy `0.768`, macro F1 `0.704` |
| UCI binary early warning | Logistic regression | accuracy `0.858`, F1 `0.783`, ROC AUC `0.910` |
| OULAD binary early warning | XGBoost | accuracy `0.843`, F1 `0.729`, ROC AUC `0.885` |

### How To Interpret Those Results

- UCI is easier and cleaner, so it gives stronger baseline performance.
- OULAD is harder because it is richer, more behavioral, and more structurally complex.
- Even so, OULAD still produced a meaningful early-warning baseline, and its selected model improved once the article-aligned boosting family was added.
- That means the project now has a credible baseline foundation for later cross-dataset analysis.

## 8. What We Have Accomplished So Far

At this point, the project has already completed:

- topic definition
- literature review
- research-gap refinement
- dataset collection
- EDA
- preprocessing
- leakage-aware baseline modeling

So the project is already past the planning stage and is now in the baseline-results stage.

## 9. What Is Not Finished Yet

The project still needs to do:

- advanced modeling and optimization
- stronger validation
- explicit cross-dataset transfer experiments
- explanation-stability analysis

These later stages are where the project becomes a stronger generalization study rather than only a baseline classifier.

## 10. Good Closing Summary For The Meeting

Suggested closing statement:

So far, we have completed the problem framing, literature review, data collection, EDA, preprocessing, and baseline modeling stages. UCI is serving as the benchmark reproduction dataset, while OULAD is serving as the richer early-warning evaluation dataset. We now have model-ready data and defensible baselines on both datasets. The next stage is to move from baseline performance into stronger validation, cross-dataset generalization, and explanation-stability analysis.
