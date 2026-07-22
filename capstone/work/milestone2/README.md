# Milestone 2

This folder contains the second milestone of the capstone project.

## Milestone 2 Goal

Milestone 2 focused on:

- literature review
- data source identification
- refining the research contribution of the project

This was the milestone where the capstone moved from a standard prediction project into a stronger reproduction-and-extension design.

## What We Did

### Step 1: Reviewed the literature

- Read and organized the relevant dropout-prediction and explainable-AI literature.
- Identified major themes around:
  - institutional prediction models
  - learning analytics and behavioral traces
  - explainable AI in education
  - the lack of strong evidence for cross-dataset robustness

### Step 2: Reframed the project contribution

- Shifted the project away from being only a single-dataset classifier.
- Reframed it as a:
  - `reproduction` study on UCI
  - `cross-dataset extension` study using OULAD

### Step 3: Defined the two-dataset design

- Confirmed `UCI Predict Students' Dropout and Academic Success` as the benchmark and reproduction dataset.
- Confirmed `OULAD` as the richer evaluation dataset because it includes temporal and behavioral learning information.

### Step 4: Clarified the research gap

- Identified that many studies report good within-dataset performance.
- Identified that there is much less evidence about:
  - generalization across datasets
  - robustness of learned patterns
  - stability of explanations across educational environments

### Step 5: Updated the analytical plan

- Defined a workflow that later milestones would follow:
  - reproduce the base paper on UCI
  - build a comparable pipeline on OULAD
  - later evaluate cross-dataset portability
  - compare explanation stability

## What This Milestone Produced

- The Milestone 2 report
- A revision guide for improving the Milestone 2 draft
- The conceptual shift that shaped the rest of the project

## Main Files

- `Activity 2.pdf`
- `Milestone2.pdf`

## Why Milestone 2 Matters

Milestone 2 is where the project became more graduate-level.

Instead of only asking:

- "Can we predict student dropout?"

the project began asking:

- "Can an explainable workflow reproduced on one educational dataset remain useful on another?"
- "Are the important risk signals stable across datasets?"

That change is what led directly into the data collection and EDA work of Milestone 3.
