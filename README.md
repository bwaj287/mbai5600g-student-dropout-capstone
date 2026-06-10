# MBAI 5600G Capstone Project

This repository contains the working documents for an MBAI 5600G Applied Integrative Analytics Capstone Project focused on student dropout prediction in higher education.

The current project direction is:

`AI-Based Student Dropout Early Warning System for Higher Education: Reproduction and Cross-Dataset Generalization on UCI and OULAD`

## Project Overview

The capstone studies whether explainable machine learning workflows for student risk prediction remain effective when moved across different educational datasets.

The project has two linked goals:

- Reproduce the explainable machine learning workflow described by Islam et al. (2025) on the UCI Predict Students' Dropout and Academic Success dataset.
- Extend that work by evaluating robustness, transferability, and explanation stability using the Open University Learning Analytics Dataset (OULAD).

## Repository Structure

```text
capstone/
  material/    code samples and a local reference-material index
  other/       private administrative documents kept out of the public repo
  work/
    milestone1/
    milestone2/
    milestone3/
      data/
      code/
    milestone4/
```

## Included Content

This public repository currently includes:

- milestone requirement PDFs used for course planning
- milestone draft documents and revision guides
- milestone structure documents for follow-up work
- selected sample notebooks and sample datasets used to study course coding style
- project-level documentation for organizing the capstone

## Excluded Content

Some local files were intentionally not published in this public repository:

- personal and administrative documents such as payment files, forms, and transcript-related records
- local course reading PDFs and ZIP files stored for private study and reference

These exclusions were made for privacy and copyright reasons.

## Current Milestone Artifacts

The working files are now organized by milestone under `capstone/work/`:

- `capstone/work/milestone1/`
  - milestone requirement PDF and Milestone 1 report files
- `capstone/work/milestone2/`
  - milestone requirement PDF, Milestone 2 report files, and revision guide
- `capstone/work/milestone3/`
  - milestone requirement PDF, Milestone 3 report files, structure document, and report code
- `capstone/work/milestone4/`
  - preprocessing and baseline-modeling workspace, including scripts, model-ready datasets, figures, and results

## Coding Style Reference

The repository also includes a small `capstone/material/code sample/` folder containing example notebooks and sample datasets used to study the professor's preferred instructional coding style. These samples suggest a style that is:

- step-by-step and notebook-friendly
- explicit in imports, transformations, and print outputs
- light on abstraction and heavy on clarity
- organized around explanation, preprocessing, visualization, and interpretation
- suitable for academic demonstrations rather than production engineering

## Data Plan

The intended analytical workflow uses:

- `UCI Predict Students' Dropout and Academic Success` as the reproduction and benchmark dataset
- `OULAD` as the primary evaluation dataset for richer behavioural and temporal analysis

The modeling direction emphasizes:

- reproduction of the base paper
- cross-dataset generalization
- explanation stability
- readiness for baseline and advanced modeling in later milestones

## Notes

This repository remains document-focused overall, but the local working tree also includes raw datasets and generated analysis artifacts used for Milestones 3 and 4.

## License

No license has been added yet. If this repository is later expanded to include code or reusable assets, a project license can be added at that stage.
