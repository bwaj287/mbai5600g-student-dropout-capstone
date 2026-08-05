# Milestone 7: Final Presentation

Milestone 7 presents the final cross-institution project. The main model uses a
fixed 16-feature semantic contract rather than institution-specific raw column
names. Four core fields are required, while 12 background, assessment, and
activity fields are optional. Separate UCI and OULAD adapters produce the same
input order for one frozen pooled model.

The presentation gives three minutes to EDA because the charts explain why the
two datasets need different local mappings. It then covers the adapter workflow,
local-only and pooled tests across enrolment and day-35/day-60/day-75 snapshots,
clustered confidence intervals, paired permutation tests, dynamic SHAP,
limitations, and the proposed third-school pilot. The presentation reports the
small but statistically significant later OULAD portability cost rather than
claiming equal performance in every context.

## Files

- `Milestone7_Presentation_Final.pptx`: final presentation
- `Milestone7_Even_Teleprompt.txt`: slide-by-slide spoken script
- `Activity 7.pdf`: presentation requirements

The embedded PowerPoint notes and the text teleprompt use the same talk track.
The teleprompt assigns nine spoken slides to each presenter. Slides 19 and 20
are appendix material. The transfer claim is limited to a fixed model behind a
documented local adapter; it does not claim that an arbitrary CSV can be scored
safely without local validation, ranking review, and calibration.
