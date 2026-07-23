# Milestone 3 Code

`milestone3_eda_pipeline.py` loads UCI and OULAD, calculates the EDA summaries,
and saves the figures used in the Milestone 3 report.

Run it from the repository root:

```bash
python3 capstone/work/milestone3/code/milestone3_eda_pipeline.py
```

Outputs are written to:

- `capstone/work/milestone3/data/analysis/`
- `capstone/work/milestone3/data/figures/`

In OULAD, VLE means Virtual Learning Environment. The large `studentVle` table
contains dated student activity and click counts. The script processes it in
chunks rather than loading the whole file into memory.

The local Word report builder is not part of the public analysis code.
