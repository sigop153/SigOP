# Built-in objectives

| folder | task | proposal | train | val | test |
|---|---|---|---|---|---|
| `s2d` | Symptom2Disease: English symptoms to one of 22 diseases | 200 | 50 | 300 | 212 |
| `lawbench` | LawBench 3-2: Chinese case facts to charges | 150 | 50 | 50 | 100 |

Rows are the ones the paper's runs scored on. Both come from the Meta-Harness
text-classification example, which vendors the MCE artifact's `symptom_diagnosis`
(`gretelai/symptom_to_diagnosis`) and `crime_prediction` (LawBench, Fei et al. 2024).
Meta-Harness ships train (memory, 200), val (search set, 50) and test. Mapping:

- proposal = Meta-Harness train. LawBench: the 150 rows its loader keeps after
  `random.Random(42).shuffle` with `num_train = 150`.
- train = Meta-Harness val, the 50-row search set.
- val = the paper's held-out monitor. S2D: 300 rows from `gretelai/symptom_to_diagnosis`
  outside every shipped split (seed 42). LawBench: the 50 rows the shuffle leaves out.
- test = Meta-Harness test.

`scripts/import_paper_splits.py` does the mapping and writes `manifest.json` with sizes and
SHA-256s. LawBench targets have the `罪名:` prefix stripped.

One S2D question is in both train and test as shipped; it is kept and counted in the
manifest under `overlapping_inputs`. val is disjoint from every other split in both tasks.
