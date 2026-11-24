# IDRiD split summary

Seed 0, val fraction 0.15, 5 CV folds, output size 512px.

| Split | Images | DR ≥ 2 | DR < 2 | DR 0 | DR 1 | DR 2 | DR 3 | DR 4 |
|---|---|---|---|---|---|---|---|---|
| train | 296 | 168 (56.8%) | 128 | 113 | 15 | 84 | 50 | 34 |
| val | 53 | 30 (56.6%) | 23 | 20 | 3 | 15 | 9 | 6 |
| test | 103 | 64 (62.1%) | 39 | 34 | 5 | 32 | 19 | 13 |
| train+val (CV pool) | 349 | 198 (56.7%) | 151 | 133 | 18 | 99 | 59 | 40 |

Excluded from B-train (64): duplicate with conflicting grades: 6, duplicate, kept first copy: 2, duplicate of a test image: 1, also in Part A: 55.
Test images with conflicting duplicate labels (kept, flagged): B_test_IDRiD_064.
Localization set: 81 Part A images, 58 matched to a B grade.
