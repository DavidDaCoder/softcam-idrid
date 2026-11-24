# IDRiD across seeds

Seeds per model: Mean SC [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], Top-k SC [0, 1, 2, 3, 4], Black-box [0, 1, 2, 3, 4], Multi SC [0, 1, 2, 3, 4], S3 SC [0, 1, 2, 3, 4], Multi SC s3-topk [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], S3 SC topk [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], Multi SC both-topk [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], Multi SC s3-topk 1% [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], Multi SC s3-topk 10% [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], Mean SC lr 0.0003 [0, 1, 2], Mean SC lr 0.003 [0, 1, 2], Mean SC lr 0.01 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Mean ± SD across seeds (test n = 103 per seed).

| Model | Accuracy | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Mean SC | 0.844 ± 0.017 | 0.911 ± 0.007 | 0.861 ± 0.019 | 0.815 ± 0.042 |
| Top-k SC | 0.849 ± 0.011 | 0.915 ± 0.010 | 0.853 ± 0.028 | 0.841 ± 0.049 |
| Black-box | 0.839 ± 0.009 | 0.909 ± 0.008 | 0.856 ± 0.013 | 0.810 ± 0.029 |
| Multi SC | 0.835 ± 0.025 | 0.889 ± 0.022 | 0.844 ± 0.011 | 0.821 ± 0.065 |
| S3 SC | 0.800 ± 0.037 | 0.865 ± 0.016 | 0.834 ± 0.032 | 0.744 ± 0.079 |
| Multi SC s3-topk | 0.851 ± 0.026 | 0.912 ± 0.020 | 0.867 ± 0.021 | 0.826 ± 0.054 |
| S3 SC topk | 0.855 ± 0.017 | 0.920 ± 0.009 | 0.856 ± 0.023 | 0.854 ± 0.054 |
| Multi SC both-topk | 0.841 ± 0.010 | 0.909 ± 0.007 | 0.853 ± 0.032 | 0.821 ± 0.058 |
| Multi SC s3-topk 1% | 0.845 ± 0.036 | 0.904 ± 0.022 | 0.861 ± 0.026 | 0.818 ± 0.069 |
| Multi SC s3-topk 10% | 0.846 ± 0.017 | 0.908 ± 0.011 | 0.845 ± 0.014 | 0.846 ± 0.055 |
| Mean SC lr 0.0003 | 0.790 ± 0.015 | 0.854 ± 0.012 | 0.839 ± 0.024 | 0.709 ± 0.015 |
| Mean SC lr 0.003 | 0.838 ± 0.015 | 0.907 ± 0.016 | 0.849 ± 0.033 | 0.821 ± 0.068 |
| Mean SC lr 0.01 | 0.839 ± 0.016 | 0.906 ± 0.009 | 0.844 ± 0.021 | 0.831 ± 0.046 |

Train / validation / test, mean ± SD across seeds (from eval_idrid_splits.py; no augmentation).
Use validation to choose hyperparameters; train vs test shows overfitting.

| Model | Seeds | Train acc | Val acc | Val AUC | Test acc | Test AUC |
|---|---|---|---|---|---|---|
| Mean SC | 10 | 0.960 ± 0.014 | 0.958 ± 0.008 | 0.969 ± 0.004 | 0.844 ± 0.017 | 0.911 ± 0.007 |
| Top-k SC | 5 | 0.957 ± 0.016 | 0.943 ± 0.000 | 0.964 ± 0.007 | 0.849 ± 0.011 | 0.915 ± 0.010 |
| Black-box | 5 | 0.958 ± 0.019 | 0.955 ± 0.010 | 0.968 ± 0.003 | 0.839 ± 0.009 | 0.909 ± 0.008 |
| Multi SC | 5 | 0.936 ± 0.030 | 0.955 ± 0.010 | 0.966 ± 0.003 | 0.835 ± 0.025 | 0.889 ± 0.022 |
| S3 SC | 5 | 0.907 ± 0.010 | 0.940 ± 0.008 | 0.957 ± 0.004 | 0.800 ± 0.037 | 0.865 ± 0.016 |
| Multi SC s3-topk | 10 | 0.950 ± 0.022 | 0.960 ± 0.006 | 0.970 ± 0.005 | 0.851 ± 0.026 | 0.912 ± 0.020 |
| S3 SC topk | 10 | 0.956 ± 0.016 | 0.962 ± 0.000 | 0.969 ± 0.006 | 0.855 ± 0.017 | 0.920 ± 0.009 |
| Multi SC both-topk | 10 | 0.938 ± 0.023 | 0.949 ± 0.013 | 0.965 ± 0.007 | 0.841 ± 0.010 | 0.909 ± 0.007 |
| Multi SC s3-topk 1% | 10 | 0.955 ± 0.025 | 0.942 ± 0.011 | 0.966 ± 0.008 | 0.845 ± 0.036 | 0.904 ± 0.022 |
| Multi SC s3-topk 10% | 10 | 0.941 ± 0.014 | 0.960 ± 0.006 | 0.970 ± 0.004 | 0.846 ± 0.017 | 0.908 ± 0.011 |
| Mean SC lr 0.0003 | 3 | 0.900 ± 0.019 | 0.937 ± 0.011 | 0.962 ± 0.001 | 0.790 ± 0.015 | 0.854 ± 0.012 |
| Mean SC lr 0.003 | 3 | 0.952 ± 0.019 | 0.943 ± 0.000 | 0.962 ± 0.002 | 0.838 ± 0.015 | 0.907 ± 0.016 |
| Mean SC lr 0.01 | 10 | 0.971 ± 0.020 | 0.953 ± 0.010 | 0.972 ± 0.008 | 0.839 ± 0.016 | 0.906 ± 0.009 |

Localization on Part A (81 images), evidence ÷ chance, mean ± SD across seeds:

| Region | Mean SC | Top-k SC | Multi SC | S3 SC | Multi SC s3-topk | S3 SC topk | Multi SC both-topk | Multi SC s3-topk 1% | Multi SC s3-topk 10% | Mean SC lr 0.0003 | Mean SC lr 0.003 | Mean SC lr 0.01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| All lesions | 3.73 ± 0.10 | 4.31 ± 0.56 | 3.69 ± 0.24 | 3.50 ± 0.29 | 8.00 ± 1.07 | 8.57 ± 0.49 | 6.23 ± 0.63 | 10.85 ± 1.67 | 6.35 ± 0.76 | 3.00 ± 0.08 | 4.03 ± 0.27 | 4.67 ± 0.52 |
| Microaneurysms | 2.95 ± 0.17 | 2.74 ± 0.39 | 2.86 ± 0.27 | 2.57 ± 0.30 | 4.26 ± 0.59 | 4.79 ± 0.73 | 3.51 ± 0.63 | 4.45 ± 1.79 | 3.74 ± 0.47 | 2.28 ± 0.09 | 2.94 ± 0.21 | 2.94 ± 0.24 |
| Haemorrhages | 2.75 ± 0.14 | 2.86 ± 0.62 | 2.81 ± 0.14 | 2.67 ± 0.21 | 3.51 ± 0.72 | 4.02 ± 0.64 | 3.24 ± 0.52 | 3.93 ± 1.28 | 3.33 ± 0.49 | 2.51 ± 0.19 | 2.74 ± 0.20 | 2.92 ± 0.33 |
| Hard exudates | 5.50 ± 0.30 | 6.71 ± 0.72 | 5.27 ± 0.55 | 4.71 ± 0.50 | 14.30 ± 2.78 | 15.01 ± 1.25 | 10.50 ± 1.38 | 22.83 ± 5.28 | 10.44 ± 1.72 | 3.98 ± 0.17 | 6.24 ± 0.59 | 7.75 ± 1.11 |
| Soft exudates | 1.97 ± 0.19 | 2.45 ± 0.59 | 2.05 ± 0.26 | 2.83 ± 0.16 | 5.70 ± 1.79 | 6.38 ± 1.49 | 4.87 ± 1.30 | 7.74 ± 4.38 | 4.33 ± 1.00 | 2.13 ± 0.30 | 1.96 ± 0.09 | 2.66 ± 0.55 |
| Optic disc | 0.45 ± 0.21 | 0.28 ± 0.23 | 0.47 ± 0.14 | 0.66 ± 0.12 | 0.31 ± 0.13 | 0.23 ± 0.12 | 0.38 ± 0.16 | 0.15 ± 0.26 | 0.45 ± 0.11 | 0.87 ± 0.27 | 0.34 ± 0.19 | 0.13 ± 0.17 |
| Pointing game hit rate | 90% ± 3 | 89% ± 3 | 80% ± 4 | 92% ± 5 | 93% ± 2 | 94% ± 1 | 84% ± 5 | 92% ± 5 | 94% ± 2 | 86% ± 1 | 91% ± 1 | 89% ± 4 |
| Max faithfulness gap | 4.8e-06 | 2.9e-06 | 9.5e-07 | 4.8e-07 | 3.8e-06 | 3.8e-06 | 1.9e-06 | 3.8e-06 | 1.9e-06 | 2.1e-06 | 6.7e-06 | 1.1e-05 |

Pointing-game chance depends on cell size (~37% for 16×16 maps, ~22% for 32×32), so compare pointing only between models with the same map size.

Per stage for Multi SC (each stage's weighted map at its own resolution), mean ± SD across seeds:

| Stage | Lesions ÷ chance | Microaneurysms ÷ chance |
|---|---|---|
| 3 | 5.25 ± 1.19 | 4.01 ± 1.16 |
| 4 | 3.17 ± 0.25 | 2.54 ± 0.23 |

Per stage for S3 SC (each stage's weighted map at its own resolution), mean ± SD across seeds:

| Stage | Lesions ÷ chance | Microaneurysms ÷ chance |
|---|---|---|
| 3 | 3.50 ± 0.29 | 2.57 ± 0.30 |

Per stage for Multi SC s3-topk (each stage's weighted map at its own resolution), mean ± SD across seeds:

| Stage | Lesions ÷ chance | Microaneurysms ÷ chance |
|---|---|---|
| 3 | 8.67 ± 0.70 | 4.64 ± 0.87 |
| 4 | 2.46 ± 0.26 | 2.07 ± 0.17 |

Per stage for S3 SC topk (each stage's weighted map at its own resolution), mean ± SD across seeds:

| Stage | Lesions ÷ chance | Microaneurysms ÷ chance |
|---|---|---|
| 3 | 8.57 ± 0.49 | 4.79 ± 0.73 |

Per stage for Multi SC both-topk (each stage's weighted map at its own resolution), mean ± SD across seeds:

| Stage | Lesions ÷ chance | Microaneurysms ÷ chance |
|---|---|---|
| 3 | 8.60 ± 1.38 | 3.94 ± 0.85 |
| 4 | 5.18 ± 0.37 | 3.41 ± 0.70 |

Per stage for Multi SC s3-topk 1% (each stage's weighted map at its own resolution), mean ± SD across seeds:

| Stage | Lesions ÷ chance | Microaneurysms ÷ chance |
|---|---|---|
| 3 | 11.61 ± 1.16 | 4.72 ± 2.18 |
| 4 | 2.25 ± 0.33 | 1.85 ± 0.22 |

Per stage for Multi SC s3-topk 10% (each stage's weighted map at its own resolution), mean ± SD across seeds:

| Stage | Lesions ÷ chance | Microaneurysms ÷ chance |
|---|---|---|
| 3 | 7.30 ± 0.52 | 4.24 ± 0.58 |
| 4 | 2.60 ± 0.24 | 2.20 ± 0.19 |

Paired by seed (Top-k SC − Mean SC, seeds [0, 1, 2, 3, 4]):

| Measure | Mean difference | Seeds where Top-k SC is higher |
|---|---|---|
| Acc | +0.010 | 3/5 |
| AUC | +0.007 | 4/5 |
| Sens | -0.003 | 1/5 |
| Spec | +0.031 | 4/5 |
| All lesions | +0.627 | 5/5 |
| Microaneurysms | -0.117 | 2/5 |
| Haemorrhages | +0.100 | 2/5 |
| Hard exudates | +1.354 | 5/5 |
| Soft exudates | +0.501 | 4/5 |
| Optic disc | -0.203 | 1/5 |
| Pointing game | +0.017 | 4/5 |

For the optic disc, lower is better (less shortcut evidence).

Paired by seed (Multi SC − Mean SC, seeds [0, 1, 2, 3, 4]):

| Measure | Mean difference | Seeds where Multi SC is higher |
|---|---|---|
| Acc | -0.004 | 2/5 |
| AUC | -0.019 | 1/5 |
| Sens | -0.013 | 1/5 |
| Spec | +0.010 | 2/5 |
| All lesions | +0.003 | 2/5 |
| Microaneurysms | +0.009 | 3/5 |
| Haemorrhages | +0.053 | 3/5 |
| Hard exudates | -0.080 | 1/5 |
| Soft exudates | +0.104 | 2/5 |
| Optic disc | -0.012 | 2/5 |
| Pointing game | -0.081 | 0/5 |

For the optic disc, lower is better (less shortcut evidence).

Paired by seed (S3 SC − Mean SC, seeds [0, 1, 2, 3, 4]):

| Measure | Mean difference | Seeds where S3 SC is higher |
|---|---|---|
| Acc | -0.039 | 0/5 |
| AUC | -0.043 | 0/5 |
| Sens | -0.022 | 2/5 |
| Spec | -0.067 | 1/5 |
| All lesions | -0.191 | 2/5 |
| Microaneurysms | -0.283 | 2/5 |
| Haemorrhages | -0.088 | 2/5 |
| Hard exudates | -0.641 | 0/5 |
| Soft exudates | +0.878 | 5/5 |
| Optic disc | +0.178 | 5/5 |
| Pointing game | +0.042 | 3/5 |

For the optic disc, lower is better (less shortcut evidence).

Paired by seed (Multi SC s3-topk − Mean SC, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

| Measure | Mean difference | Seeds where Multi SC s3-topk is higher |
|---|---|---|
| Acc | +0.008 | 5/10 |
| AUC | +0.001 | 8/10 |
| Sens | +0.006 | 5/10 |
| Spec | +0.010 | 5/10 |
| All lesions | +4.264 | 10/10 |
| Microaneurysms | +1.310 | 9/10 |
| Haemorrhages | +0.755 | 8/10 |
| Hard exudates | +8.799 | 10/10 |
| Soft exudates | +3.725 | 10/10 |
| Optic disc | -0.139 | 2/10 |
| Pointing game | +0.035 | 7/10 |

For the optic disc, lower is better (less shortcut evidence).

Paired by seed (S3 SC topk − Mean SC, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

| Measure | Mean difference | Seeds where S3 SC topk is higher |
|---|---|---|
| Acc | +0.012 | 5/10 |
| AUC | +0.008 | 7/10 |
| Sens | -0.005 | 3/10 |
| Spec | +0.038 | 6/10 |
| All lesions | +4.839 | 10/10 |
| Microaneurysms | +1.844 | 10/10 |
| Haemorrhages | +1.264 | 10/10 |
| Hard exudates | +9.513 | 10/10 |
| Soft exudates | +4.410 | 10/10 |
| Optic disc | -0.218 | 1/10 |
| Pointing game | +0.042 | 8/10 |

For the optic disc, lower is better (less shortcut evidence).

Paired by seed (Multi SC both-topk − Mean SC, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

| Measure | Mean difference | Seeds where Multi SC both-topk is higher |
|---|---|---|
| Acc | -0.003 | 2/10 |
| AUC | -0.002 | 3/10 |
| Sens | -0.008 | 4/10 |
| Spec | +0.005 | 4/10 |
| All lesions | +2.498 | 10/10 |
| Microaneurysms | +0.556 | 8/10 |
| Haemorrhages | +0.489 | 8/10 |
| Hard exudates | +4.998 | 10/10 |
| Soft exudates | +2.896 | 10/10 |
| Optic disc | -0.070 | 4/10 |
| Pointing game | -0.053 | 2/10 |

For the optic disc, lower is better (less shortcut evidence).

Paired by seed (Multi SC s3-topk 1% − Mean SC, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

| Measure | Mean difference | Seeds where Multi SC s3-topk 1% is higher |
|---|---|---|
| Acc | +0.001 | 6/10 |
| AUC | -0.008 | 5/10 |
| Sens | +0.000 | 6/10 |
| Spec | +0.003 | 6/10 |
| All lesions | +7.122 | 10/10 |
| Microaneurysms | +1.504 | 7/10 |
| Haemorrhages | +1.178 | 7/10 |
| Hard exudates | +17.333 | 10/10 |
| Soft exudates | +5.771 | 9/10 |
| Optic disc | -0.294 | 1/10 |
| Pointing game | +0.026 | 6/10 |

For the optic disc, lower is better (less shortcut evidence).

Paired by seed (Multi SC s3-topk 10% − Mean SC, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

| Measure | Mean difference | Seeds where Multi SC s3-topk 10% is higher |
|---|---|---|
| Acc | +0.002 | 4/10 |
| AUC | -0.003 | 6/10 |
| Sens | -0.016 | 2/10 |
| Spec | +0.031 | 6/10 |
| All lesions | +2.614 | 10/10 |
| Microaneurysms | +0.793 | 9/10 |
| Haemorrhages | +0.581 | 10/10 |
| Hard exudates | +4.937 | 10/10 |
| Soft exudates | +2.357 | 10/10 |
| Optic disc | -0.002 | 6/10 |
| Pointing game | +0.049 | 9/10 |

For the optic disc, lower is better (less shortcut evidence).

Paired by seed (Mean SC lr 0.0003 − Mean SC, seeds [0, 1, 2]):

| Measure | Mean difference | Seeds where Mean SC lr 0.0003 is higher |
|---|---|---|
| Acc | -0.061 | 0/3 |
| AUC | -0.053 | 0/3 |
| Sens | -0.021 | 0/3 |
| Spec | -0.128 | 0/3 |
| All lesions | -0.642 | 0/3 |
| Microaneurysms | -0.544 | 0/3 |
| Haemorrhages | -0.184 | 0/3 |
| Hard exudates | -1.288 | 0/3 |
| Soft exudates | +0.182 | 2/3 |
| Optic disc | +0.410 | 3/3 |
| Pointing game | -0.004 | 1/3 |

For the optic disc, lower is better (less shortcut evidence).

Paired by seed (Mean SC lr 0.003 − Mean SC, seeds [0, 1, 2]):

| Measure | Mean difference | Seeds where Mean SC lr 0.003 is higher |
|---|---|---|
| Acc | -0.013 | 1/3 |
| AUC | -0.001 | 2/3 |
| Sens | -0.010 | 1/3 |
| Spec | -0.017 | 2/3 |
| All lesions | +0.382 | 3/3 |
| Microaneurysms | +0.114 | 3/3 |
| Haemorrhages | +0.052 | 2/3 |
| Hard exudates | +0.969 | 3/3 |
| Soft exudates | +0.006 | 2/3 |
| Optic disc | -0.121 | 2/3 |
| Pointing game | +0.045 | 3/3 |

For the optic disc, lower is better (less shortcut evidence).

Paired by seed (Mean SC lr 0.01 − Mean SC, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

| Measure | Mean difference | Seeds where Mean SC lr 0.01 is higher |
|---|---|---|
| Acc | -0.005 | 2/10 |
| AUC | -0.005 | 4/10 |
| Sens | -0.017 | 1/10 |
| Spec | +0.015 | 8/10 |
| All lesions | +0.935 | 9/10 |
| Microaneurysms | -0.015 | 3/10 |
| Haemorrhages | +0.164 | 7/10 |
| Hard exudates | +2.256 | 9/10 |
| Soft exudates | +0.687 | 8/10 |
| Optic disc | -0.316 | 0/10 |
| Pointing game | -0.007 | 3/10 |

For the optic disc, lower is better (less shortcut evidence).
