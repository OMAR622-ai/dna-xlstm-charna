| Task | Metric | HyenaDNA | Mamba-PS | Mamba-PH | xLSTM-PS | xLSTM-PH | GBST-BERT | Transformer++ | seeds |
|---|---|---|---|---|---|---|---|---|---|
| H3 | MCC | 0.779±0.037 | 0.799±0.029 | 0.815±0.048 | 0.796±0.014 | **0.824±0.010** | 0.817±0.022 | 0.800±0.025 | 3 |
| H3K14ac | MCC | 0.612±0.065 | 0.541±0.212 | 0.631±0.026 | 0.570±0.008 | 0.598±0.017 | **0.674±0.004** | 0.557±0.028 | 3 |
| H3K36me3 | MCC | 0.613±0.041 | 0.609±0.109 | 0.601±0.129 | 0.588±0.019 | 0.625±0.010 | **0.650±0.023** | 0.593±0.022 | 3 |
| H3K4me1 | MCC | 0.512±0.024 | 0.488±0.102 | 0.523±0.039 | 0.490±0.012 | 0.526±0.001 | **0.571±0.024** | 0.486±0.010 | 3 |
| H3K4me2 | MCC | 0.455±0.095 | 0.388±0.101 | 0.487±0.170 | 0.489±0.024 | 0.504±0.012 | **0.554±0.020** | 0.380±0.033 | 3 |
| H3K4me3 | MCC | **0.549±0.056** | 0.440±0.202 | 0.544±0.045 | 0.520±0.019 | 0.537±0.012 | 0.521±0.170 | 0.388±0.194 | 3 |
| H3K79me3 | MCC | 0.672±0.048 | 0.676±0.026 | **0.697±0.077** | 0.662±0.011 | **0.697±0.007** | 0.668±0.123 | 0.667±0.037 | 3 |
| H3K9ac | MCC | 0.581±0.061 | 0.604±0.048 | 0.622±0.030 | 0.622±0.013 | 0.627±0.008 | **0.648±0.024** | 0.600±0.046 | 3 |
| H4 | MCC | 0.763±0.044 | 0.789±0.020 | 0.811±0.022 | 0.793±0.011 | **0.813±0.008** | 0.790±0.060 | 0.804±0.019 | 3 |
| H4ac | MCC | 0.564±0.038 | 0.525±0.240 | 0.621±0.054 | 0.558±0.018 | 0.583±0.014 | **0.633±0.020** | 0.533±0.015 | 3 |
| enhancers | MCC | 0.517±0.117 | 0.491±0.066 | **0.546±0.073** | 0.375±0.030 | 0.545±0.024 | 0.353±0.058 | 0.383±0.037 | 3 |
| enhancers_types | MCC | 0.386±0.185 | 0.416±0.095 | 0.439±0.054 | 0.444±0.046 | **0.466±0.011** | 0.428±0.167 | 0.388±0.011 | 3 |
| promoter_all | F1 | 0.960±0.005 | 0.967±0.004 | **0.970±0.004** | 0.962±0.002 | 0.967±0.001 | **0.970±0.001** | 0.968±0.002 | 3 |
| promoter_no_tata | F1 | 0.959±0.011 | 0.968±0.006 | 0.968±0.010 | 0.963±0.002 | 0.970±0.001 | **0.971±0.004** | 0.970±0.001 | 3 |
| promoter_tata | F1 | 0.944±0.040 | **0.957±0.015** | 0.953±0.016 | 0.948±0.006 | 0.952±0.005 | 0.941±0.010 | 0.952±0.004 | 3 |
| splice_sites_all | Accuracy | 0.956±0.011 | 0.927±0.021 | 0.940±0.027 | 0.965±0.006 | 0.974±0.004 | **0.980±0.001** | 0.974±0.006 | 3 |
| splice_sites_acceptors | F1 | 0.958±0.010 | 0.936±0.077 | 0.937±0.033 | 0.970±0.005 | 0.953±0.008 | **0.976±0.003** | 0.961±0.022 | 3 |
| splice_sites_donors | F1 | 0.949±0.024 | 0.874±0.289 | 0.948±0.025 | 0.962±0.004 | 0.951±0.005 | **0.981±0.006** | 0.918±0.087 | 3 |

Mean ± (max − min) over seeds of the test metric. Paper values: Bio-xLSTM Table 1 (10 seeds, per-task hyperparameter search). Ours: final pretrained models (10,000 steps), Bio-xLSTM's DNA-xLSTM-PH settings per task (Table A5), PH; for splice acceptors/donors also LR 1e-3 and 5e-4, chosen by validation score. Bold = best mean per task.
Tasks won (ties counted for each): HyenaDNA 1, Mamba-PS 1, Mamba-PH 3, xLSTM-PS 0, xLSTM-PH 4, GBST-BERT 11, Transformer++ 0

Setting used per task and model (best mean validation score when several were run): H3 GBST-BERT: LR-8e-4_BATCH_SIZE-128; H3 Transformer++: LR-8e-4_BATCH_SIZE-128; H3K14ac GBST-BERT: LR-6e-4_BATCH_SIZE-128; H3K14ac Transformer++: LR-6e-4_BATCH_SIZE-128; H3K36me3 GBST-BERT: LR-6e-4_BATCH_SIZE-64; H3K36me3 Transformer++: LR-6e-4_BATCH_SIZE-64; H3K4me1 GBST-BERT: LR-8e-4_BATCH_SIZE-128; H3K4me1 Transformer++: LR-8e-4_BATCH_SIZE-128; H3K4me2 GBST-BERT: LR-6e-4_BATCH_SIZE-64; H3K4me2 Transformer++: LR-6e-4_BATCH_SIZE-64; H3K4me3 GBST-BERT: LR-8e-4_BATCH_SIZE-128; H3K4me3 Transformer++: LR-8e-4_BATCH_SIZE-128; H3K79me3 GBST-BERT: LR-1e-3_BATCH_SIZE-128; H3K79me3 Transformer++: LR-1e-3_BATCH_SIZE-128; H3K9ac GBST-BERT: LR-4e-4_BATCH_SIZE-64; H3K9ac Transformer++: LR-4e-4_BATCH_SIZE-64; H4 GBST-BERT: LR-8e-4_BATCH_SIZE-64; H4 Transformer++: LR-8e-4_BATCH_SIZE-64; H4ac GBST-BERT: LR-4e-4_BATCH_SIZE-64; H4ac Transformer++: LR-4e-4_BATCH_SIZE-64; enhancers GBST-BERT: LR-2e-3_BATCH_SIZE-512; enhancers Transformer++: LR-2e-3_BATCH_SIZE-512; enhancers_types GBST-BERT: LR-2e-3_BATCH_SIZE-512; enhancers_types Transformer++: LR-2e-3_BATCH_SIZE-512; promoter_all GBST-BERT: LR-4e-4_BATCH_SIZE-64; promoter_all Transformer++: LR-4e-4_BATCH_SIZE-64; promoter_no_tata GBST-BERT: LR-1e-3_BATCH_SIZE-128; promoter_no_tata Transformer++: LR-1e-3_BATCH_SIZE-128; promoter_tata GBST-BERT: LR-3e-3_BATCH_SIZE-128; promoter_tata Transformer++: LR-3e-3_BATCH_SIZE-128; splice_sites_all GBST-BERT: LR-8e-4_BATCH_SIZE-64; splice_sites_all Transformer++: LR-8e-4_BATCH_SIZE-64; splice_sites_acceptors GBST-BERT: LR-1e-3_BATCH_SIZE-128; splice_sites_acceptors Transformer++: LR-2e-3_BATCH_SIZE-128; splice_sites_donors GBST-BERT: LR-1e-3_BATCH_SIZE-128; splice_sites_donors Transformer++: LR-1e-3_BATCH_SIZE-128
