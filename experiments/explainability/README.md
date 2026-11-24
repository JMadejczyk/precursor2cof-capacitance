# Explainability experiment

## Start here

**Questions.** Which models predict capacitance (`capacity_max`, 114 COF substrates) better than chance? What do they use to do it? Is there chemistry in there, or only something trivial? The same questions for the electrocatalysis model (J), which can rank materials within a paper.

**Main findings**

1. **Monomer size is the main signal.** A straight line on heavy-atom count alone gives R² 0.10 and Spearman 0.42 (permutation p = 0.01). Larger monomers have lower capacitance. The effect holds inside both classes: Spearman −0.31 within the 29 CTFs and −0.40 within the 85 other COFs.
2. **The best models are only modestly better than size.**
   - TabPFN or RF on the RF-best set (EState + functional groups), RF on the BRR-best set, and BRR or RF on ECFP reach R² 0.17–0.19 and Spearman 0.48–0.52, which is 7–9 RMSE better than size in every one of the 10 CV repeats. TabPFN ties RF (R² 0.188 vs 0.186) and is not better on the other tables.
   - 46 of the 61 model × feature-table combinations beat a shuffled target (BH q < 0.05), but only 9 beat the size baseline in every repeat.
   - XGBoost, KNN, MACCS and the custom patterns don't beat size. Lasso on the combined sets is in between (better than size in 7–8 of 10 repeats, by 1.5–3.5 RMSE).
   - No model ranks the CTFs well: Spearman within CTFs is 0.01–0.39, compared with about 0.5 within the non-CTFs.
3. **The models mostly predict from size.**
   - The top SHAP features are size proxies: benzene-ring count, aromatic CH (EState `aaCH`, ECFP 849/726), substituted aromatic C (ECFP 356), molecular weight and diameter. Each has Spearman 0.8–0.98 with heavy-atom count, and all lower the prediction.
   - The models' held-out predictions have Spearman −0.63 to −0.76 with heavy-atom count.
4. **There is a small signal beyond size.**
   - The predictions keep a partial Spearman of +0.30 to +0.37 with capacitance once heavy-atom count is controlled for.
   - Trained directly on the size-corrected capacitance, RF gets Spearman ≈ 0.3 but R² only 0.02–0.04. This is p ≈ 0.02 per configuration and q ≈ 0.06 after correcting for the 54 configurations, so it is **suggestive, not established**.
   - The features behind it agree across ECFP, EState and the combined sets:
     - **Higher capacitance:** aromatic S (thiophene; EState `aaS`, ECFP 675) and aromatic ring N (pyridine/triazine type; EState `aaN`, ECFP 378).
     - **Lower capacitance:** OH (EState `sOH`, ECFP 807), NH₂ (`sNH2`) and the aryl imine N=C (ECFP 116).
5. **Checked in the data without any model ([feature_checks.py](feature_checks.py)), only thiophene stands out.** Substrates with aromatic S sit +78 above the size trend in the median (14 substrates, 6 of them CTFs; Mann–Whitney p = 0.02). OH (−55, p = 0.14), NH₂, aromatic N and imines are not individually significant once size is accounted for. All of these p-values are descriptive: the groups were picked after looking at SHAP.
6. **Electrocatalysis (J).** The one model that ranks well within papers (RF on whole-monomer ECFP, clean rows, z-scored J; within-paper accuracy 0.85, reproduced here) relies on these features:
   - **Better catalyst (more negative J):** aryl–aryl carbons (ECFP 136), substituted aromatic C (356), aryl-substituted triazine C (361), amide C(=O)N (486) and aromatic C–OH (946).
   - **Worse catalyst:** aromatic CH (726) and aryl C=O / aldehyde (963, 593).
   - Removing the largest paper (Wang2024) doesn't change the ranking of features.
   - The same model on EState + functional groups is at chance (0.49), so only the ECFP model is worth reading.

**Bottom line.** "Small monomer" explains about half of what the models can predict. The rest is a weak, consistent hint that thiophene S and aromatic ring N go with higher capacitance and OH/NH₂ with lower, but 114 substrates aren't enough to confirm it. Thiophene is the one effect also visible in the raw data.

## Files

| file | what it does |
|---|---|
| [common.py](common.py) | Data, feature tables, readable feature names (EState atom types, RDKit `fr_*` groups), models, CV splits, metrics |
| [cv.py](cv.py) | Every target × table × model: 10 × repeated stratified 5-fold, plus a shuffled-target null (100 permutations; 30 for TabPFN). Writes `results/cv/<target>/<table>/<model>/` |
| [summarize.py](summarize.py) | `results/summary.csv`: metrics, permutation p, BH q, comparison with the size baseline, Spearman within CTF / non-CTF |
| [explain.py](explain.py) | Out-of-fold SHAP over all 50 fold fits for chosen configurations. Writes `results/explain/<target>__<table>__<model>/` (`importance.csv`, `shap_oof.csv`, `family_share.csv`) |
| [feature_checks.py](feature_checks.py) | Model-free check of the substructures SHAP points at: `results/feature_checks.csv` |
| [electro_shap.py](electro_shap.py) | Held-out (leave-one-paper-out) SHAP for the J model. Needs `experiments/electrocatalysis/features.py` first. Writes `results/electro/...` |
| [plots.py](plots.py) | All figures, written to `results/plots/` (see [Figures](#figures)) |

`results/` is not in the repo. Everything in it is regenerated by the scripts.

## Run

From the repo root:

```bash
export PYTHONPATH=.
.venv/bin/python experiments/explainability/cv.py                    # CPU models, ~40 min on 20 cores (longer if other jobs share the CPU)
.venv/bin/python experiments/explainability/cv.py --models tabpfn --tables ecfp rf_best brr_best --targets capacity   # GPU, ~5–15 min per table depending on CPU load
.venv/bin/python experiments/explainability/summarize.py
.venv/bin/python experiments/explainability/explain.py --config capacity:rf_best:random_forest capacity:brr_best:random_forest \
    capacity:ecfp:bayes capacity:ecfp:random_forest size_resid:ecfp:random_forest size_resid:estate:random_forest
.venv/bin/python experiments/explainability/feature_checks.py
.venv/bin/python experiments/electrocatalysis/features.py && .venv/bin/python experiments/explainability/electro_shap.py
.venv/bin/python experiments/explainability/plots.py
```

## Figures

All figures are written to `results/plots/` by [plots.py](plots.py).

| figure | what it shows |
|---|---|
| `1_models.png` | **A:** held-out R² of the top configurations (dot = mean of 10 repeats, bar = range), with each one's 95th percentile under a shuffled target and the size-only baseline. **B:** how strongly each top model's prediction tracks heavy-atom count, and what is left once size is partialled out |
| `2_size.png` | Capacitance against heavy-atom count, with a separate fit for CTFs and for the other COFs |
| `3_shap_capacity.png` | SHAP beeswarms of RF on EState + functional groups and of RF on ECFP. Colour is the feature value; `· size` marks size proxies |
| `4_beyond_size.png` | Mean \|SHAP\| of RF trained on the size-corrected capacitance, coloured by direction |
| `5_feature_checks.png` | Model-free check: size-corrected capacitance with vs without each substructure, with bootstrap 95% intervals |
| `6_pred_vs_true.png` | Held-out predictions against measured capacitance: size only vs the best model |
| `7_shap_electro.png` | SHAP beeswarm of the J model (RF on ECFP, held-out paper). Left = better catalyst |

## Setup

- **Data:** `cof_matched/data_substrate_cofs.csv` and the precomputed feature tables in `cof_matched/rth_0.99/features/`, the same as the main pipeline.
- **Feature tables:**

  | table | content |
  |---|---|
  | `size` | heavy-atom count only |
  | `descriptor` | the custom descriptor: diameter, mol. weight, heteroatoms, rotatable bonds, C/N/O % |
  | `custom_pattern` | the custom patterns |
  | `estate` | EState sums |
  | `functional_groups` | RDKit `fr_*` counts |
  | `maccs` | MACCS keys |
  | `ecfp` | ECFP4 counts |
  | `rf_best` | EState + functional groups |
  | `brr_best` | descriptor + ECFP + EState + functional groups |

- **Targets:**
  - `capacity`: `capacity_max` as it is.
  - `size_resid`: capacitance minus a straight-line fit on heavy-atom count, which asks whether a model learns anything beyond size. The fit uses all rows; it has 2 parameters, so the leak into CV is negligible.
- **Models:** BRR, Lasso (`LassoCV`), RF, XGBoost, KNN, TabPFN (GPU) and a median baseline, all with fixed hyperparameters. There is no tuning, because a shuffled-target null must go through exactly the same procedure as the real run, and Optuna inside 100 permutations would be too slow. The repo's tuned numbers are in `notebooks/main_experiment_matrices.ipynb`.
- **CV:** repeated stratified 5-fold (10 repeats) on capacity binned into 5 groups. Every model and target uses the same folds. Metrics are computed on the pooled predictions of each repeat.
- **Null:** the rows of the target are shuffled, and the folds are stratified on the *shuffled* capacity, the same way the real run stratifies on the real one. The identity permutation reproduces the real run's first 2 repeats. Both the null and the real statistic are averaged over 2 repeats, and p = (1 + k) / (1 + n_perm). The smallest possible p is 0.0099 (0.032 for TabPFN). A first version reused the real-capacity folds for the shuffled targets, which favoured the real run (the median model got p = 0.04); this is fixed.
- **SHAP:** `TreeExplainer` for RF and XGBoost, `LinearExplainer` for BRR and Lasso. Every fold model explains only its own test rows, so each value is for a substrate the model didn't see. `top10_freq` is the share of the 50 fold fits in which a feature is in the top 10, as a measure of stability. `direction` > 0 means more of the feature raises the predicted capacitance.

## Reading the feature names

| feature | meaning |
|---|---|
| ECFP 849 | aromatic CH carbon (radius 0) |
| ECFP 356 | aromatic C with no H: ring fusion or a substituent |
| ECFP 726 / 875 | aromatic CH with its neighbours (radius 1) |
| ECFP 136 | aromatic C bonded to another aryl ring (biaryl) |
| ECFP 675 | aromatic S |
| ECFP 378 | aromatic N with no H (pyridine/triazine type) |
| ECFP 807 | hydroxyl O |
| ECFP 116 | imine N, aryl–N=C |
| EState `aaCH`, `aasC`, `aaN`, `aaS`, `sOH`, `sNH2`, `dO` | aromatic CH, substituted aromatic C, aromatic N, aromatic S, OH, NH₂, =O |

`importance.csv` lists the substructures behind every ECFP bit.

## Caveats

- **114 substrates and no tuning.** The R² values are small, and their spread across CV repeats is about 0.02–0.04.
- **Size dominates the importance rankings.** The models and SHAP can't separate "large" from the many features correlated with large (aromatic C, benzene rings, MW, diameter). Read those features as one size factor, not as separate effects.
- **BRR spreads importance thinly over hundreds of correlated ECFP bits.** No bit holds more than about 1%, so its per-feature SHAP is unstable (no top-10 feature appears in ≥ 80% of folds). Read its direction only together with RF.
- **The chemistry hints were found by looking at SHAP.** The p-values in `feature_checks.csv` are descriptive, not confirmatory. Thiophene rests on 14 substrates.
- **No paper information for capacitance.** Unlike J, there is no paper column for the capacitance data, so a lab or measurement effect that is confounded with chemistry (for example, one group making the thiophene COFs) can't be ruled out.
- **J:** 39 materials in multi-material papers, and the configuration was chosen as the best of about 480 (see `../electrocatalysis/README.md`).
