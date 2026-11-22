# CTF substrate split experiment

> **Frozen.** The numbers below were produced on 2026-10-24 with the substrate split from before PR #4. That PR changed the splitter and regenerated `substrate_split/output/substrate_split.csv`, so rerunning these scripts will give different numbers. The conclusion was confirmed independently by PR #3 (tuned RF and BRR, best feature sets). The current work on the main/side split is in [../electrocatalysis/](../electrocatalysis/).

Does splitting a substrate into **main chain** and **side chains** help predict COF capacitance (`capacity_max`)?
This follows up the meeting task to compare three ways of encoding a substrate, using the same representation and the same models for each:

| approach | features |
|---|---|
| `a1_full` | `F(full substrate)` |
| `a2_main_side` | `F(main chain) ‖ F(side chains)` |
| `a3_all` | `F(full) ‖ F(main) ‖ F(side)` |

The evaluation uses only the 29 CTF (nitrile) substrates, because all of them split cleanly with no flags (see [substrate_split/README.md](../../substrate_split/README.md)).

**Related:** PR #3 (`notebooks/main_experiment_matrices.ipynb`) runs the same comparison with tuned RF and BRR on the best feature sets from greedy selection. Its conclusion agrees with this one. This experiment adds MACCS, TabPFN, KNN, XGBoost and Lasso, SHAP importance per block, and paired per-repeat statistics.

## Setups

Both setups use **the same CTF test folds**, so their results can be compared pair by pair.

1. **CTF-only** ([config.yaml](config.yaml) → `results/`, [analysis.ipynb](analysis.ipynb)): train and test on the 29 CTFs only.
2. **All COFs** ([config_all_cofs.yaml](config_all_cofs.yaml) → `results_all_cofs/`, [analysis_all_cofs.ipynb](analysis_all_cofs.ipynb)): test on the CTFs. The training set is the CTF training fold plus all 85 non-CTF COFs. Non-CTFs are never in a test set.

## Method

- **Data:** [substrate_split/output/substrate_split.csv](../../substrate_split/output/substrate_split.csv). Side chains are stored as fragments joined by `.`, with `*` marking attachment points. A substrate with no side chain gets an all-zero side block.
- **Representations:** ECFP4 counts (1024 bits) and MACCS counts. Each part is encoded the same way and the column names carry a `full__`, `main__` or `side__` prefix. After concatenation, the same variance filter as the main pipeline is applied (`rth=0.99`).
- **Feature counts after filtering:**

  | setup | representation | a1 | a2 | a3 |
  |---|---|---|---|---|
  | CTF-only | ECFP | 215 | 250 | 465 |
  | CTF-only | MACCS | 65 | 89 | 154 |
  | All COFs | ECFP | 364 | 428 | 792 |
  | All COFs | MACCS | 110 | 171 | 281 |

- **Models:** BRR (`bayes`), Lasso, Random Forest, XGBoost, KNN and TabPFN (default settings, CPU). No hyperparameter tuning, because each fold has only about 23 CTF training samples. MLP and Chemprop are skipped.
- **CV:** repeated stratified 5-fold (5 folds × 10 repeats) over the CTFs, stratified on capacity binned into 5 groups. Each repeat predicts every CTF exactly once. Metrics are computed on the pooled predictions of each repeat, which gives 10 paired values per configuration.
- **Paired comparison:** Wilcoxon signed-rank test on the per-repeat differences against `a1_full`. The repeats reuse the same molecules, so these p-values are optimistic.
- **Interpretability:**
  - SHAP: tree explainer for RF and XGBoost, linear explainer for BRR, kernel explainer for KNN (CTF-only setup only).
  - Lasso: absolute coefficients.
  - TabPFN: no SHAP, because it is too slow on CPU.
  - Importance is summed per block (`full`, `main`, `side`).

## Results

The tables show the mean over 10 repeats. Lower RMSE is better; higher Spearman is better.

### Main and side chain split (ECFP, RMSE)

| model | CTF-only a1 | a2 | a3 | All COFs a1 | a2 | a3 |
|---|---|---|---|---|---|---|
| TabPFN | **146.1** | 155.8 | 150.3 | 157.2 | 162.0 | 158.1 |
| Random Forest | 155.8 | 160.4 | 158.8 | **152.3** | 155.8 | 155.9 |
| BRR | 160.2 | 167.1 | 162.3 | 154.0 | 159.0 | 155.5 |
| XGBoost | 165.0 | 203.1 | 165.8 | 156.4 | 184.2 | 172.3 |
| KNN | 162.8 | 168.8 | 165.2 | 163.4 | 168.4 | 161.9 |
| Lasso | 157.3 | 180.5 | 153.5 | 190.7 | 200.5 | 180.7 |

Median-of-train baseline RMSE: **167** for CTF-only and 188 for all COFs. The all-COFs baseline is higher because its training median includes the non-CTFs. Use 167 as the reference in both setups.

### Main and side chain split (MACCS, RMSE)

| model | CTF-only a1 | a2 | a3 | All COFs a1 | a2 | a3 |
|---|---|---|---|---|---|---|
| TabPFN | 171.8 | 167.2 | 168.8 | 169.5 | 168.5 | 166.8 |
| Random Forest | 186.6 | 168.4 | 175.3 | 181.6 | 161.2 | 176.0 |
| BRR | 163.8 | 174.3 | 169.8 | 167.0 | 164.2 | 163.8 |
| XGBoost | 213.6 | 177.2 | 201.5 | 198.4 | 170.8 | 191.6 |
| KNN | 184.8 | 170.0 | 176.3 | 179.4 | 160.9 | 165.9 |

Lasso on MACCS is broken by its untuned default `alpha=1.0` (CTF-only RMSE 430–535), so it is left out of this table.

### Training on all COFs vs CTFs only (ECFP, `a1_full`)

| model | RMSE, CTF-only | RMSE, all COFs | Spearman, CTF-only | Spearman, all COFs |
|---|---|---|---|---|
| TabPFN | 146.1 | 157.2 | 0.35 | 0.35 |
| Random Forest | 155.8 | 152.3 | 0.30 | 0.42 |
| BRR | 160.2 | 154.0 | 0.18 | 0.40 |
| XGBoost | 165.0 | 156.4 | 0.27 | 0.45 |

### What the models "see"

- **Block share in `a2`:** the main-chain block carries most of the importance. Its share is 45–82% in the CTF-only setup and 68–88% in the all-COFs setup. The side-chain block carries 18–55% in the CTF-only setup and 13–32% in the all-COFs setup.
- **Block share in `a3`:** the tree models and Lasso rely mostly on the `full` block. XGBoost and Lasso put up to about 95% of their importance there in the CTF-only setup.
- **Top features:** the same ECFP bits lead whether they are computed on the full substrate or on the main chain. CTF-only examples are bits 102 and 305; all-COFs examples are 849, 356 and 875. Between 6 and 9 of the top 15 features are shared between `a1` and the main block of `a2`.

## Conclusions

1. **Splitting doesn't help with ECFP.** In both setups, `a2` is worse than `a1` for every model (by 3–38 RMSE), and `a3` roughly ties `a1`. The loss comes from the 11 CTFs that have side chains.
2. **Splitting helps with MACCS, but MACCS is the weaker representation.** `a2` beats `a1` for most models, by up to 36 RMSE and 0.4 Spearman. Even so, MACCS `a2` only reaches the baseline and doesn't beat ECFP `a1`. A likely reason is that side-chain atoms set MACCS keys that have nothing to do with the main chain, so separating them into their own block removes that noise rather than adding new information. This fits the gain showing up mostly on CTFs *without* side chains.
3. **The data is still the bottleneck.** The best R² is about 0.18 (TabPFN, ECFP, CTF-only). Every model is close to the median baseline.
4. **Adding non-CTF COFs helps ranking more than error.** Spearman rises by about 0.1–0.2 for BRR, RF and XGBoost, while RMSE drops by only 3–10. TabPFN is the exception and gets worse.

## Caveats

- 18 of the 29 CTFs have no side chain, so the three approaches differ only on 11 molecules.
- No hyperparameter tuning was done. Lasso results are unreliable.
- The p-values are optimistic because the repeated CV folds share data.
- The all-COFs setup includes the substrates whose split has a flag (16 `large_side_chain` and 3 `no_ends_fallback_diameter`). They appear only in training.

## Reproduce

Run from the repo root:

```bash
export PYTHONPATH=.
.venv/bin/python experiments/ctf_substrate_split/build_features.py --config_path experiments/ctf_substrate_split/config.yaml
.venv/bin/python experiments/ctf_substrate_split/train.py --config_path experiments/ctf_substrate_split/config.yaml
# then run analysis.ipynb; repeat with config_all_cofs.yaml and analysis_all_cofs.ipynb
```

`train.py --models bayes lasso` limits the run to the listed models. Finished runs, meaning those that already have a `predictions.csv`, are skipped.

Approximate runtime on an M1 Pro (CPU):

| setup | BRR, Lasso, RF, XGBoost | KNN and TabPFN |
|---|---|---|
| CTF-only | about 1.5 min | about 18 min |
| All COFs (everything together) | about 74 min, mostly TabPFN | |

Set `tabpfn.device: cuda` in the config to run TabPFN on the 3090.

## Possible next steps

- Run on the imine substrates, where side chains are more common. This needs a check of the flagged splits first.
- Try MOF datasets that have several target properties, to test whether the split matters for some properties and not others.
- Tune Lasso (`alpha`) before using it in any comparison.
