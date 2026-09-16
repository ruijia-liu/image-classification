# Methods and interpretation

## Unit of independence

Patches are not automatically independent samples. The source object identified by `subject_key` is the splitting unit. Repeated captures must use the same source identifier, including across domains and times. Observation IDs identify one capture, not an independent source. IDs must never be constructed from the class label to hide an inconsistent source assignment.

The MATLAB cache has patch rows and observation rows. Observation features summarize the patches using mean, standard deviation and median. Most original analysis modules use observation means. The larger model study uses patch features and reports patch, observation (`object`), source, and group aggregation. With this generic schema, source and group are the same level; they are not independent pieces of evidence. Its retained selection score averages source, observation and group accuracy, thereby weighting source accuracy twice. The Python interface instead selects using patch-level balanced accuracy. These two objectives must not be treated as identical.

## Validation and transfer

- `classify_by_time`: nested group-aware tuning for each of two fixed SVM families. Comparing their outer scores is exploratory model comparison; the maximum is not an unbiased estimate of a subsequently selected family.
- `validate_across_domains`: leave one domain out, with tuning confined to the remaining domains and a source-overlap check. A domain boundary alone is not an independence guarantee.
- `validate_across_time`: strict mode removes test sources seen during training. The optional longitudinal mode allows repeat sources and answers a different question: transfer to another time for possibly known sources. Do not combine the modes as one estimate of unseen-source generalization.
- `run_model_study`: hyperparameters are selected inside each outer training partition. Its model-ranking table is exploratory. Cross-set model selection uses only the training set; shared test sources are removed. Top-k evaluation repeats model selection, feature ranking and tuning inside each outer training fold. Choosing the best k using these reported scores would require another independent evaluation.
- Python classical CLI: selects both model family and hyperparameters inside each outer training partition. Python CNNs use a fixed group-disjoint train/validation/test split, with early stopping on validation loss.

Insufficient-data behavior is not uniform across the retained MATLAB routines: folds may be reduced, unsupported subsets skipped, or inner tuning may log a fallback. Inspect private warning logs and actual fold counts. The Python classical path fails explicitly for invalid class coverage. No original performance number is included.

## Interpretation

Channel profiles average sources before estimating uncertainty. Welch tests and Cohen's d compare source-averaged values within each time; Benjamini-Hochberg adjusts valid p-values across channels within that time, not globally across all analyses. Univariate tests remain exploratory and require their distributional assumptions to be assessed.

The linear-SVM module summarizes coefficients of standardized features across grouped folds. Coefficients reflect model dependence and channel correlation, not a causal effect. Its signed coefficients follow MATLAB's class ordering; use the explicit class order before interpreting direction.

Permutation importance measures the change in held-out predictive scores after shuffling a feature. Correlated channels and within-source correlations affect interpretation. The integrated study's full-subset importance ranking describes a model chosen using that subset and is exploratory. Its top-k path computes a fresh training-only ranking in each outer fold, rather than reusing that global ranking.

PCA and t-SNE are exploratory visualizations fitted to the analyzed subset, not predictors evaluated on held-out data. A shared t-SNE embedding supports within-embedding visual time comparisons; t-SNE distances, centroid movement and apparent class separation do not quantify original-space effect sizes or predictive generalization. Separately fitted embeddings do not share an axis system.

OOF aggregation supports majority voting and mean positive-class scores. The MATLAB aggregation default threshold is zero for SVM decision margins. Set `analysis.score_threshold` to 0.5 only when the supplied scores are calibrated positive-class probabilities. A margin is not a probability. The generic Python path uses hard-label group voting; its CNN patch threshold is 0.5.

## Publication revisions

1. Replaced application-specific labels, input names, metadata fields and physical channel values with generic labels, domains, times and channel indices.
2. Replaced data discovery with private configuration and a generic table adapter; removed embedded study-specific report conclusions.
3. Made the integrated study's hyperparameter tuning training-only in every outer fold; the earlier implementation could tune once using the full subset.
4. Replaced reuse of a global feature ranking for top-k scoring with training-only ranking and model selection inside outer folds.
5. Removed overlap between training and test sources in integrated cross-set transfer.
6. Treated SVM outputs as decision scores; added rejection of mixed-fold or duplicate-patch aggregation.
7. Used direct confusion-count F1 computation, source-level univariate averaging, explicit source identifiers, and rejection of impossible grouped folds.

Other algorithmic structures are retained where possible. None of these changes implies that the private original experiments were rerun.

## MATLAB input contract

The recommended entry point is `import_feature_table(X, meta)` after `config(privateRoot, C)`:

| Input | Contract |
| --- | --- |
| `X` | Finite numeric N-by-C patch-feature matrix; C >= 2 |
| `meta.analysis_label` | `class0` or `class1` |
| `meta.subject_key` | Globally unique source-object ID, unchanged across repeat captures |
| `meta.observation_key` | Unique capture ID, shared by the patches from that capture |
| `meta.domain` | Generic acquisition-domain ID |
| `meta.time` | Finite nonnegative numeric time index |

Optional HDF5 inspection/extraction expects private `class0.mat` and `class1.mat` v7.3 files under the configured root. Each contains `imageRecords`, a cell array of scalar structures with `class`, `domain`, `time`, `object_number`, `object_name`, `patch_size`, `total_patch_number`, `channels`, `locations_yx`, and `mean_values`. Use `object_name` as the globally unique source ID, `channels = 1:C`, `mean_values` as patches-by-channels, and `locations_yx` as patches-by-2. The generic extractor permits one observation per source/domain/time; use the table adapter for multiple captures at the same time. Only trusted local MAT files should be used.

The original optional patch branch of `classify_by_time` performs capped sampling diagnostics; it does not produce patch-level OOF predictions. `evaluate_source_level` therefore requires an explicit private OOF prediction file. Do not substitute in-sample predictions or observation-level predictions for patch votes. The Python classifier API or a separate grouped patch-classification workflow can be used to create suitable predictions locally.

## Checks

Python tests use generated arrays. MATLAB workflow checks generate private temporary features and v7.3 records, run extraction, quality checks, profiles, embeddings, classification, transfer, importance, score aggregation, model comparison and top-k analysis, then delete their temporary directory. The compact integrated-study check uses linear SVM and logistic regression; optional CNN tests require TensorFlow. Software checks do not establish predictive performance or universal MATLAB-version compatibility.
