# Implementation map

This release includes substantial adapted MATLAB source, plus a smaller Python refactor. It is not an unchanged reproduction of the original research project. Identifiers, labels, schemas, and private I/O have been generalized throughout. This document deliberately does not name the underlying dataset, application, people, or original project directories.

| Public module | Origin and retained substance |
| --- | --- |
| `matlab/inspect_mat_files.m` | Adapted original implementation: v7.3/HDF5 introspection, object references, field dimensions, record sampling, channel consistency, repeated-source checks |
| `matlab/extract_feature_tables.m` | Adapted original implementation: lightweight mean-feature reads, preallocation, orientation checks, patch/observation tables and caching |
| `matlab/check_data_quality.m` | Adapted original implementation: duplicate keys, missingness, source consistency, uneven patch counts, class/time/domain coverage and confounding |
| `matlab/analyse_spectra.m` | Adapted original implementation: source-averaged channel profiles, standard errors, confidence intervals, class differences and time comparisons |
| `matlab/analyse_pca_tsne.m` | Adapted original implementation: PCA scores/loadings/variance, t-SNE, metadata views, repeated-source connections and subgroup diagnostics |
| `matlab/analyse_tsne_time_trajectory.m` | Adapted original implementation: one shared embedding per domain, time-specific class centroids and descriptive distances |
| `matlab/make_grouped_folds.m` | Adapted original implementation: source-level class balancing, reproducible assignment and fold-separation checks |
| `matlab/classify_by_time.m` | Adapted original implementation: grouped nested linear/RBF SVM evaluation for each time, OOF predictions and fold metrics |
| `matlab/evaluate_source_level.m` | Adapted original implementation: patch OOF aggregation, majority votes, mean scores, comparison of observation and patch metrics |
| `matlab/validate_across_domains.m` | Adapted original implementation: leave-one-domain-out training-side tuning and source-overlap checks |
| `matlab/validate_across_time.m` | Adapted original implementation: source-disjoint time transfer and an explicitly separate repeated-source mode |
| `matlab/analyse_band_importance.m` | Adapted original implementation: effect size, Welch testing, FDR, SVM coefficients and held-out permutation importance |
| `matlab/run_model_study.m` | Adapted original integrated study: model comparison, multiple aggregation levels, training-side transfer selection, permutation ranking and top-k comparison; data loaders replaced and evaluation revised |
| `matlab/load_cached_data.m` | Adapted original cache-loading and schema-validation implementation; channels are now configurable |
| `matlab/run_pipeline.m` | Adapted original step controller, logging, cache checks and generated-file tracking |
| `matlab/config.m`, `matlab/import_feature_table.m` | New generic configuration and private table adapter replacing application-specific inputs |
| `src/image_classification/evaluation.py` | Refactored from the model-selection notebook; includes nested model-family selection |
| `src/image_classification/cnn.py` | Adapted architectures from the 1D/2D CNN notebooks, wrapped in a generic private-input interface |
| Other Python modules, tests, docs and release tools | New packaging, validation and publication support |

## Deliberate exclusions

- Original datasets, images, fitted models, notebook outputs, plots and reports.
- Dataset-specific file discovery, acquisition channel values, experiment exclusions and identifier parsers.
- A report generator containing fixed application-specific conclusions.
- Early teaching notebooks and a simpler random-row SVM example superseded by source-grouped evaluation.
- A separate channel-overview script whose plotting functionality is already represented in the retained analysis modules.

No historical result is claimed to be reproduced by these adapted implementations. See `METHODS.md` for substantive changes to evaluation.
