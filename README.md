# Image Classification with Group-Aware Evaluation

A code-only MATLAB and Python project for binary classification of image patches. The MATLAB workflow preserves the main analysis implementations: feature extraction, quality checks, exploratory analysis, grouped classification, transfer evaluation, interpretability, and feature-subset studies. The Python package provides a smaller classifier/CNN interface.

This repository presents a generalized implementation of an image-analysis workflow. It contains no research dataset, sample images, fitted weights, experiment outputs, or performance claims. Tests generate artificial inputs in memory; they verify software behavior, not predictive performance.

## Methods

- HDF5/MAT structure inspection, lightweight feature caching, and patch-to-observation aggregation.
- Missing-value, duplicate-key, repeated-measurement, class-coverage, and domain-confounding checks.
- Channel profiles with source-level uncertainty, PCA, t-SNE, and trajectories in a shared embedding.
- Leave-one-domain-out validation and cross-time transfer, with a separate optional repeated-source mode.
- Welch tests, Cohen's d, Benjamini-Hochberg FDR, linear SVM coefficients, and permutation importance.
- A MATLAB model-comparison study including boosted trees, multi-level voting, and nested top-k feature evaluation.
- Channel-mean features extracted from image patches.
- Linear and RBF support vector machines, logistic regression, random forests, and extra trees.
- Nested group-aware cross-validation: both model family and hyperparameters are selected inside the training portion of each outer fold.
- Scaling fitted within each training fold through scikit-learn pipelines.
- Optional 1D CNN over ordered channel features and 2D CNN over image patches.
- Patch-level and group-level balanced accuracy, F1, and accuracy, calculated locally only.

See [implementation map](docs/CODE_MAP.md) for what was retained versus rewritten, and [methods and limitations](docs/METHODS.md) for evaluation details. There are no reported benchmark results.

## MATLAB analysis workflow

Requires MATLAB and Statistics and Machine Learning Toolbox. The artificial-input workflow checks are run with:

```matlab
addpath('tests');
run_matlab_tests;
```

For private use, start by configuring an existing directory outside this repository:

```matlab
addpath('matlab');
config(privateRoot, size(X, 2));
import_feature_table(X, meta);
run_pipeline("quality");
run_pipeline("spectra");
run_pipeline("embedding");
run_pipeline("classify");
run_pipeline("cross_domain");
run_pipeline("cross_time");
run_pipeline("importance");
run_pipeline("trajectory");
run_model_study();
```

Here `X` is an N-by-C private patch-feature matrix, with generic channel columns. `meta` is a table with `analysis_label` (`class0`/`class1`), `subject_key`, `observation_key`, `domain`, and numeric nonnegative `time`. Supply globally unique independent-source identifiers; repeated observations share a `subject_key`. All metadata within one observation must agree. These variables are user-provided; no data file is distributed.

The alternative HDF5 path uses `inspect_mat_files` and `extract_feature_tables`; see the [input contract](docs/METHODS.md#matlab-input-contract). All MATLAB caches, figures, predictions, and tables are written to the configured private directory. Several original analysis functions replace their previous outputs when rerun; use a fresh private directory to retain each run. `run_model_study` refuses to overwrite its output directory.

## Python installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
```

For CNN training, additionally install `python -m pip install -e ".[cnn]"`. TensorFlow is optional and hardware support depends on your platform.

## Python private input contract

Prepare an NPZ file outside the repository, containing exactly these arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `images` | `(N, H, W, C)` | Finite numeric image patches, channels last |
| `labels` | `(N,)` | Integer binary labels, `0` and `1` |
| `groups` | `(N,)` | Nonempty string identifiers for independent source objects |

Use globally unique group identifiers. All patches, repeat captures, and related observations of one source object must share its group. Each group must have one consistent label. The loader rejects object arrays and does not load pickled objects. No dataset-specific conversion or identifier mapping is included.

```bash
python -m image_classification --data /absolute/private/input.npz --output /absolute/private/evaluation.json
python -m image_classification --data /absolute/private/input.npz --output /absolute/private/cnn.json --model cnn2d --epochs 50
```

Use `--model cnn1d` for ordered channel-mean features, or the default `--model classical` for nested model selection. The output path is mandatory and must be outside the project tree. Existing output files are not overwritten. Outputs contain local metrics and configuration; no images, predictions, group IDs, or fitted weights are saved. Do not upload generated outputs.

## Python evaluation design and limitations

Classical evaluation uses three outer folds and two inner folds by default. Every fold must contain both classes and disjoint groups; insufficient or unsuitable data raise an error rather than silently falling back to a weaker evaluation. Fold counts are configurable with `--outer-folds` and `--inner-folds`.

The CNN path uses one seeded group-level train/validation/test split. Channel normalization is fitted on training images only. Early stopping uses validation loss, and the test set is evaluated after training. A fixed split may be unsuitable for a small dataset and will fail if a class is missing. Sample proportions depend on group sizes.

Classical model selection optimizes patch-level balanced accuracy. Group metrics use majority voting, with ties assigned to class 1; large groups therefore receive more weight during model fitting and selection than during group-level reporting. CNN probabilities are thresholded at 0.5. A 1D convolution assumes meaningful channel order. Channel means discard spatial information; the 2D model retains it. Group separation alone does not establish generalization to a new acquisition setting or population.

## Repository layout

```text
src/image_classification/   Data validation, features, evaluation, CNNs, CLI
matlab/                    Adapted original analysis implementations and private I/O
docs/                      Methodology and implementation provenance
tests/                     In-memory software checks
scripts/check_release.py    Allowlist-based release audit
PUBLICATION.md             Publication boundary and release instructions
```

Run `python scripts/check_release.py` before publication. Only publish this standalone directory or an audited archive of it. A `.gitignore` cannot protect sensitive content embedded inside an allowed source file or remove files already committed to Git history.

No license grant is included. Add a license only after confirming the right to license all code being released.
