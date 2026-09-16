# Publication boundary

This is a standalone, code-only release. No original notebooks, notebook outputs, reports, data, figures, acquisition metadata, model artifacts, or prior Git history are included. Class names and input identifiers are generic. The original research workflow has been refactored; this package does not reproduce or establish any research finding.

## Before uploading

1. Run `python -m pytest`, the MATLAB `run_matlab_tests` workflow where available, and `python scripts/check_release.py`.
2. Review the source and README for private information, including comments and strings.
3. Upload only this directory's approved source files to a new repository. Never upload the enclosing workspace or original project folder.
4. Keep private inputs and newly generated evaluation files outside the repository.
5. Review `git diff --cached` and `git ls-files` before every commit. Ignore rules are not an access-control mechanism.

The audit rejects any unexpected file except known local Python caches, environment files under `.venv`, and package installation metadata. Release archives must be built from the explicit source allowlist, not by zipping a working tree recursively.

The optional CNN implementation requires TensorFlow. Base tests exercise data validation, grouped evaluation, and the CLI using artificial arrays only. Tests requiring TensorFlow skip when it is unavailable; a skipped test does not establish CNN runtime compatibility.

MATLAB tests generate private temporary inputs and outputs outside the repository, including temporary v7.3/HDF5 files. They do not use the original dataset. The public implementation map names only generalized modules. Any separate local provenance map containing original paths must remain outside the release.
