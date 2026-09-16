"""Reject unexpected publication files. This is not a semantic privacy guarantee."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    ".gitignore", "README.md", "PUBLICATION.md", "pyproject.toml",
    "src/image_classification/__init__.py", "src/image_classification/__main__.py",
    "src/image_classification/data.py", "src/image_classification/evaluation.py",
    "src/image_classification/cnn.py", "tests/test_workflow.py", "scripts/check_release.py",
    "docs/CODE_MAP.md",
    "docs/METHODS.md",
    "matlab/analyse_band_importance.m",
    "matlab/analyse_pca_tsne.m",
    "matlab/analyse_spectra.m",
    "matlab/analyse_tsne_time_trajectory.m",
    "matlab/check_data_quality.m",
    "matlab/classify_by_time.m",
    "matlab/config.m",
    "matlab/evaluate_source_level.m",
    "matlab/extract_feature_tables.m",
    "matlab/import_feature_table.m",
    "matlab/inspect_mat_files.m",
    "matlab/load_cached_data.m",
    "matlab/make_grouped_folds.m",
    "matlab/run_model_study.m",
    "matlab/run_pipeline.m",
    "matlab/validate_across_domains.m",
    "matlab/validate_across_time.m",
    "tests/run_matlab_tests.m",
}


def audit():
    problems = []
    found = set()
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(p in {".git", ".venv", "__pycache__", ".pytest_cache"} or p.endswith(".egg-info") for p in rel.parts):
            continue
        if path.is_symlink():
            problems.append(f"Symbolic link: {rel.as_posix()}")
            continue
        if not path.is_file():
            continue
        name = rel.as_posix()
        found.add(name)
        if name not in ALLOWED:
            problems.append(f"Unexpected file: {name}")
            continue
        content = path.read_text(encoding="utf-8")
        # Generic checks only: manual review must also cover domain identifiers.
        for pattern in (r"[A-Za-z]:\\Users\\", r"/content[/]drive/", r"-----BEGIN [A-Z ]+ PRIVATE KEY-----",
                        r"gh[pousr]_[A-Za-z0-9]{20,}"):
            if re.search(pattern, content):
                problems.append(f"Potential private content: {name}")
    problems.extend(f"Missing file: {name}" for name in sorted(ALLOWED - found))
    return problems


if __name__ == "__main__":
    issues = audit()
    if issues:
        raise SystemExit("\n".join(issues))
    print(f"Release audit passed: {len(ALLOWED)} source and documentation files.")
