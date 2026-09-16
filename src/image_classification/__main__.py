"""Command-line entry point; private outputs must stay outside the project."""
import argparse
import json
from pathlib import Path
from .data import channel_means, load_data
from .evaluation import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["classical", "cnn1d", "cnn2d"], default="classical")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    project = Path(__file__).resolve().parents[2]
    if output.is_relative_to(project):
        parser.error("Output must be outside the project tree")
    if output.exists():
        parser.error("Output already exists; choose a new private filename")
    if not output.parent.is_dir():
        parser.error("Output parent directory must already exist")
    images, y, groups = load_data(args.data)
    if args.model == "classical":
        result = evaluate(channel_means(images), y, groups, args.outer_folds, args.inner_folds, args.seed)
    else:
        from .cnn import evaluate_cnn
        result = evaluate_cnn(images, y, groups, args.model, args.epochs, args.seed)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Evaluation complete. Output saved to the requested private location.")


if __name__ == "__main__":
    main()
