import argparse
import json
from pathlib import Path

import numpy as np

from .data import prepare
from .run import METHODS, evaluate, train


def compare(paths, source_path, output):
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    reference = {(row["dataset"], row["subject"]): row for row in source["subjects"]}
    with (Path(source_path).parent / "predictions.json").open(encoding="utf-8") as stream:
        source_trials = {row["trial_id"]: row["y"] for row in json.load(stream)}
    rows = []
    for path in paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        with (Path(path).parent / "predictions.json").open(encoding="utf-8") as stream:
            trials = {row["trial_id"]: row["y"] for row in json.load(stream)}
        if trials != source_trials:
            raise ValueError("Negative transfer requires the exact same labeled test trials")
        subjects = []
        for row in result["subjects"]:
            key = (row["dataset"], row["subject"])
            subjects.append(dict(**row, accuracy_difference=row["accuracy"] - reference[key]["accuracy"],
                                 balanced_accuracy_difference=row["balanced_accuracy"] - reference[key]["balanced_accuracy"],
                                 harmed=row["accuracy"] < reference[key]["accuracy"]))
        datasets = {}
        for dataset in sorted({row["dataset"] for row in subjects}):
            selected = [row for row in subjects if row["dataset"] == dataset]
            datasets[dataset] = dict(balanced_accuracy=float(np.mean([row["balanced_accuracy"] for row in selected])),
                                     harmed_subjects=sum(row["harmed"] for row in selected), subjects=len(selected))
        rows.append(dict(method=result["config"]["method"], contamination=result["config"]["contamination"],
                         datasets=datasets, subjects=subjects))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def plot(path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    for dataset in sorted({name for row in rows for name in row["datasets"]}):
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in sorted({row["method"] for row in rows}):
            selected = sorted([row for row in rows if row["method"] == method and dataset in row["datasets"]],
                              key=lambda row: row["contamination"])
            ax.plot([row["contamination"] for row in selected],
                    [row["datasets"][dataset]["balanced_accuracy"] for row in selected], marker="o", label=method)
        ax.set(xlabel="Reference contamination fraction", ylabel="Subject-mean balanced accuracy", title=dataset)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(destination / f"{dataset}_contamination.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(prog="python -m restguard")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare")
    data.add_argument("--manifest", required=True)
    data.add_argument("--output", required=True)
    data.add_argument("--fs", type=int, default=128)
    data.add_argument("--duration", type=float, default=2)
    data.add_argument("--low", type=float, default=4)
    data.add_argument("--high", type=float, default=40)
    source = commands.add_parser("train")
    source.add_argument("--data", required=True)
    source.add_argument("--output", required=True)
    source.add_argument("--device", default="cpu")
    source.add_argument("--seed", type=int, default=0)
    source.add_argument("--epochs", type=int, default=300)
    source.add_argument("--patience", type=int, default=30)
    source.add_argument("--batch-size", type=int, default=64)
    source.add_argument("--lr", type=float, default=0.001)
    source.add_argument("--shrinkage", type=float, default=0.1)
    source.add_argument("--epsilon", type=float, default=0.000001)
    source.add_argument("--update-rate", type=float, default=0.1)
    source.add_argument("--radius", type=float, default=0.5)
    source.add_argument("--amplitude-factor", type=float, default=1.5)
    source.add_argument("--derivative-factor", type=float, default=1.5)
    target = commands.add_parser("evaluate")
    target.add_argument("--data", required=True)
    target.add_argument("--checkpoint", required=True)
    target.add_argument("--output", required=True)
    target.add_argument("--method", choices=METHODS, default="restguard")
    target.add_argument("--device", default="cpu")
    target.add_argument("--seed", type=int, default=0)
    target.add_argument("--contamination", type=float, default=0)
    target.add_argument("--contamination-strength", type=float, default=3)
    target.add_argument("--adapt-lr", type=float, default=0.00001)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--metrics", nargs="+", required=True)
    comparison.add_argument("--source", required=True)
    comparison.add_argument("--output", default="outputs/comparison.json")
    visual = commands.add_parser("plot")
    visual.add_argument("--comparison", required=True)
    visual.add_argument("--output", default="outputs/figures")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.output, args.fs, args.duration, args.low, args.high)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "compare":
        compare(args.metrics, args.source, args.output)
    else:
        plot(args.comparison, args.output)


if __name__ == "__main__":
    main()
