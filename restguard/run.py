import copy
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .alignment import RestGuard, covariance, symmetric_function
from .data import load
from .model import EEGNet


METHODS = ("restguard", "source", "causal_ea", "whole_session_ea", "no_quality",
           "no_bound", "task_covariance", "unfrozen")


def quality(x, amplitude, derivative, minimum):
    return bool(x.shape[1] >= minimum and np.isfinite(x).all()
                and np.all(np.max(np.abs(x), axis=1) <= amplitude)
                and np.all(np.max(np.abs(np.diff(x, axis=1)), axis=1) <= derivative))


def train(args):
    if not 0 <= args.shrinkage <= 1 or not 0 < args.update_rate <= 1 or args.radius <= 0:
        raise ValueError("Invalid covariance/update settings")
    if min(args.epochs, args.batch_size, args.patience) < 1:
        raise ValueError("Positive training budgets required")
    if min(args.epsilon, args.lr, args.amplitude_factor, args.derivative_factor) <= 0:
        raise ValueError("Positive covariance regularization, learning rate and quality multipliers required")
    data = load(args.data)
    source = data["split"] == "train"
    rest = source & (data["kind"] == "rest") & data["protocol_rest"]
    task = source & (data["kind"] == "task")
    validation = (data["split"] == "val") & (data["kind"] == "task")
    if not rest.any() or not task.any() or not validation.any():
        raise ValueError("Training requires source task/rest and separate validation task epochs")
    if set(data["y"][task]) != {0, 1} or set(data["y"][validation]) != {0, 1}:
        raise ValueError("Source and validation task sets must each contain both classes")
    reference = np.mean([covariance(x, args.shrinkage, args.epsilon) for x in data["x"][rest]], axis=0)
    whitener = symmetric_function(reference, lambda value: value ** -0.5)
    x = (whitener @ data["x"]).astype(np.float32)
    source_amplitudes = np.max(np.abs(data["x"][rest]), axis=2)
    source_derivatives = np.max(np.abs(np.diff(data["x"][rest], axis=2)), axis=2)
    amplitude = np.quantile(source_amplitudes, 0.99, axis=0) * args.amplitude_factor
    derivative = np.quantile(source_derivatives, 0.99, axis=0) * args.derivative_factor
    torch.manual_seed(args.seed)
    model = EEGNet(x.shape[-1]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[task]), torch.tensor(data["y"][task], dtype=torch.long)),
                        batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    best, stale = np.inf, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch, y in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch.to(args.device)), y.to(args.device))
            loss.backward()
            optimizer.step()
            model.constrain()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(F.cross_entropy(model(torch.from_numpy(x[validation]).to(args.device)),
                                                     torch.tensor(data["y"][validation], dtype=torch.long, device=args.device)))
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation_loss=validation_loss))
        if validation_loss < best:
            best, stale = validation_loss, 0
            torch.save(dict(state=copy.deepcopy(model.state_dict()), reference=torch.from_numpy(reference),
                            amplitude=torch.from_numpy(amplitude), derivative=torch.from_numpy(derivative),
                            samples=x.shape[-1], fs=data["fs"], args=vars(args), epoch=epoch), destination / "source.pt")
        else:
            stale += 1
        print(json.dumps(history[-1]), flush=True)
        if stale >= args.patience:
            break
    (destination / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def evaluate(args):
    if not 0 <= args.contamination <= 1:
        raise ValueError("Contamination must be in [0,1]")
    data = load(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    settings = checkpoint["args"]
    if data["x"].shape[-1] != checkpoint["samples"]:
        raise ValueError("Checkpoint and evaluation epoch lengths differ")
    if data["fs"] != checkpoint["fs"]:
        raise ValueError("Checkpoint and evaluation sampling rates differ")
    reference = checkpoint["reference"].numpy()
    amplitude, derivative = checkpoint["amplitude"].numpy(), checkpoint["derivative"].numpy()
    rng = np.random.default_rng(args.seed)
    results = []
    trace = []
    test = data["split"] == "test"
    sessions = sorted(set(zip(data["dataset"][test], data["subject"][test], data["session"][test])))
    for dataset, subject, session in sessions:
        indices = np.flatnonzero(test & (data["dataset"] == dataset) & (data["subject"] == subject) & (data["session"] == session))
        trials = indices[data["kind"][indices] == "task"]
        trials = trials[np.argsort(data["start"][trials], kind="stable")]
        rest = indices[(data["kind"][indices] == "rest") & data["protocol_rest"][indices]]
        if args.method == "task_covariance":
            rest = trials.copy()
        rest = rest[np.argsort(data["end"][rest], kind="stable")]
        reference_segments = {}
        for row in rest:
            segment = data["x"][row].copy()
            if rng.random() < args.contamination:
                transform = np.eye(3) + args.contamination_strength * np.outer(rng.normal(size=3), rng.normal(size=3))
                altered = transform @ segment
                segment = altered * (np.linalg.norm(segment) / max(np.linalg.norm(altered), 1e-8))
            reference_segments[row] = segment
        model = EEGNet(checkpoint["samples"]).to(args.device)
        model.load_state_dict(checkpoint["state"])
        model.eval()
        model.requires_grad_(args.method == "unfrozen")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.adapt_lr) if args.method == "unfrozen" else None
        aligner = RestGuard(reference, settings["shrinkage"], settings["epsilon"], settings["update_rate"],
                            settings["radius"], bounded=args.method != "no_bound",
                            euclidean=args.method in ("causal_ea", "whole_session_ea"))
        if args.method == "whole_session_ea" and len(rest):
            covariances = [aligner.whitener @ covariance(reference_segments[i], settings["shrinkage"], settings["epsilon"]) @ aligner.whitener
                           for i in rest if quality(reference_segments[i], amplitude, derivative, data["fs"])]
            if covariances:
                aligner.average = np.mean(covariances, axis=0)
        pointer = 0
        for index in trials:
            while pointer < len(rest) and data["end"][rest[pointer]] < data["start"][index]:
                row = rest[pointer]
                segment = reference_segments[row]
                accepted = args.method == "no_quality" or quality(segment, amplitude, derivative, data["fs"])
                if accepted and args.method not in ("source", "whole_session_ea"):
                    aligner.update(segment)
                trace.append(dict(dataset=str(dataset), subject=str(subject), session=str(session),
                                  trial_id=str(data["id"][index]), reference_id=str(data["id"][row]),
                                  reference_end=float(data["end"][row]), trial_start=float(data["start"][index]),
                                  accepted=accepted, state_norm=float(np.linalg.norm(aligner.state, "fro"))))
                pointer += 1
            x = aligner.transform(data["x"][index])
            tensor = torch.from_numpy(x[None]).to(args.device)
            with torch.no_grad():
                probability = model(tensor).softmax(1).cpu().numpy()[0]
            results.append(dict(dataset=str(dataset), subject=str(subject), session=str(session),
                                trial_id=str(data["id"][index]), y=int(data["y"][index]),
                                prediction=int(probability.argmax()), probability=probability.tolist(),
                                state_norm=float(np.linalg.norm(aligner.state, "fro")),
                                perturbation_bound=aligner.displacement_bound()))
            if optimizer:
                optimizer.zero_grad()
                p = model(tensor).softmax(1)
                loss = -(p * p.clamp_min(1e-8).log()).sum(1).mean()
                loss.backward()
                optimizer.step()
                model.constrain()
    if not results:
        raise ValueError("No held-out task trials")
    subjects = []
    for identity in sorted({(r["dataset"], r["subject"]) for r in results}):
        rows = [r for r in results if (r["dataset"], r["subject"]) == identity]
        y, prediction = [r["y"] for r in rows], [r["prediction"] for r in rows]
        if set(y) != {0, 1}:
            raise ValueError(f"Subject {identity} needs both task classes")
        subjects.append(dict(dataset=identity[0], subject=identity[1],
                             accuracy=float(accuracy_score(y, prediction)),
                             balanced_accuracy=float(balanced_accuracy_score(y, prediction)), trials=len(rows)))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    payload = dict(config=vars(args), subjects=subjects,
                   subject_balanced_accuracy=float(np.mean([r["balanced_accuracy"] for r in subjects])),
                   datasets={dataset: float(np.mean([r["balanced_accuracy"] for r in subjects if r["dataset"] == dataset]))
                             for dataset in sorted({r["dataset"] for r in subjects})})
    (destination / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (destination / "predictions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (destination / "alignment.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
