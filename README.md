# REST-Guard

Implementation of **Causal rest-referenced alignment with bounded updates for cross-subject motor-imagery EEG decoding**.

## Installation

Use Python 3.11 or 3.12 for installation. Run from the repository root in an activated environment.

```bash
python -m pip install -r requirements.txt
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Use [BCI Competition IV dataset 2a](https://www.bbci.de/competition/iv/#dataset2a) and the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/). The common task is left/right motor imagery using channels C3, Cz and C4. Select source, validation and held-out subjects before epoch extraction. A subject cannot occur in two partitions of the same dataset.

Create a JSON manifest listing EDF, GDF or FIF recordings. Supply actual channel names in C3/Cz/C4 order:

```json
{
  "recordings": [
    {
      "path": "raw/A01T.gdf",
      "events": "events/A01T.json",
      "dataset": "BCI2a",
      "subject": "A01",
      "session": "train_session",
      "time_offset": 0,
      "split": "train",
      "channels": ["EEG-C3", "EEG-Cz", "EEG-C4"]
    }
  ]
}
```

The manifest must include the chosen train, validation and test subjects. Paths are relative to the manifest. Channel strings must match the actual release; the example is a schema illustration, not a guarantee of names in every GDF reader.

For multiple files in one chronological session, assign their supported session identity and a nonnegative `time_offset` in seconds. Event starts/ends remain local to each recording; the offset places them on the common session clock. This allows an earlier baseline recording to precede later imagery runs when the acquisition protocol supports that order. Otherwise keep independent sessions separate; adaptation resets at each session.

Each recording's event file is a JSON list:

```json
[
  {"kind": "rest", "start": 5.0, "end": 25.0, "protocol_rest": true},
  {"kind": "task", "start": 30.0, "end": 32.0, "label": 0},
  {"kind": "task", "start": 40.0, "end": 42.0, "label": 1}
]
```

Use real event timing from the dataset protocol. Labels 0/1 mean left/right imagery. BCI 2a task event codes and PhysioNet T1/T2 annotations must be interpreted using their task/run definitions: T1/T2 do not have one universal left/right meaning across every run. Rest is eligible only when explicitly supported by the recording protocol and marked `protocol_rest: true`. A pre-cue gap alone is insufficient. For held-out 2a sessions, supply the released evaluation labels in the annotation file for scoring; the adaptation code never reads them during its updates.

For a cross-dataset study, train/validate on the declared source cohort and put the selected external subjects in test. Use a separate source training job for each within-cohort held-out-subject protocol. Keep the actual subject set in the manifest rather than assuming repository totals equal eligible trial counts.

```bash
python -m restguard prepare --manifest data/manifest.json --output data/eeg.npz --fs 128 --duration 2 --low 4 --high 40
```

Preparation converts MNE-loaded signals to microvolts, selects the three channels, applies a causal fourth-order Butterworth bandpass, and harmonizes sampling with causal rational FIR resampling. Filtering/resampling retain their causal delay; no backward filter or delay compensation imports future samples. The fixed passband, 128-Hz grid and two-second windows are implementation conventions for the paper's unspecified harmonization details.

Task events contribute their first complete requested window. Protocol-supported rest intervals are tiled into nonoverlapping windows. Every window retains its actual start/completion time. The NPZ stores `x [N,3,T]`, labels, event kind, timing, rest eligibility, dataset/subject/session, partition and event IDs. The source reference and technical thresholds are fitted later from training data only.

## Training and inference

```bash
python -m restguard train --data data/eeg.npz --output outputs/source --seed 0
```

## Evaluation

```bash
python -m restguard evaluate --data data/eeg.npz --checkpoint outputs/source/source.pt --method restguard --output outputs/restguard
python -m restguard evaluate --data data/eeg.npz --checkpoint outputs/source/source.pt --method source --output outputs/source_only
python -m restguard evaluate --data data/eeg.npz --checkpoint outputs/source/source.pt --method causal_ea --output outputs/causal_ea
```

Only eligible rest completed before a trial may update its alignment.

## Comparisons and ablations

```bash
python -m restguard evaluate --data data/eeg.npz --checkpoint outputs/source/source.pt --method restguard --contamination 0.2 --seed 0 --output outputs/restguard_c20
python -m restguard evaluate --data data/eeg.npz --checkpoint outputs/source/source.pt --method causal_ea --contamination 0.2 --seed 0 --output outputs/ea_c20
```

Further methods: `no_quality`, `no_bound`, `task_covariance`, `unfrozen` and transductive `whole_session_ea`.

## Outputs

`outputs/source/source.pt` stores source weights/reference/thresholds. Each evaluation writes `metrics.json`, `predictions.json` and `alignment.json`.

```bash
python -m restguard compare --metrics outputs/restguard/metrics.json outputs/causal_ea/metrics.json outputs/restguard_c20/metrics.json outputs/ea_c20/metrics.json --source outputs/source_only/metrics.json --output outputs/comparison.json
python -m restguard plot --comparison outputs/comparison.json --output outputs/figures
```
