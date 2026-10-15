# REST-Guard

Implementation of **Causal rest-referenced alignment with bounded updates for cross-subject motor-imagery EEG decoding**.

REST-Guard trains an EEGNet source decoder, then adapts a covariance-alignment front end using only eligible resting segments completed before each task trial. The decoder stays frozen. A source-centered Frobenius constraint bounds accumulated logarithmic covariance displacement. Outputs include balanced accuracy, per-subject accuracy differences and negative-transfer counts.

## Install

Use Python 3.11 or 3.12:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate `.venv` before installation (`source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1` in PowerShell). Install the appropriate CUDA PyTorch build first if needed. Training and evaluation accept `--device cpu` or `--device cuda`.

## 1. Obtain and describe EEG recordings

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

## 2. Prepare harmonized epochs

```bash
python -m restguard prepare --manifest data/manifest.json --output data/eeg.npz --fs 128 --duration 2 --low 4 --high 40
```

Preparation converts MNE-loaded signals to microvolts, selects the three channels, applies a causal fourth-order Butterworth bandpass, and harmonizes sampling with causal rational FIR resampling. Filtering/resampling retain their causal delay; no backward filter or delay compensation imports future samples. The fixed passband, 128-Hz grid and two-second windows are implementation conventions for the paper's unspecified harmonization details.

Task events contribute their first complete requested window. Protocol-supported rest intervals are tiled into nonoverlapping windows. Every window retains its actual start/completion time. The NPZ stores `x [N,3,T]`, labels, event kind, timing, rest eligibility, dataset/subject/session, partition and event IDs. The source reference and technical thresholds are fitted later from training data only.

## 3. Train the source decoder

```bash
python -m restguard train --data data/eeg.npz --output outputs/source --seed 0 --device cuda
```

EEGNet uses eight temporal filters, depth multiplier two, 16 separable filters, ELU, batch normalization, pooling and dropout 0.25. Source trials are whitened by the inverse square root of the mean shrunk source-rest covariance. Adam optimizes source cross-entropy; separate validation-subject loss selects `source.pt`. Spatial and classifier weights use EEGNet-style norm constraints.

| Argument | Default | Meaning |
|---|---:|---|
| `--epochs`, `--patience`, `--batch-size` | 300, 30, 64 | Source training budget |
| `--lr` | 0.001 | Source Adam learning rate |
| `--shrinkage` | 0.1 | Covariance identity shrinkage |
| `--epsilon` | 0.000001 | Positive covariance diagonal floor in squared microvolts |
| `--update-rate` | 0.1 | Target reference exponential update coefficient |
| `--radius` | 0.5 | Fixed source-centered logarithmic-state radius |
| `--amplitude-factor` | 1.5 | Multiplier on the 99th percentile of source-rest channel peak amplitudes |
| `--derivative-factor` | 1.5 | Corresponding sample-difference threshold multiplier |

Technical eligibility checks require finite samples, at least one second of data, channel amplitudes below source thresholds and no excessive sample jump. The checks use source rest only and do not consume target task labels. `source.pt` includes weights, covariance reference, quality thresholds, source epoch dimensions and source training settings. Fix shrinkage, radius and thresholds using source/development subjects before evaluating held-out users.

## 4. Execute causal adaptation and comparisons

```bash
python -m restguard evaluate --data data/eeg.npz --checkpoint outputs/source/source.pt --method restguard --output outputs/restguard --device cuda
python -m restguard evaluate --data data/eeg.npz --checkpoint outputs/source/source.pt --method source --output outputs/source_only --device cuda
python -m restguard evaluate --data data/eeg.npz --checkpoint outputs/source/source.pt --method causal_ea --output outputs/causal_ea --device cuda
```

Each target subject/session starts with zero log displacement. Before a trial, the queue consumes only references with `reference_end < trial_start`. Eligible rest covariance is source-whitened, symmetrized, eigenvalue-floored and logged. The exponential update is projected around the fixed zero origin, and `exp(-A/2)` corrects the source-whitened task trial. No rest means the state stays unchanged.

| Method | Behavior |
|---|---|
| `restguard` | Quality-gated past rest, source-centered projection, frozen EEGNet |
| `source` | The same source-whitened EEGNet with no target reference updates |
| `causal_ea` | Quality-gated past-rest Euclidean covariance average |
| `whole_session_ea` | Whole-session rest covariance reference, a separately named transductive comparator |
| `no_quality` | Skips amplitude/derivative eligibility while retaining protocol/timing restrictions |
| `no_bound` | Removes the source-centered projection |
| `task_covariance` | Uses only completed past task epochs, without their labels |
| `unfrozen` | Adds post-prediction task entropy steps to the decoder |

`unfrozen` uses SGD at `--adapt-lr 0.00001`, with batch-normalization statistics in evaluation mode. Prediction always precedes that trial's parameter update. It is a distinct target-parameter adaptation regime. Decoder architecture and initial weights are identical across the listed comparisons. CSP-LDA, EEG Conformer, ResTL and SAR are separate systems in the paper; the commands above cover the common-decoder comparisons and component ablations implemented here.

## 5. Reference-contamination study

```bash
python -m restguard evaluate --data data/eeg.npz --checkpoint outputs/source/source.pt --method restguard --contamination 0.2 --seed 0 --output outputs/restguard_c20 --device cuda
python -m restguard evaluate --data data/eeg.npz --checkpoint outputs/source/source.pt --method causal_ea --contamination 0.2 --seed 0 --output outputs/ea_c20 --device cuda
```

Repeat fractions 0/0.1/0.2/0.3/0.4 with identical seeds. The explicit corruption is a random rank-one channel mixing operation followed by global energy restoration. `--contamination-strength` defaults to 3. Every method receives the same corrupted reference epochs for the same input/reference regime and seed. Causal methods still consume them only after completion; the whole-session comparator uses its declared future access. The contamination operator is a controlled implementation choice, not a fabricated measurement of a particular physiological artifact.

## 6. Metrics and visualization

Evaluation writes `metrics.json`, `predictions.json` and `alignment.json`. Metrics include per-subject ordinary and balanced accuracy, dataset-specific subject-mean balanced accuracy and the overall subject-balanced mean. Predictions retain exact trial IDs, probabilities, covariance-state norm and its transformation bound. The alignment trace lists each consumed reference's completion time beside the trial start time and eligibility result.

For bounded log updates, the reported perturbation bound is `exp(norm(A)/2)-1`. Euclidean comparisons report the corresponding exact correction spectral displacement. These transformation quantities do not replace measured decoding accuracy.

```bash
python -m restguard compare --metrics outputs/restguard/metrics.json outputs/causal_ea/metrics.json outputs/restguard_c20/metrics.json outputs/ea_c20/metrics.json --source outputs/source_only/metrics.json --output outputs/comparison.json
python -m restguard plot --comparison outputs/comparison.json --output outputs/figures
```

Comparison requires exactly the same trial IDs and ground-truth labels as the source-only run. A harmed subject has lower ordinary accuracy than that source model on those trials. The output records accuracy/balanced-accuracy differences and harmed-subject counts separately per dataset, using the actual number of evaluated subjects. Figures plot balanced accuracy against reference contamination.

Source training, adaptation outputs, recordings and checkpoints remain outside Git through `.gitignore`. The repository's CI installs dependencies and checks source compilation/imports without training or processing EEG data.
