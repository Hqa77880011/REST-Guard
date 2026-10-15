import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, firwin, sosfilt, upfirdn
from math import gcd


def prepare(manifest, output, fs=128, duration=2, low=4, high=40):
    import mne

    if fs < 1 or duration <= 0 or round(fs * duration) < 32:
        raise ValueError("Positive sampling rate and epochs of at least 32 samples required")
    path = Path(manifest).resolve()
    records = json.loads(path.read_text(encoding="utf-8"))["recordings"]
    roles = {}
    samples = []
    for record in records:
        identity = (record["dataset"], record["subject"])
        if identity in roles and roles[identity] != record["split"]:
            raise ValueError("Subjects must be separated into train/val/test before epoching")
        roles[identity] = record["split"]
        if record["split"] not in ("train", "val", "test"):
            raise ValueError("Split must be train, val or test")
        source = path.parent / record["path"]
        if source.suffix.lower() == ".edf":
            raw = mne.io.read_raw_edf(source, preload=True, verbose=False)
        elif source.suffix.lower() == ".gdf":
            raw = mne.io.read_raw_gdf(source, preload=True, verbose=False)
        elif source.suffix.lower() == ".fif":
            raw = mne.io.read_raw_fif(source, preload=True, verbose=False)
        else:
            raise ValueError("Use EDF, GDF or FIF recordings")
        if len(record["channels"]) != 3:
            raise ValueError("Supply actual source channel names in C3,Cz,C4 order")
        raw.pick(record["channels"])
        raw.reorder_channels(record["channels"])
        values = raw.get_data() * 1e6
        native_fs = float(raw.info["sfreq"])
        if abs(native_fs - round(native_fs)) > 1e-6 or not 0 < low < high < min(native_fs, fs) / 2:
            raise ValueError("Integer native/target sampling rates and a valid passband are required")
        sos = butter(4, [low, high], btype="bandpass", fs=native_fs, output="sos")
        values = sosfilt(sos, values, axis=1)
        divisor = gcd(int(native_fs), fs)
        up, down = fs // divisor, int(native_fs) // divisor
        if up != down:
            taps = firwin(10 * max(up, down) + 1, 1 / max(up, down), window=("kaiser", 5)) * up
            size = int(values.shape[1] * fs / native_fs)
            values = upfirdn(taps, values, up=up, down=down, axis=1)[:, :size]
        event_path = path.parent / record["events"]
        events = json.loads(event_path.read_text(encoding="utf-8"))
        time_offset = float(record.get("time_offset", 0))
        if not np.isfinite(time_offset) or time_offset < 0:
            raise ValueError("Recording time offsets must be finite and nonnegative")
        for event_index, event in enumerate(events):
            if event["kind"] not in ("task", "rest"):
                raise ValueError("Event kind must be task or rest")
            start, end = float(event["start"]), float(event["end"])
            if not 0 <= start < end <= values.shape[1] / fs:
                raise ValueError("Event outside the recording")
            supported = bool(event.get("protocol_rest", False))
            if event["kind"] == "task" and event.get("label") not in (0, 1):
                raise ValueError("Task labels are left/right class IDs 0 and 1")
            starts = [start] if event["kind"] == "task" else np.arange(start, end - duration + 1e-8, duration)
            for position in starts:
                if position + duration > end + 1e-8:
                    raise ValueError("Task event is shorter than the requested epoch duration")
                first = int(round(position * fs))
                length = int(round(duration * fs))
                x = values[:, first:first + length]
                if x.shape[1] != length or not np.isfinite(x).all():
                    raise ValueError("Invalid EEG samples in an epoch")
                samples.append((x.astype(np.float32), event.get("label", -1), event["kind"],
                                float(position + time_offset), float(position + duration + time_offset), supported,
                                record["dataset"], record["subject"], record["session"], record["split"],
                                f"{record['dataset']}:{record['subject']}:{record['session']}:{source.stem}:{event_index}:{position:.6f}"))
    if not samples:
        raise ValueError("No eligible task/rest epochs in the manifest")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = ("y", "kind", "start", "end", "protocol_rest", "dataset", "subject", "session", "split", "id")
    payload = {name: np.array([row[index + 1] for row in samples]) for index, name in enumerate(names)}
    if len(np.unique(payload["id"])) != len(samples):
        raise ValueError("Duplicate recording/session/event identities in manifest")
    np.savez_compressed(destination, x=np.stack([row[0] for row in samples]), fs=fs, **payload)


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("x", "y", "kind", "start", "end", "protocol_rest",
                "dataset", "subject", "session", "split", "id")}
        data["fs"] = int(archive["fs"])
    if data["x"].ndim != 3 or data["x"].shape[1] != 3 or not np.isfinite(data["x"]).all():
        raise ValueError("Expected finite C3/Cz/C4 epochs [N,3,T] in microvolts")
    if set(data["split"]) != {"train", "val", "test"}:
        raise ValueError("All train/val/test subjects are required")
    for dataset, subject in set(zip(data["dataset"], data["subject"])):
        index = (data["dataset"] == dataset) & (data["subject"] == subject)
        if len(np.unique(data["split"][index])) != 1:
            raise ValueError("Subject leakage across splits")
    return data
