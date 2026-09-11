"""Signal detection sweep: 9 modes x in-band SNR grid, plus false alarms.

This is the A03 evidence producer for the P1 detector: it prints the
detection success-rate curve (mode x SNR) and the parameter-estimation
errors at the reference SNR, all measured against the generator truth
through the frozen ``detect_result_v1`` contract.

Usage::

    .venv/bin/python benchmarks/detect_sweep.py
    .venv/bin/python benchmarks/detect_sweep.py --snr -5,0,5,10,15,20 --json out.json
"""
import argparse
import json
import time

import numpy as np

from signal_analysis.core_api import detect_signals, generate_iq
from signal_analysis.evaluation import evaluate_detections, signal_truth

RATE_HZ = 1e6

#: One scenario per generator mode: the nine modes of ``MODES``.
SCENARIOS = {
    "am": {"offset": -200e3, "bandwidth": 80e3},
    "fm": {"offset": 150e3, "bandwidth": 120e3},
    "ssb": {"offset": -350e3, "bandwidth": 60e3, "side": "usb"},
    "ask2": {"offset": 250e3, "bandwidth": 100e3},
    "qpsk": {"offset": 0.0, "bandwidth": 200e3},
    "qam16": {"offset": -100e3, "bandwidth": 300e3},
    "qam64": {"offset": 100e3, "bandwidth": 300e3},
    "fh_rc": {"offset": 0.0, "bandwidth": 400e3},
    "fh_video": {"offset": 0.0, "bandwidth": 300e3},
}
MODES = tuple(SCENARIOS)


def run_case(mode, snr_db, seeds, duration, nfft):
    """Detect one (mode, SNR) cell over ``seeds`` realisations."""
    runs = []
    for seed in range(seeds):
        spec = {"mode": mode, "power_dbfs": -10.0, **SCENARIOS[mode]}
        samples, summary = generate_iq(RATE_HZ, duration, [spec],
                                       noise={"snr_db": snr_db}, seed=seed + 1)
        started = time.perf_counter()
        result, _ = detect_signals(samples, RATE_HZ, {"nfft": nfft})
        elapsed = time.perf_counter() - started
        truth = signal_truth(summary)
        metrics = evaluate_detections(truth, result["detections"])
        runs.append({"metrics": metrics, "wall_s": elapsed, "nfft": nfft,
                     "detections": result["detections"]})
    return runs


def accuracy_ok(pair, bin_width):
    """True when one matched pair satisfies the P1 estimation gates.

    Centre gate: 2 FFT bins or 2% of the truth bandwidth, whichever is
    larger (a hopping session is located from the channels that were
    actually visited, so the error scales with the band, not the bin).
    Bandwidth gate: 20%. In-band SNR gate: 2 dB.
    """
    width = pair.get("truth_bandwidth_hz") or 0.0
    band_error = pair.get("bandwidth_relative_error")
    snr_error = pair.get("snr_error_db")
    if band_error is None or snr_error is None:
        return False
    return (abs(band_error) <= 0.20
            and abs(snr_error) <= 2.0
            and abs(pair.get("center_error_hz") or 0.0)
            <= max(2.0 * bin_width, 0.02 * width))


def summarise(runs):
    """Aggregate per-realisation metrics into one cell summary."""
    bin_width = RATE_HZ / runs[0].get("nfft", 512) if runs else 1.0
    success = sum(1 for run in runs
                  if run["metrics"]["missed"] == 0
                  and run["metrics"]["false_alarm"] == 0)
    errors = [pair for run in runs for pair in run["metrics"]["pairs"]]
    accurate = [pair for pair in errors if accuracy_ok(pair, bin_width)]
    return {
        "trials": len(runs),
        "success": success,
        "rate": success / len(runs) if runs else 0.0,
        "matched": len(errors),
        "accurate": len(accurate),
        "accuracy_rate": len(accurate) / len(errors) if errors else None,
        "center_mae_hz": float(np.mean([abs(p["center_error_hz"]) for p in errors]))
        if errors else None,
        "bandwidth_mape": float(np.mean([abs(p["bandwidth_relative_error"]) for p in errors]))
        if errors else None,
        "snr_mae_db": float(np.mean([abs(p["snr_error_db"]) for p in errors]))
        if errors else None,
        "wall_s": float(np.mean([run["wall_s"] for run in runs])) if runs else None,
    }


def noise_only(trials, duration, nfft, power_dbfs=-10.0):
    """False-alarm behaviour on pure noise (no signal present)."""
    detections = []
    for seed in range(trials):
        samples, _ = generate_iq(RATE_HZ, duration, [],
                                 noise={"power_dbfs": power_dbfs}, seed=1000 + seed)
        result, _ = detect_signals(samples, RATE_HZ, {"nfft": nfft})
        detections.append(len(result["detections"]))
    rate = sum(1 for count in detections if count) / len(detections)
    return {"trials": trials, "runs_with_detection": sum(1 for c in detections if c),
            "false_alarm_rate": rate, "max_detections": max(detections) if detections else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snr", default="-5,0,5,10,15,20",
                        help="comma separated in-band SNR grid in dB")
    parser.add_argument("--seeds", type=int, default=3, help="realisations per cell")
    parser.add_argument("--duration", type=float, default=0.5, help="record length in seconds")
    parser.add_argument("--nfft", type=int, default=512)
    parser.add_argument("--reference-snr", type=float, default=10.0)
    parser.add_argument("--noise-trials", type=int, default=20)
    parser.add_argument("--json", help="write the full result matrix here")
    args = parser.parse_args()

    grid = [float(value) for value in args.snr.split(",") if value.strip()]
    report = {"sample_rate_hz": RATE_HZ, "duration_s": args.duration,
              "nfft": args.nfft, "seeds": args.seeds, "snr_grid": grid,
              "cells": {}, "scenarios": SCENARIOS}
    print(f"rate={RATE_HZ:.0f} Hz duration={args.duration}s nfft={args.nfft} "
          f"seeds={args.seeds} trials={len(grid) * len(MODES) * args.seeds}")
    print("\n检测成功率 [%]  (mode x in-band SNR dB)\n")
    header = "  mode      " + "".join(f"{snr:>7.0f}" for snr in grid)
    print(header)
    for mode in MODES:
        row = []
        for snr in grid:
            cell = summarise(run_case(mode, snr, args.seeds, args.duration, args.nfft))
            report["cells"][f"{mode}@{snr:g}"] = cell
            row.append(f"{cell['rate'] * 100:>7.0f}")
        print(f"  {mode:<9s} " + "".join(row))

    print(f"\n检测与参数估计 @ {args.reference_snr:g} dB\n")
    print("  mode      检出率 [%]  精度达标率 [%]  center/B [%]  bandwidth [%]  SNR [dB]  wall [ms]")
    for mode in MODES:
        cell = report["cells"].get(f"{mode}@{args.reference_snr:g}")
        if not cell:
            continue
        width = SCENARIOS[mode]["bandwidth"]
        center_pct = (cell["center_mae_hz"] / width * 100.0
                      if cell["center_mae_hz"] is not None else float("nan"))
        accuracy = cell["accuracy_rate"]
        print(f"  {mode:<9s} {cell['rate'] * 100:>9.0f}  "
              f"{(accuracy * 100 if accuracy is not None else float('nan')):>13.0f}  "
              f"{center_pct:>11.3f}  "
              f"{(cell['bandwidth_mape'] or 0) * 100:>12.3f}  "
              f"{(cell['snr_mae_db'] or 0):>8.3f}  {cell['wall_s'] * 1e3:>9.1f}")

    report["noise_only"] = noise_only(args.noise_trials, args.duration, args.nfft)
    print(f"\n纯噪声虚警: {report['noise_only']['runs_with_detection']}/"
          f"{report['noise_only']['trials']} 次出现目标 "
          f"(虚警率 {report['noise_only']['false_alarm_rate'] * 100:.1f}%)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"已写入 {args.json}")


if __name__ == "__main__":
    main()
